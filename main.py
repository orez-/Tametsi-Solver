import itertools
import enum
import math
import time

import attr
import numpy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import pyautogui
import scipy.ndimage
import scipy.spatial

import ocr

REVEALED_TILES_COUNT_DOWN = False
COLUMN_HINTS_COUNT_DOWN = False

tile_ocr = ocr.get_ocr_api(psm=10, char_whitelist="0123456789?")


class Color(tuple, enum.Enum):
    BACKGROUND_COLOR = (0x14, 0x00, 0x23)
    COLORLESS_TILE_COLOR = (0x80, 0x80, 0x80)
    SAFE_TILE_COLOR = (0x33, 0x33, 0x33)
    FLAGGED_TILE_COLOR = (0xE6, 0xE6, 0xE6)


def color_is_close(color1, color2):
    return sum(abs(c2 - c1) for c1, c2 in zip(color1, color2)) < 30


class TileColorSet:
    def __init__(self):
        self.colors = [Color.COLORLESS_TILE_COLOR, Color.SAFE_TILE_COLOR, Color.FLAGGED_TILE_COLOR]

    def get_color(self, color):
        for known_color in self.colors:
            if color_is_close(color, known_color):
                return known_color
        color = tuple(map(int, color))
        self.colors.append(color)
        return color


class TileState(enum.Enum):
    flagged = 'flagged'
    safe = 'safe'
    unsolved = 'unsolved'


@attr.s(auto_attribs=True, eq=False)
class Tile:
    polygon: [(int, int)] = attr.ib(repr=False)
    color: (int, int, int)
    text: str
    tile_state: TileState
    mask_area: [] = attr.ib(repr=False)
    box: (slice, slice) = attr.ib(repr=False)

    @classmethod
    def new(cls, *, polygon, color, mask_area, box, text=None):
        state = TileState.unsolved
        if color == Color.SAFE_TILE_COLOR:
            state = TileState.safe
            color = None
        elif color == Color.FLAGGED_TILE_COLOR:
            state = TileState.flagged
            color = None
        return cls(
            polygon=polygon, color=color, text=text, tile_state=state, mask_area=mask_area, box=box
        )

    @property
    def click_point(self):
        """
        The point to click on the figure.

        Equivalent to the center of the tile. This is only valid if the
        tile shapes are all convex.
        """
        # TODO: centroid, not average.
        return (
            round(sum(x for x, _ in self.polygon) / len(self.polygon)),
            round(sum(y for _, y in self.polygon) / len(self.polygon)),
        )

    @property
    def is_flagged(self):
        return self.tile_state == TileState.flagged

    @property
    def is_safe(self):
        return self.tile_state == TileState.safe

    @property
    def is_unsolved(self):
        return self.tile_state == TileState.unsolved

    @property
    def number(self):
        return int(self.text) if self.text and self.text != "?" else None


@attr.s(auto_attribs=True)
class Constraint:
    min_mines: int
    max_mines: int

    def merge(self, min_mines, max_mines=None):
        """
        Can accept `min_mines` and `max_mines`, or a single `constraint`.
        """
        if max_mines is None:
            max_mines = min_mines.max_mines
            min_mines = min_mines.min_mines
        return Constraint(
            min_mines=max(min_mines, self.min_mines), max_mines=min(max_mines, self.max_mines)
        )

    def remove_mines(self, mines):
        return Constraint(
            min_mines=max(0, self.min_mines - mines), max_mines=self.max_mines - mines
        )


@attr.s(auto_attribs=True)
class Board:
    # The simulator currently (unfortunately) relies on this being an ordered list
    tiles: [Tile]
    _adjacencies: {Tile: {Tile}}
    constraints: {(Tile,): Constraint}
    color_set: "ColorSet"

    @classmethod
    def new(cls, tiles, adjacencies, color_set):
        tiles = list(tiles)
        board = Board(tiles=tiles, adjacencies=adjacencies, constraints={}, color_set=color_set)
        board.generate_constraints()
        return board

    def tile_at_mouse(self, x, y):
        for tile in self.tiles:
            if is_point_in_polygon((x, y), tile.polygon):
                return tile
        return None

    def generate_adjacency_constraint(self, tile):
        number = tile.number
        if number is None:
            return
        adjacencies = self._adjacencies[tile]
        # Have to decide how to interpret this number. If revealed tiles
        # do NOT count down, we need to drop the number of adjacent flags
        # from our constraint.
        if not REVEALED_TILES_COUNT_DOWN:
            number -= sum(1 for adj in adjacencies if adj.is_flagged)
        unsolved_neighbors = frozenset(adj for adj in adjacencies if adj.is_unsolved)
        constraint = Constraint(min_mines=number, max_mines=number)
        merge_or_add(self.constraints, unsolved_neighbors, constraint)

    def generate_constraints(self, color_count=None):
        for tile in self.tiles:
            self.generate_adjacency_constraint(tile)

    def constraint_for(self, tiles):
        tiles = frozenset(tiles)
        if tiles not in self.constraints:
            return Constraint(min_mines=0, max_mines=len(tiles))
        return self.constraints[tiles]

    def _apply_certainties(self):
        newly_solved = {}
        for tiles, constraint in self.constraints.items():
            if constraint.max_mines == 0:
                for tile in tiles:
                    newly_solved[tile] = TileState.safe
            elif constraint.min_mines == len(tiles):
                for tile in tiles:
                    newly_solved[tile] = TileState.flagged

        for tile, state in newly_solved.items():
            tile.tile_state = state

        self.remove_tiles_from_constraints(newly_solved.keys())
        return newly_solved

    def remove_tiles_from_constraints(self, newly_solved):
        """
        Remove the given tiles from all constraints.

        Once a tile is solved it doesn't need to factor into the constraints.
        This function applies the effect of these tiles on all current constraints.
        """
        new_constraints = {}
        for tiles, constraint in self.constraints.items():
            changed = newly_solved & tiles
            if changed:
                num_allocated = sum(1 for tile in changed if tile.is_flagged)
                # subtracting dict keys demotes a frozenset to a set :(
                tiles = frozenset(tiles - newly_solved)
                constraint = constraint.remove_mines(num_allocated)

            merge_or_add(new_constraints, tiles, constraint)
        self.constraints = new_constraints
        self._drop_vacuous_constraints()

    def _drop_vacuous_constraints(self):
        """
        Drop constraints expressing no information.

        ie, a constraint that states that x tiles contain between 0 and x mines.
        Thanks constraint, good tip.
        """
        self.constraints = {
            tiles: constraint
            for tiles, constraint in self.constraints.items()
            if not (constraint.min_mines == 0 and constraint.max_mines == len(tiles))
        }

    def _subdivide_overlapping_regions(self):
        # examine overlapping regions
        new_regions = {}
        # compare each region to each other
        combos = itertools.combinations(self.constraints.items(), 2)
        for (tiles1, constr1), (tiles2, constr2) in combos:
            overlap = frozenset(tiles1 & tiles2)
            if not overlap:
                continue

            tiles1_exclusive = frozenset(tiles1 - overlap)
            tiles2_exclusive = frozenset(tiles2 - overlap)
            # find the upper and lower bounds for each that can fit in the overlap.
            # commit as many as possible from both to the overlap
            overlap_max = min(len(overlap), constr2.max_mines, constr1.max_mines)
            # commit as few as possible, and fill them outside of the overlap first.
            overlap_min = max(
                constr2.min_mines - len(tiles2_exclusive),
                constr1.min_mines - len(tiles1_exclusive),
                0,
            )
            merge_or_add(
                new_regions, overlap, self.constraint_for(overlap).merge(overlap_min, overlap_max)
            )
            merge_or_add(
                new_regions,
                tiles1_exclusive,
                self.constraint_for(tiles1_exclusive).merge(
                    min_mines=constr1.min_mines - overlap_max,
                    max_mines=constr1.max_mines - overlap_min,
                ),
            )
            merge_or_add(
                new_regions,
                tiles2_exclusive,
                self.constraint_for(tiles2_exclusive).merge(
                    min_mines=constr2.min_mines - overlap_max,
                    max_mines=constr2.max_mines - overlap_min,
                ),
            )
        for tiles, constraint in new_regions.items():
            merge_or_add(self.constraints, tiles, constraint)
        self._drop_vacuous_constraints()


def merge_or_add(dict_, tiles, constraint):
    if tiles in dict_:
        constraint = dict_[tiles].merge(constraint)
    dict_[tiles] = constraint


def visualize_labeled(labeled):
    import pylab

    pylab.imshow(labeled, origin='lower', interpolation='nearest')
    pylab.colorbar()
    pylab.title("Labeled clusters")
    pylab.show()


def print_color(color, text=None):
    color = tuple(int(c) for c in color)
    if text is None:
        text = tuple(map(hex, color))
    print(f"\x1b[48;2;{color[0]};{color[1]};{color[2]}m{str(text)}\x1b[0m")


def expand_area(object_area, spacing):
    xs, ys = object_area
    return (
        slice(max(0, xs.start - spacing), xs.stop + spacing),
        slice(max(0, ys.start - spacing), ys.stop + spacing),
    )


def fetch_adjacencies(labeled, object_area, object_index, spacing=15):
    # TODO: this is super rough, definitely not always right
    expanded_area = expand_area(object_area, spacing)
    result = set(numpy.unique(labeled[expanded_area]))
    result -= {0, object_index}
    return result


def mask_only_white(array):
    lum = round(sum(array) / 3)
    return 255 - (lum if lum > 240 else 0)


def parse_tile_color(tile_pixels, color_set):
    color = numpy.median(tile_pixels, axis=0)
    return color_set.get_color(color)


def parse_tile(image, labeled, object_area, index, color_set):
    # limit both large arrays to just the area we care about,
    # then fetch the array of only the pixel values that are labeled with our index.
    # tile_pixels[i] = [r, g, b]
    figure_filter = labeled[object_area] == index
    tile_pixels = image[object_area][figure_filter]
    color = parse_tile_color(tile_pixels, color_set)

    shape = parse_polygon(object_area, figure_filter)
    adjacencies = fetch_adjacencies(labeled, object_area, index)
    text = None
    if color == Color.SAFE_TILE_COLOR:
        text = read_tile_number(image[object_area])

    return (
        Tile.new(polygon=shape, color=color, text=text, mask_area=figure_filter, box=object_area),
        adjacencies,
    )


def read_tile_number(image_slice):
    """
    Given a numpy slice of a tile, return the text from it.

    See https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
    for options + explanations.
    """
    mask = (
        (image_slice[:, :, 0] > 240)
        & (image_slice[:, :, 1] > 240)
        & (image_slice[:, :, 2] > 240)
    )
    labeled, _ = scipy.ndimage.label(mask)
    object_areas = scipy.ndimage.find_objects(labeled)
    filtered_areas = [
        (i, (ys, xs))
        for i, (ys, xs) in enumerate(object_areas, 1)
        # filter out areas too thin to be text
        if xs.stop - xs.start > 10 and ys.stop - ys.start > 10
        # filter out weird aspect ratios
        and 0.25 <= (xs.stop - xs.start) / (ys.stop - ys.start) <= 0.7
    ]
    result = None

    for index, area in filtered_areas:
        # question mark is two strokes, god help me.
        # need to manually connect the dot back to the character
        ys, xs = area
        dot_potentials = [
            (i, oys)
            for i, (oys, oxs) in enumerate(object_areas, 1)
            # roughly square
            if 0.9 <= (oxs.stop - oxs.start) / (oys.stop - oys.start) <= 1.1
            # relatively centered with our glyph
            and math.isclose(((xs.stop + xs.start) / 2), ((oxs.stop + oxs.start) / 2), abs_tol=1)
            # just below our glyph
            and ys.stop < oys.start <= ys.stop + 10
        ]
        if dot_potentials:
            # Could likely safely just call it a question mark now, but just to be safe..
            # For real though if there's more than one of these I quit.
            [(dot_index, oys)] = dot_potentials
            # extend the y slice down below the dot
            ys = slice(ys.start, oys.stop)
            area = (ys, xs)
            # note this _is_ going to poison the source, but.
            # i truly dont care.
            labeled[labeled == dot_index] = index

        # Ok here's the meat.
        area = expand_area(area, 3)
        subslice = numpy.copy(image_slice[area])
        # blank everything that's not our character
        subslice[labeled[area] != index] = [0, 0, 0]

        lum_mask = numpy.apply_along_axis(mask_only_white, 2, subslice)

        img = PIL.Image.fromarray(lum_mask.astype('uint8'), 'L')
        text = tile_ocr.image_to_string(img)

        if text:
            if result:
                print("uh found multiple, rip.", repr(text), repr(result))
                return ""
            result = text

    return result or None


def reparse_updated_tiles(board, updated_tiles, image):
    for tile in updated_tiles:
        tile.text = read_tile_number(image[tile.box])
        if tile.number is not None:
            board.generate_adjacency_constraint(tile)

    # need to check the other tiles: if we find a natural 0 it automatically expands.
    # we could BFS off of `updated_tiles` to minimize lookups, but who cares
    updated = set()
    for tile in board.tiles:
        if not tile.is_unsolved:
            continue
        tile_pixels = image[tile.box][tile.mask_area]
        color = parse_tile_color(tile_pixels, board.color_set)
        if color == Color.SAFE_TILE_COLOR:
            updated.add(tile)
            tile.tile_state = TileState.safe
            tile.text = read_tile_number(image[tile.box])
            if tile.number is not None:
                board.generate_adjacency_constraint(tile)
    board.remove_tiles_from_constraints(updated)


def parse_polygon(object_area, figure_filter):
    """
    Fetch the points of the polygon from the image data.
    """
    # Generate two grids the size of the object area, `range` style, sorta.
    ys, xs = numpy.mgrid[object_area]
    # Convert em to a grid of points.
    # grid[x, y] = [x, y], except offset by the start of object_area
    grid = numpy.concatenate((xs[..., numpy.newaxis], ys[..., numpy.newaxis]), axis=-1)
    figure = grid[figure_filter]
    hull = scipy.spatial.ConvexHull(figure)

    vertices = [(figure[vertex, 0], figure[vertex, 1]) for vertex in hull.vertices]
    return simplify_polygon(vertices)


def get_distance_from_line(p, p0, p1):
    base_distance = dist(p0, p1)
    area = get_triangle_area([p, p0, p1])
    return area * 2 / base_distance


def determinant(p, p0, p1):
    px, py = p
    p0x, p0y = p0
    p1x, p1y = p1
    return (p1x - p0x) * (py - p0y) - (p1y - p0y) * (px - p0x)


def is_point_in_polygon(p, vertices):
    return len({determinant(p, p0, p1) > 0 for p0, p1 in zip(vertices, rotate(vertices, 1))}) == 1


def get_triangle_area(vertices):
    (ax, ay), (bx, by), (cx, cy) = vertices
    return abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2


def dist(p, q):
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))


def is_short_triangle(vertices, tolerance=0.01):
    a, _, c = vertices
    base_distance = dist(a, c)
    area = get_triangle_area(vertices)
    altitude = area * 2 / base_distance
    print(altitude, base_distance)
    return (altitude / base_distance) < tolerance


def rotate(list_, rotate_amt):
    return list_[rotate_amt:] + list_[:rotate_amt]


def simplify_polygon(vertices):
    # TODO: actually figure out how to simplify the polygon
    return vertices


def get_tile_draw_color(tile):
    if tile.is_flagged:
        return Color.FLAGGED_TILE_COLOR
    if tile.is_safe:
        return Color.SAFE_TILE_COLOR
    return tile.color


def draw_board(board, highlighted=None):
    font = get_font()
    image = PIL.Image.new('RGB', (2400, 2160), (0, 0, 0))
    pdraw = PIL.ImageDraw.Draw(image)
    for tile in board.tiles:
        color = get_tile_draw_color(tile)
        if isinstance(highlighted, (int, Tile)):
            if isinstance(highlighted, int):
                focus, adj = list(board._adjacencies.items())[highlighted]
            else:
                focus = highlighted
                adj = board._adjacencies[highlighted]
            if tile == focus:
                color = (0, 0, 0xFF)
            if tile in adj:
                color = (0xFF, 0, 0)
        elif highlighted is not None:
            if tile in highlighted:
                color = (0xFF, 0, 0)
        pdraw.polygon(tile.polygon, fill=color)
        if tile.text:
            pdraw.text(
                (tile.click_point[0] - 20, tile.click_point[1] - 20),
                tile.text,
                font=font,
                fill=(255, 255, 255),
            )

    # # highlight vertices, for debug
    # for tile in board.tiles:
    #     for vertex in tile.polygon:
    #         pdraw.ellipse([vertex[0] - 3, vertex[1] - 3, vertex[0] + 3, vertex[1] + 3], fill=(0xFF, 0, 0))
    return image


def get_font():
    font_files = ['Arial.ttf', 'arial.ttf']
    for name in font_files:
        try:
            return PIL.ImageFont.truetype(name, 40)
        except OSError:
            pass
    raise RuntimeError("couldn't load a font")


def parse_board(image):
    mask = (
        (image[:, :, 0] != Color.BACKGROUND_COLOR[0])
        | (image[:, :, 1] != Color.BACKGROUND_COLOR[1])
        | (image[:, :, 2] != Color.BACKGROUND_COLOR[2])
    )
    # `labeled` creates an array[x, y] = idx
    labeled, _ = scipy.ndimage.label(mask)
    # visualize_labeled(labeled)
    object_areas = scipy.ndimage.find_objects(labeled)

    color_set = TileColorSet()
    adjacency_pointers = dict(
        parse_tile(image, labeled, object_area, index, color_set)
        for index, object_area in enumerate(object_areas, 1)
    )
    tiles = list(adjacency_pointers)

    adjacencies = {
        tile: {tiles[idx - 1] for idx in adjacencies}
        for tile, adjacencies in adjacency_pointers.items()
    }
    return Board.new(tiles=tiles, adjacencies=adjacencies, color_set=color_set)


def click_tiles(actions):
    for tile, state in actions.items():
        x, y = tile.click_point
        button = 'left' if state == TileState.safe else 'right'
        # TODO: yow
        pyautogui.click(x=x + 720, y=y, button=button)


def get_board_screenshot():
    image = pyautogui.screenshot()
    return get_board_area(image)


def get_board_area(image):
    image = image.convert('RGB')
    array = numpy.array(image)
    return array[:, 720:3120]


def solve_live_game():
    print("alright, switch to the game now")
    time.sleep(3)
    image = get_board_screenshot()
    board = parse_board(image)
    i = 100
    while i > 0:
        i -= 1
        board._subdivide_overlapping_regions()
        results = board._apply_certainties()
        if results:
            click_tiles(results)
            image = get_board_screenshot()
            updated_tiles = [tile for tile, state in results.items() if state == TileState.safe]
            reparse_updated_tiles(board, updated_tiles, image)
            i = 100
    return board


if __name__ == '__main__':
    solve_live_game()
