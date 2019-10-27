import collections
import enum

import attr
import numpy
import PIL.Image
import scipy.ndimage
import scipy.spatial


BACKGROUND_COLOR = (0x14, 0x00, 0x23)
COLORLESS_TILE_COLOR = (0x80, 0x80, 0x80)
SAFE_TILE_COLOR = (0x33, 0x33, 0x33)
FLAGGED_TILE_COLOR = (0xE6, 0xE6, 0xE6)


def color_is_close(color1, color2):
    return sum(abs(c2 - c1) for c1, c2 in zip(color1, color2)) < 30


class TileColorSet:
    def __init__(self):
        self.colors = [COLORLESS_TILE_COLOR, SAFE_TILE_COLOR, FLAGGED_TILE_COLOR]

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
    polygon: [(int, int)]
    color: (int, int, int)
    number: int
    tile_state: TileState

    @property
    def click_point(self):
        """
        The point to click on the figure.

        Equivalent to the center of the tile. This is only valid if the
        tile shapes are all convex.
        """
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


@attr.s(auto_attribs=True)
class Constraint:
    min_mines: int
    max_mines: int
    tiles: {Tile}


@attr.s(auto_attribs=True)
class Board:
    tiles: {Tile}
    _adjacencies: {Tile: {Tile}}
    constraints: [Constraint]

    @classmethod
    def new(cls, tiles):
        tiles = list(tiles)
        return Board(
            tiles=tiles,
            adjacencies=collections.defaultdict(set),
            constraints=[],
        )

    def add_adjacency(self, tile1, tile2):
        self._adjacencies[tile1].add(tile2)
        self._adjacencies[tile2].add(tile1)

    def generate_constraints(self, color_count):
        constraints = []
        for tile, adjacencies in self._adjacencies.items():
            if tile.number is None:
                continue
            unsolved_neighbors = {adj_tile for adj_tile in adjacencies if adj_tile.is_unsolved}
            constraints.append(
                Constraint(min_mines=tile.number, max_mines=tile.number, tiles=unsolved_neighbors)
            )


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


def parse_tile(image, labeled, object_area, index, color_set):
    # limit both large arrays to just the area we care about,
    # then fetch the array of only the pixel values that are labeled with our index.
    # tile_pixels[i] = [r, g, b]
    figure_filter = labeled[object_area] == index
    tile_pixels = image[object_area][figure_filter]
    color = numpy.median(tile_pixels, axis=0)
    color = color_set.get_color(color)

    shape = parse_polygon(object_area, figure_filter)

    return Tile(
        polygon=shape,
        color=color,
        number=None,
        tile_state=TileState.unsolved,
    )


def parse_polygon(object_area, figure_filter):
    """
    Fetch the points of the polygon from the image data.
    """
    # Generate two grids the size of the object area, `range` style, sorta.
    xs, ys = numpy.mgrid[object_area]
    # Convert em to a grid of points.
    # grid[x, y] = [x, y], except offset by the start of object_area
    grid = numpy.concatenate((xs[..., numpy.newaxis], ys[..., numpy.newaxis]), axis=-1)
    figure = grid[figure_filter]
    hull = scipy.spatial.ConvexHull(figure)

    # This is a good start, but it's still too rough.
    # Might need to do a manual second pass to flatten nearly-colinear segments.
    return [
        (figure[vertex, 0], figure[vertex, 1])
        for vertex in hull.vertices
    ]


def draw_board(board):
    image = PIL.Image.new('RGBA', (2400, 2160), (0,0,0,255))
    pdraw = PIL.ImageDraw.Draw(image)
    for tile in board.tiles:
        pdraw.polygon(tile.polygon, fill=tile.color)
    return image


def parse_board():
    import PIL.ImageDraw
    image = PIL.Image.open('Capture.png')
    image = image.convert('RGB')
    array = numpy.array(image)
    board_area = array[:, 720:3120].swapaxes(0, 1)

    mask = (
        (board_area[:, :, 0] != BACKGROUND_COLOR[0])
        | (board_area[:, :, 1] != BACKGROUND_COLOR[1])
        | (board_area[:, :, 2] != BACKGROUND_COLOR[2])
    )
    # `labeled` creates an array[x, y] = idx
    labeled, num_objs = scipy.ndimage.label(mask)
    # visualize_labels(labeled)
    object_areas = scipy.ndimage.find_objects(labeled)

    color_set = TileColorSet()
    tiles = [
        parse_tile(board_area, labeled, object_area, index, color_set)
        for index, object_area in enumerate(object_areas, 1)
    ]
    return Board.new(tiles)


def main():
    board = parse_board()
    draw_board(board).show()


if __name__ == '__main__':
    main()
