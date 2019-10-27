import collections
import enum
import math

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
    mask_area: []

    @classmethod
    def new(self, polygon, color, mask_area, number=None):
        state = TileState.unsolved
        if color == SAFE_TILE_COLOR:
            state = TileState.safe
            color = None
        elif color == FLAGGED_TILE_COLOR:
            state = TileState.flagged
            color = None
        return Tile(
            polygon=polygon,
            color=color,
            number=number,
            tile_state=state,
            mask_area=mask_area,
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
    def new(cls, tiles, adjacencies):
        tiles = list(tiles)
        board = Board(
            tiles=tiles,
            adjacencies=adjacencies,
            constraints=[],
        )
        board.generate_constraints()
        return board

    def generate_constraints(self, color_count=None):
        constraints = []
        for tile, adjacencies in self._adjacencies.items():
            if tile.number is None:
                continue
            unsolved_neighbors = {adj_tile for adj_tile in adjacencies if adj_tile.is_unsolved}
            constraints.append(
                Constraint(min_mines=tile.number, max_mines=tile.number, tiles=unsolved_neighbors)
            )
        self.constraints = constraints


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


def parse_tile(image, labeled, object_area, index, color_set):
    # limit both large arrays to just the area we care about,
    # then fetch the array of only the pixel values that are labeled with our index.
    # tile_pixels[i] = [r, g, b]
    figure_filter = labeled[object_area] == index
    tile_pixels = image[object_area][figure_filter]
    color = numpy.median(tile_pixels, axis=0)
    color = color_set.get_color(color)

    shape = parse_polygon(object_area, figure_filter)
    adjacencies = fetch_adjacencies(labeled, object_area, index)

    return Tile.new(
        polygon=shape,
        color=color,
        number=None,
        mask_area=figure_filter,
    ), adjacencies


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

    vertices = [
        (figure[vertex, 0], figure[vertex, 1])
        for vertex in hull.vertices
    ]
    return simplify_polyon(vertices)


def get_distance_from_line(p, p0, p1):
    base_distance = dist(p0, p1)
    area = get_triangle_area([p, p0, p1])
    return area * 2 / base_distance


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


def simplify_polyon(vertices):
    return vertices


# def simplify_polyon(vertices, ε=5):
#     """
#     Shave unnecessary vertices from polygon.

#     https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
#     """
#     if len(vertices) <= 2:
#         return list(vertices)

#     # Find the point with the maximum distance
#     dmax, index = max(
#         (get_distance_from_line(vert, vertices[0], vertices[-1]), i)
#         for i, vert in enumerate(vertices[1:-1], 1)
#     )

#     # If max distance is greater than epsilon, recursively simplify
#     # Otherwise, flatten the range.
#     if dmax <= ε:
#         return [vertices[0], vertices[-1]]

#     results1 = simplify_polyon(vertices[:index + 1], ε)
#     results2 = simplify_polyon(vertices[index:], ε)
#     return results1[:-1] + results2


# def simplify_polyon(vertices):
#     """
#     Combine line segments that are mostly colinear.
#     """
#     # # XXX this is an absurd amount of tolerance. We might want a better solution for this.
#     # ε = math.pi / 5
#     # # print("original", len(vertices))
#     new_vertices = []
#     triangle_iter = zip(vertices, rotate(vertices, 1), rotate(vertices, 2))
#     for vertices in triangle_iter:
#         if not is_short_triangle(vertices):
#             (ax, ay), (bx, by), (cx, cy) = vertices
#             new_vertices.append((bx, by))
#     return new_vertices


# def simplify_polyon(vertices):
#     """
#     Combine line segments that are mostly colinear.
#     """
#     # XXX this is an absurd amount of tolerance. We might want a better solution for this.
#     ε = math.pi / 5
#     # print("original", len(vertices))
#     new_vertices = []
#     line_segment_iter = zip(vertices + [vertices[0]], vertices[1:] + vertices[:2])
#     # line_segment_iter = list(line_segment_iter)
#     # print(line_segment_iter)
#     (x0, y0), (x1, y1) = next(line_segment_iter)
#     # last_angle = first_angle = math.atan2(y1 - y0, x1 - x0)

#     # We skip setting the first vertex. We special case it at the end,
#     # since we might want to omit it.
#     for (x1, y1), (x2, y2) in line_segment_iter:
#         # Compare
#         current_angle = math.atan2(y2 - y1, x2 - x1)
#         potential_angle = math.atan2(y2 - y0, x2 - x0)
#         # print(angle)
#         if not math.isclose(current_angle, potential_angle, abs_tol=ε):
#             new_vertices.append((x1, y1))
#             x0, y0 = x1, y1

#     # current_angle = math.atan2(y2 - y1, x2 - x1)
#     # potential_angle = math.atan2(y2 - y0, x2 - x0)
#     # if not math.isclose(last_angle, first_angle, abs_tol=ε):
#     #     new_vertices.append(vertices[0])

#     # print("final", len(new_vertices), new_vertices)
#     # assert len(new_vertices) <= 6, new_vertices
#     return new_vertices


def get_tile_draw_color(tile):
    if tile.is_flagged:
        return FLAGGED_TILE_COLOR
    if tile.is_safe:
        return SAFE_TILE_COLOR
    return tile.color


def draw_board(board, i=None):
    image = PIL.Image.new('RGBA', (2400, 2160), (0,0,0,255))
    pdraw = PIL.ImageDraw.Draw(image)
    for tile in board.tiles:
        color = get_tile_draw_color(tile)
        if i is not None:
            focus, adj = list(board._adjacencies.items())[i]
            if tile == focus:
                color = (0, 0, 0xFF)
            if tile in adj:
                color = (0xFF, 0, 0)
        pdraw.polygon(tile.polygon, fill=color)

    # # highlight vertices, for debug
    # for tile in board.tiles:
    #     for vertex in tile.polygon:
    #         pdraw.ellipse([vertex[0] - 3, vertex[1] - 3, vertex[0] + 3, vertex[1] + 3], fill=(0xFF, 0, 0))
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
    adjacency_pointers = dict(
        parse_tile(board_area, labeled, object_area, index, color_set)
        for index, object_area in enumerate(object_areas, 1)
    )
    tiles = list(adjacency_pointers)

    adjacencies = {
        tile: {tiles[idx - 1] for idx in adjacencies}
        for tile, adjacencies in adjacency_pointers.items()
    }
    return Board.new(
        tiles=tiles,
        adjacencies=adjacencies,
    )


def main():
    board = parse_board()
    draw_board(board).show()
    # for i in range(100):
    #     draw_board(board, i).show()
    #     input()


if __name__ == '__main__':
    main()
