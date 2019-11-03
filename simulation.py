import argparse
import collections
import json
import pathlib
import sys
import unittest.mock

import attr
import numpy
import PIL.Image
import PIL.ImageDraw


def rotate(list_, rotate_amt):
    return list_[rotate_amt:] + list_[:rotate_amt]


def determinant(p, p0, p1):
    px, py = p
    p0x, p0y = p0
    p1x, p1y = p1
    return (p1x - p0x) * (py - p0y) - (p1y - p0y) * (px - p0x)


def is_point_in_polygon(p, vertices):
    return len({determinant(p, p0, p1) > 0 for p0, p1 in zip(vertices, rotate(vertices, 1))}) == 1


class SimulationException(Exception):
    """Raised when the simulation believes the solver is misbehaving"""


@attr.s(auto_attribs=True, eq=False)
class Tile:
    is_mine: bool
    is_exposed: bool
    polygon: [(int, int)]
    auto_expand: [int]


def load_board_image(path):
    return numpy.array(PIL.Image.open(path).convert("RGB"), dtype='uint8')


@attr.s(auto_attribs=True)
class Simulation:
    current_image: numpy.array
    final_image: numpy.array
    tiles: [Tile]

    @classmethod
    def from_path(cls, path):
        test_path = pathlib.Path("tests") / path
        with open(test_path / 'cells.json', "r") as file:
            tile_data = json.load(file)
        tiles = [
            Tile(
                polygon=[tuple(coord) for coord in tile['polygon']],
                is_mine=tile['is_mine'],
                is_exposed=tile['is_exposed'],
                auto_expand=tile.get('auto_expand', []),
            )
            for tile in tile_data
        ]
        return cls(
            current_image=load_board_image(test_path / 'blank.png'),
            final_image=load_board_image(test_path / 'final.png'),
            tiles=tiles,
        )

    def _expose_tile(self, tile):
        # Paste in the new tile, and if it's `auto_expand` expand its neighbors, and so on.
        tiles = collections.deque([tile])
        while tiles:
            tile = tiles.popleft()
            if tile.is_exposed:  # :nothingtodohere:
                continue
            tile.is_exposed = True
            self.current_image = paste_tile(self.current_image, self.final_image, tile.polygon)
            tiles.extend(map(self.tiles.__getitem__, tile.auto_expand))

    def click(self, *, x, y, button):
        for tile in self.tiles:
            if is_point_in_polygon((x, y), tile.polygon):
                if tile.is_mine != (button == "right"):
                    raise SimulationException(f"mislabeled a tile! ({x}, {y}), clicked {button}")
                self._expose_tile(tile)
                return
        raise SimulationException(f"clicked on nothing ({x}, {y})")

    def screenshot(self):
        return PIL.Image.fromarray(self.current_image, 'RGB')


def paste_tile(current, final, polygon):
    mask_img = PIL.Image.new('1', (current.shape[1], current.shape[0]), 0)
    PIL.ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(mask_img)
    # This terrible incantation creates a view on our h,w mask to make it h,w,3.
    # Why numpy can't just treat `final` and `current` as h,w arrays (with elements of 3-arrays)
    # in `where` is a mystery to me ᖍ(•⟝•)ᖌ
    broadcast_mask = numpy.broadcast_to(mask[..., None], mask.shape + (3,))
    return numpy.where(broadcast_mask, final, current)


def assert_same_tile(tile1, tile2):
    # This is like bare-minimum reliability. It's very possible eg two
    # tiles as right triangles forming a square could have the same
    # bounding boxes. Let me know if you have a better idea x_x
    xs1, ys1 = tile1.box
    xs2, ys2 = tile2.box
    assert abs(xs1.start - xs2.start) <= 1, (tile1.box, tile2.box)
    assert abs(xs1.stop - xs2.stop) <= 1, (tile1.box, tile2.box)
    assert abs(ys1.start - ys2.start) <= 1, (tile1.box, tile2.box)
    assert abs(ys1.stop - ys2.stop) <= 1, (tile1.box, tile2.box)


def serialize_tile(initial_tile, final_tile, final_board):
    assert_same_tile(initial_tile, final_tile)

    ser_tile = {
        "is_mine": final_tile.is_flagged,
        "is_exposed": not initial_tile.is_unsolved,
        # TODO: god, stop.
        "polygon": [[int(x) + 720, int(y)] for x, y in final_tile.polygon],
    }
    # We only need to auto-expand tiles that are safe, don't start exposed,
    # and have no adjacent mines
    if not (ser_tile["is_mine"] or ser_tile["is_exposed"] or final_tile.text):
        ser_tile["auto_expand"] = [
            final_board.tiles.index(adj) for adj in final_board._adjacencies[final_tile]
        ]
    return ser_tile


def serialize_tiles(path):
    import main

    test_dir = pathlib.Path("tests") / path
    blank_board = main.parse_board(PIL.Image.open(test_dir / 'blank.png'))
    final_board = main.parse_board(PIL.Image.open(test_dir / 'final.png'))

    # This is an absurdly rough match up, but I'm not sure how to improve accuracy confidence :#
    tiles = [
        serialize_tile(itile, ftile, final_board)
        for itile, ftile in zip(blank_board.tiles, final_board.tiles)
    ]
    with open(test_dir / "cells.json", "w") as file:
        json.dump(tiles, file)


def run_simulation(path):
    import main

    sim = Simulation.from_path(path)
    with unittest.mock.patch("main.pyautogui", sim):
        main.solve_live_game()

    sim.screenshot().show()


def main():
    parser = argparse.ArgumentParser()
    groups = parser.add_subparsers(required=True)

    run_sim = groups.add_parser('run', help="run a simulated test case")
    run_sim.add_argument("path")
    run_sim.set_defaults(fn=run_simulation)

    ser_tiles = groups.add_parser(
        'create',
        help="create tile information for a test case. "
        "This command makes use of the main board parsing functionality that's being tested, "
        "so always doublecheck the results!",
    )
    ser_tiles.add_argument("path")
    ser_tiles.set_defaults(fn=serialize_tiles)

    if len(sys.argv) <= 1:
        parser.print_help()
        return

    args = parser.parse_args()
    args.fn(args.path)


if __name__ == '__main__':
    main()
