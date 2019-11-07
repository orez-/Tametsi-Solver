# Tametsi-Solver
:camera: Solver for the game [Tametsi](https://store.steampowered.com/app/709920/Tametsi/) - parses the screen for information and solves.

[See a sample demo](https://i.imgur.com/mgS8rj5.mp4)

Tametsi is a Windows-only game, so this project is developed for a Windows environment.

---

The problem statement being solved by this project is very similar to that of my [HexCells solver](https://github.com/orez-/Hexcells-Solver), with a few interesting differences:
- Tametsi is presented much more straightforward than HexCells.
Less shading on the tiles, no confetti when clicking a tile, no shifting background.
In general this means screenshots of Tametsi boards tend to be much less hassle to process.
- While HexCells's tiles are all hexagons, Tametsi includes all sorts of differently shaped tiles and two adjacency patterns.
This makes identifying tiles and their adjacencies more challenging.
- Tametsi only has one type of constraint, the region count constraint, where HexCells also includes constraints on contiguity of flags.
This doesn't mean Tametsi is an _easier_ game, but it does mean it is less complex to solve.
