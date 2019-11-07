import pathlib
import time
import webbrowser


def render_tile(board, tile, index, colors):
    points = ' '.join(f"{x},{y}" for x, y in tile.polygon)
    color = f"color{colors.index(tile.draw_color())}"
    adj_indexes = [board.tiles.index(adj_tile) for adj_tile in board._adjacencies[tile]]
    adj_classes = " ".join(f"tile{i}adj" for i in adj_indexes)
    x, y = tile.click_point
    return f"""<polygon points="{points}" id="tile{index}" class="tile {color} {adj_classes}" />
  <text x="{x}" y="{y}", text-anchor="middle" fill="white" font-size="40">{tile.text or ""}</text>
    """


def render_color_counts(board):
  scale = 120
  return '\n'.join(
      f'  <text x="3820" y="{i * scale}" text-anchor="end" fill="{color_to_hex(color)}" '
      f'font-size="{scale}">{count}</text>'
      for i, (color, count) in enumerate(board.get_remaining_color_count().items(), 1)
  )


def get_colors(board):
    return list({tile.draw_color() for tile in board.tiles})


def color_to_hex(color):
    return "#" + ''.join(map("{:02x}".format, color))


def tile_adjacency_styles(i):
    return f"  .tile{i}:hover ~ .tile{i}adj {{ fill: red; }}"


def render_board(board):
    colors = get_colors(board)
    color_classes = "\n  ".join(
        f".color{i} {{ fill: {color_to_hex(color)} }}" for i, color in enumerate(colors)
    )
    tiles = "\n  ".join(render_tile(board, tile, i, colors) for i, tile in enumerate(board.tiles))
    adjacency_styles = ""
    color_counts = render_color_counts(board)
    return f"""<!DOCTYPE html>
<html>
<head>
  <script>
  document.addEventListener("mouseover", function (event) {{
    if (!event.target.matches || !event.target.matches('.tile')) return;
    let id = event.target.id.substring(4);
    let adjs = document.getElementsByClassName(`tile${{id}}adj`);
    for (var adj of adjs) {{
      adj.classList.add("hover");
    }}
  }});
  document.addEventListener("mouseout", function (event) {{
    if (!event.target.matches || !event.target.matches('.tile')) return;
    let id = event.target.id.substring(4);
    let adjs = document.getElementsByClassName(`tile${{id}}adj`);
    for (var adj of adjs) {{
      adj.classList.remove("hover");
    }}
  }});
  </script>
  <style>
  {color_classes}
  .tile:hover {{
    stroke: yellow;
    stroke-width: 10px;
  }}
  .tile.hover {{
    stroke: cyan;
    stroke-width: 10px;
  }}
  text {{
    cursor: default;
    user-select: none;
    pointer-events: none;
  }}
{adjacency_styles}
  </style>
</head>
<body>

<svg width="1536" height="864" viewBox="0 0 3840 2160" style="background-color: rgb(20, 0, 35)">
  {tiles}
  {color_counts}
  Sorry, your browser does not support inline SVG.
</svg>

</body>
</html>
"""


def visualize(board):
    html = render_board(board)
    filename = pathlib.Path(f"visualizations/board_{time.time()}.html").resolve()
    with open(filename, "w") as file:
        file.write(html)

    webbrowser.get(using="chrome").open(f"file://{filename}")
