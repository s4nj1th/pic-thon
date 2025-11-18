import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def img_to_lightness_grid(path, size=256):
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def render_line_art(lightness_grid, out_path="output.png"):
    h, w = lightness_grid.shape
    y_positions = np.arange(h)

    plt.figure(figsize=(6, 6))
    for y in range(h):
        row = lightness_grid[y]
        x = np.arange(w)
        thickness = 0.5 + 3.0 * (1.0 - row)
        for i in range(w - 1):
            plt.plot(
                [x[i], x[i + 1]],
                [y, y],
                linewidth=min(thickness[i], 1),
                color="black",
            )

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


if __name__ == "__main__":
    grid = img_to_lightness_grid("input/01.jpg")
    render_line_art(grid, "output/01.jpg")
