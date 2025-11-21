#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def img_to_lightness_grid(path: str, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L")
    resample = Image.LANCZOS  # type: ignore[attr-defined]
    img = img.resize((size, size), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


ALLOWED_DIRECTIONS = [
    "horizontal",
    "vertical",
    "diagonal_right",
    "diagonal_left",
    "crosshatch",
    "radial",
    "circular",
]


def render_line_art(
    lightness_grid: np.ndarray,
    out_path: str = "output.png",
    num_strokes: int = 256,
    direction: str = "horizontal",
    stroke_thickness_range: Tuple[float, float] = (0.5, 3.0),
    spacing_factor: float = 1.0,
    curvature: float = 0.0,
    opacity_factor: float = 1.0,
    color: str = "black",
    background_color: str = "white",
    invert_lightness: bool = False,
    output_resolution: Optional[Tuple[int, int]] = None,
) -> None:
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(
            f"Invalid direction: {direction}. Must be one of {ALLOWED_DIRECTIONS}"
        )

    h, w = lightness_grid.shape
    min_thick, max_thick = stroke_thickness_range

    if output_resolution is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = output_resolution

    if invert_lightness:
        lightness_grid = 1.0 - lightness_grid

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, out_w)
    ax.set_ylim(0, out_h)
    ax.set_facecolor(background_color)

    if direction == "horizontal":
        _draw_horizontal_strokes(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            spacing_factor,
            curvature,
            opacity_factor,
            color,
            out_w,
            out_h,
        )

    elif direction == "vertical":
        _draw_vertical_strokes(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            spacing_factor,
            curvature,
            opacity_factor,
            color,
            out_w,
            out_h,
        )

    elif direction == "diagonal_right":
        _draw_diagonal_strokes(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            spacing_factor,
            curvature,
            opacity_factor,
            color,
            out_w,
            out_h,
            right=True,
        )

    elif direction == "diagonal_left":
        _draw_diagonal_strokes(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            spacing_factor,
            curvature,
            opacity_factor,
            color,
            out_w,
            out_h,
            right=False,
        )

    elif direction == "crosshatch":
        _draw_crosshatch(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            spacing_factor,
            opacity_factor,
            color,
            out_w,
            out_h,
        )

    elif direction == "radial":
        _draw_radial_strokes(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            opacity_factor,
            color,
            out_w,
            out_h,
        )

    elif direction == "circular":
        _draw_circular_strokes(
            ax,
            lightness_grid,
            num_strokes,
            min_thick,
            max_thick,
            opacity_factor,
            color,
            out_w,
            out_h,
        )

    ax.invert_yaxis()
    ax.axis("off")
    plt.savefig(
        out_path, bbox_inches="tight", pad_inches=0, dpi=300, facecolor=background_color
    )
    plt.close()


def _draw_horizontal_strokes(
    ax,
    grid,
    num_strokes,
    min_thick,
    max_thick,
    spacing,
    curv,
    alpha,
    color,
    out_w,
    out_h,
):
    h, w = grid.shape
    y_positions = np.linspace(0, out_h, max(1, num_strokes))

    for y_pos in y_positions:
        grid_y = int((y_pos / out_h) * (h - 1))
        grid_y = np.clip(grid_y, 0, h - 1)

        row = grid[grid_y]
        x = np.linspace(0, out_w, w)

        y_wave = y_pos + curv * np.sin(x / out_w * 4 * np.pi) * (out_h / 50)

        for i in range(len(x) - 1):
            x_idx = int((x[i] / out_w) * (w - 1))
            x_idx = np.clip(x_idx, 0, w - 1)
            lightness = row[x_idx]
            thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)

            ax.plot(
                [x[i], x[i + 1]],
                [y_wave[i], y_wave[i + 1]],
                linewidth=thickness,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
            )


def _draw_vertical_strokes(
    ax,
    grid,
    num_strokes,
    min_thick,
    max_thick,
    spacing,
    curv,
    alpha,
    color,
    out_w,
    out_h,
):
    h, w = grid.shape
    x_positions = np.linspace(0, out_w, max(1, num_strokes))

    for x_pos in x_positions:
        grid_x = int((x_pos / out_w) * (w - 1))
        grid_x = np.clip(grid_x, 0, w - 1)

        col = grid[:, grid_x]
        y = np.linspace(0, out_h, h)

        x_wave = x_pos + curv * np.sin(y / out_h * 4 * np.pi) * (out_w / 50)

        for i in range(len(y) - 1):
            y_idx = int((y[i] / out_h) * (h - 1))
            y_idx = np.clip(y_idx, 0, h - 1)
            lightness = col[y_idx]
            thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)

            ax.plot(
                [x_wave[i], x_wave[i + 1]],
                [y[i], y[i + 1]],
                linewidth=thickness,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
            )


def _draw_diagonal_strokes(
    ax,
    grid,
    num_strokes,
    min_thick,
    max_thick,
    spacing,
    curv,
    alpha,
    color,
    out_w,
    out_h,
    right=True,
):
    h, w = grid.shape
    max_dim = max(out_h, out_w)

    for i in range(max(1, num_strokes)):
        offset = (i / (num_strokes - 1) - 0.5) * max_dim * 2 if num_strokes > 1 else 0

        points = []
        for t in np.linspace(0, 1, 500):
            if right:
                x = t * out_w + offset
                y = t * out_h
            else:
                x = t * out_w - offset
                y = t * out_h

            if 0 <= x < out_w and 0 <= y < out_h:
                grid_x = int((x / out_w) * (w - 1))
                grid_y = int((y / out_h) * (h - 1))
                grid_x = np.clip(grid_x, 0, w - 1)
                grid_y = np.clip(grid_y, 0, h - 1)

                lightness = grid[grid_y, grid_x]
                thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)
                points.append((x, y, thickness))

        if len(points) > 1:
            for j in range(len(points) - 1):
                x1, y1, t1 = points[j]
                x2, y2, t2 = points[j + 1]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=(t1 + t2) / 2,
                    color=color,
                    alpha=alpha,
                    solid_capstyle="round",
                )


def _draw_crosshatch(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    _draw_diagonal_strokes(
        ax,
        grid,
        max(1, num_strokes // 2),
        min_thick,
        max_thick,
        spacing,
        0,
        alpha * 0.7,
        color,
        out_w,
        out_h,
        right=True,
    )
    _draw_diagonal_strokes(
        ax,
        grid,
        max(1, num_strokes // 2),
        min_thick,
        max_thick,
        spacing,
        0,
        alpha * 0.7,
        color,
        out_w,
        out_h,
        right=False,
    )


def _draw_radial_strokes(
    ax, grid, num_strokes, min_thick, max_thick, alpha, color, out_w, out_h
):
    h, w = grid.shape
    center_x, center_y = out_w / 2, out_h / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    for angle in np.linspace(0, 2 * np.pi, max(1, num_strokes), endpoint=False):
        radii = np.linspace(0, max_radius, 200)

        for i in range(len(radii) - 1):
            x1 = center_x + radii[i] * np.cos(angle)
            y1 = center_y + radii[i] * np.sin(angle)
            x2 = center_x + radii[i + 1] * np.cos(angle)
            y2 = center_y + radii[i + 1] * np.sin(angle)

            if 0 <= x1 < out_w and 0 <= y1 < out_h:
                grid_x = int((x1 / out_w) * (w - 1))
                grid_y = int((y1 / out_h) * (h - 1))
                grid_x = np.clip(grid_x, 0, w - 1)
                grid_y = np.clip(grid_y, 0, h - 1)

                lightness = grid[grid_y, grid_x]
                thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=thickness,
                    color=color,
                    alpha=alpha,
                    solid_capstyle="round",
                )


def _draw_circular_strokes(
    ax, grid, num_strokes, min_thick, max_thick, alpha, color, out_w, out_h
):
    h, w = grid.shape
    center_x, center_y = out_w / 2, out_h / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    for radius in np.linspace(5, max_radius, max(1, num_strokes)):
        angles = np.linspace(0, 2 * np.pi, 360)

        for i in range(len(angles) - 1):
            x1 = center_x + radius * np.cos(angles[i])
            y1 = center_y + radius * np.sin(angles[i])
            x2 = center_x + radius * np.cos(angles[i + 1])
            y2 = center_y + radius * np.sin(angles[i + 1])

            if 0 <= x1 < out_w and 0 <= y1 < out_h:
                grid_x = int((x1 / out_w) * (w - 1))
                grid_y = int((y1 / out_h) * (h - 1))
                grid_x = np.clip(grid_x, 0, w - 1)
                grid_y = np.clip(grid_y, 0, h - 1)

                lightness = grid[grid_y, grid_x]
                thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=thickness,
                    color=color,
                    alpha=alpha,
                    solid_capstyle="round",
                )


def _safe_stem(path: Path) -> str:
    return path.stem


def _ensure_output_dir(dirpath: Path) -> None:
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render stylized line-art strokes from an input image. "
        "Requires input file, direction, and stroke count to be specified."
    )
    parser.add_argument("-i", "--input", help="Input image path", required=True)
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (e.g. 'output.png' or 'results/myart.jpg')",
        required=True,
    )
    parser.add_argument(
        "--size",
        help="Size to resize image to for the lightness grid (square)",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-s",
        "--strokes",
        type=int,
        help="Number of strokes to render",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--direction",
        choices=ALLOWED_DIRECTIONS,
        help="Direction of strokes",
        required=True,
    )
    parser.add_argument(
        "--curvature",
        type=float,
        help="Curvature amount for wavy effects (default: 0.0)",
        default=0.0,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    out_path = Path(args.output)
    out_dir = out_path.parent
    if out_dir and str(out_dir) != ".":
        _ensure_output_dir(out_dir)

    original_img = Image.open(input_path)
    orig_width, orig_height = original_img.size

    grid = img_to_lightness_grid(str(input_path), size=args.size)

    render_line_art(
        grid,
        str(out_path),
        num_strokes=args.strokes,
        direction=args.direction,
        output_resolution=(orig_width, orig_height),
        curvature=args.curvature,
    )
    print(f"Wrote {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
