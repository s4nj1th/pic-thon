#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def img_to_lightness_grid(path: str, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L")
    resample = Image.LANCZOS  # type: ignore
    img = img.resize((size, size), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


ALLOWED_DIRECTIONS = [
    "horizontal",
    "vertical",
    "diagonal_right",
    "diagonal_left",
    "reverse_diagonal_right",
    "reverse_diagonal_left",
    "crosshatch",
    "radial",
    "circular",
    "spiral",
    "halftone",
]


def render_line_art(
    lightness_grid: np.ndarray,
    out_path: str = "output.png",
    num_strokes: int = 256,
    direction: str = "horizontal",
    stroke_thickness_range: Optional[Tuple[float, float]] = None,
    spacing_factor: float = 1.0,
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

    if stroke_thickness_range is None:
        max_thick = max(0.5, 3.0 - (num_strokes / 100))
        min_thick = max(0.2, max_thick / 6)
        stroke_thickness_range = (min_thick, max_thick)

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
    ax.invert_yaxis()
    ax.axis("off")

    renderer_params = {
        "ax": ax,
        "grid": lightness_grid,
        "num_strokes": num_strokes,
        "min_thick": min_thick,
        "max_thick": max_thick,
        "spacing": spacing_factor,
        "alpha": opacity_factor,
        "color": color,
        "out_w": out_w,
        "out_h": out_h,
    }

    if direction == "horizontal":
        draw_horizontal_strokes(**renderer_params)
    elif direction == "vertical":
        draw_vertical_strokes(**renderer_params)
    elif direction == "diagonal_right":
        draw_diagonal_strokes(**renderer_params, right=True, reverse=False)
    elif direction == "diagonal_left":
        draw_diagonal_strokes(**renderer_params, right=False, reverse=False)
    elif direction == "reverse_diagonal_right":
        draw_diagonal_strokes(**renderer_params, right=True, reverse=True)
    elif direction == "reverse_diagonal_left":
        draw_diagonal_strokes(**renderer_params, right=False, reverse=True)
    elif direction == "crosshatch":
        draw_crosshatch(**renderer_params)
    elif direction == "radial":
        draw_radial_strokes(**renderer_params)
    elif direction == "circular":
        draw_circular_strokes(**renderer_params)
    elif direction == "spiral":
        draw_spiral_strokes(**renderer_params)
    elif direction == "halftone":
        draw_halftone(**renderer_params)

    plt.savefig(
        out_path, bbox_inches="tight", pad_inches=0, dpi=300, facecolor=background_color
    )
    plt.close()


def draw_horizontal_strokes(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    h, w = grid.shape
    space_per_stroke = out_h / max(1, num_strokes)

    for idx in range(num_strokes):
        y_pos = idx * space_per_stroke + space_per_stroke / 2
        grid_y = int((y_pos / out_h) * (h - 1))
        grid_y = np.clip(grid_y, 0, h - 1)

        row = grid[grid_y]
        x = np.linspace(0, out_w, w)

        for i in range(len(x) - 1):
            x_idx = int((x[i] / out_w) * (w - 1))
            x_idx = np.clip(x_idx, 0, w - 1)
            lightness = row[x_idx]
            thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)

            ax.plot(
                [x[i], x[i + 1]],
                [y_pos, y_pos],
                linewidth=thickness,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
            )


def draw_vertical_strokes(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    h, w = grid.shape
    space_per_stroke = out_w / max(1, num_strokes)

    for idx in range(num_strokes):
        x_pos = idx * space_per_stroke + space_per_stroke / 2
        grid_x = int((x_pos / out_w) * (w - 1))
        grid_x = np.clip(grid_x, 0, w - 1)

        col = grid[:, grid_x]
        y = np.linspace(0, out_h, h)

        for i in range(len(y) - 1):
            y_idx = int((y[i] / out_h) * (h - 1))
            y_idx = np.clip(y_idx, 0, h - 1)
            lightness = col[y_idx]
            thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)

            ax.plot(
                [x_pos, x_pos],
                [y[i], y[i + 1]],
                linewidth=thickness,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
            )


def draw_diagonal_strokes(
    ax,
    grid,
    num_strokes,
    min_thick,
    max_thick,
    spacing,
    alpha,
    color,
    out_w,
    out_h,
    right=True,
    reverse=False,
):
    h, w = grid.shape
    max_dim = max(out_h, out_w)
    space_per_stroke = (max_dim * 2) / max(1, num_strokes)

    for i in range(max(1, num_strokes)):
        offset = (i - num_strokes / 2 + 0.5) * space_per_stroke
        points = []

        for t in np.linspace(0, 1, 500):
            if reverse:
                x = t * out_w + (offset if right else -offset)
                y = out_h - t * out_h
            else:
                x = t * out_w + (offset if right else -offset)
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


def draw_crosshatch(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    half_strokes = max(1, num_strokes // 2)
    draw_diagonal_strokes(
        ax,
        grid,
        half_strokes,
        min_thick,
        max_thick,
        spacing,
        alpha * 0.7,
        color,
        out_w,
        out_h,
        right=True,
        reverse=False,
    )
    draw_diagonal_strokes(
        ax,
        grid,
        half_strokes,
        min_thick,
        max_thick,
        spacing,
        alpha * 0.7,
        color,
        out_w,
        out_h,
        right=False,
        reverse=False,
    )


def draw_radial_strokes(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    h, w = grid.shape
    center_x, center_y = out_w / 2, out_h / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    angle_spacing = (2 * np.pi) / max(1, num_strokes)

    for idx in range(max(1, num_strokes)):
        angle = idx * angle_spacing
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


def draw_circular_strokes(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    h, w = grid.shape
    center_x, center_y = out_w / 2, out_h / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    radius_step = max_radius / max(1, num_strokes + 1)

    for idx in range(1, max(1, num_strokes) + 1):
        radius = idx * radius_step
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


def draw_spiral_strokes(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    h, w = grid.shape
    center_x, center_y = out_w / 2, out_h / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    total_points = num_strokes * 360

    points = []
    for i in range(total_points):
        angle = (i / total_points) * num_strokes * 2 * np.pi
        radius = (i / total_points) * max_radius

        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)

        if 0 <= x < out_w and 0 <= y < out_h:
            grid_x = int((x / out_w) * (w - 1))
            grid_y = int((y / out_h) * (h - 1))
            grid_x = np.clip(grid_x, 0, w - 1)
            grid_y = np.clip(grid_y, 0, h - 1)

            lightness = grid[grid_y, grid_x]
            thickness = min_thick + (max_thick - min_thick) * (1.0 - lightness)
            points.append((x, y, thickness))

    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1, t1 = points[i]
            x2, y2, t2 = points[i + 1]
            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=(t1 + t2) / 2,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
            )


def draw_halftone(
    ax, grid, num_strokes, min_thick, max_thick, spacing, alpha, color, out_w, out_h
):
    h, w = grid.shape

    dots_per_row = max(10, num_strokes)
    dots_per_col = int(dots_per_row * (out_h / out_w))

    x_spacing = out_w / dots_per_row
    y_spacing = out_h / dots_per_col

    max_radius = min(x_spacing, y_spacing) * 0.4
    min_radius = max_radius * 0.1

    for row_idx in range(dots_per_col):
        for col_idx in range(dots_per_row):
            x = (col_idx + 0.5) * x_spacing
            y = (row_idx + 0.5) * y_spacing

            grid_x = int((x / out_w) * (w - 1))
            grid_y = int((y / out_h) * (h - 1))
            grid_x = np.clip(grid_x, 0, w - 1)
            grid_y = np.clip(grid_y, 0, h - 1)

            lightness = grid[grid_y, grid_x]

            dot_radius = min_radius + (max_radius - min_radius) * (1.0 - lightness)

            if dot_radius > min_radius * 1.5:
                circle = mpatches.Circle((x, y), dot_radius, color=color, alpha=alpha)
                ax.add_patch(circle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render stylized line-art strokes from an input image."
    )
    parser.add_argument("-i", "--input", help="Input image path", required=True)
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (optional, auto-generated if not specified)",
        required=False,
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
        help="Number of strokes to render (default: 100)",
        default=100,
    )
    parser.add_argument(
        "-d",
        "--direction",
        choices=ALLOWED_DIRECTIONS,
        help="Direction of strokes (default: circular)",
        default="circular",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        ext = input_path.suffix
        out_path = (
            input_path.parent / f"{stem}_output_{args.direction}_{args.strokes}{ext}"
        )

    out_dir = out_path.parent
    if out_dir and str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    original_img = Image.open(input_path)
    orig_width, orig_height = original_img.size

    grid = img_to_lightness_grid(str(input_path), size=args.size)

    render_line_art(
        grid,
        str(out_path),
        num_strokes=args.strokes,
        direction=args.direction,
        output_resolution=(orig_width, orig_height),
    )
    print(f"Wrote {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
