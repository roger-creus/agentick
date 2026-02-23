"""Isometric coordinate math utilities for 2:1 projection."""

from __future__ import annotations


def grid_to_screen(
    row: int, col: int, tile_w: int, tile_h: int
) -> tuple[int, int]:
    """Convert grid (row, col) to screen pixel position.

    Standard 2:1 isometric projection where:
    - tile_w is the full width of the tile image
    - tile_h is the diamond-base height (typically tile_w // 2)

    Returns:
        (screen_x, screen_y) pixel coordinates
    """
    sx = (col - row) * (tile_w // 2)
    sy = (col + row) * (tile_h // 2)
    return sx, sy


def screen_to_grid(
    sx: int, sy: int, tile_w: int, tile_h: int
) -> tuple[int, int]:
    """Inverse: screen pixel to grid cell (for mouse interaction)."""
    half_w = tile_w // 2
    half_h = tile_h // 2
    if half_w == 0 or half_h == 0:
        return 0, 0
    col = (sx / half_w + sy / half_h) / 2
    row = (sy / half_h - sx / half_w) / 2
    return int(round(row)), int(round(col))


def calculate_canvas_size(
    rows: int,
    cols: int,
    tile_w: int,
    tile_h: int,
    tile_depth: int,
    padding_top: int = 40,
) -> tuple[int, int]:
    """Calculate required canvas size for the isometric grid.

    Args:
        rows: Number of grid rows
        cols: Number of grid columns
        tile_w: Full tile image width
        tile_h: Diamond-base height
        tile_depth: Vertical extent of tile cube above the diamond base
        padding_top: Extra space at top for HUD

    Returns:
        (width, height) in pixels
    """
    width = (rows + cols) * (tile_w // 2)
    height = (rows + cols) * (tile_h // 2) + tile_depth + padding_top
    return width, height


def calculate_offset(rows: int, tile_w: int) -> int:
    """Calculate X offset to center the isometric diamond.

    The isometric grid forms a diamond shape. The top-left cell projects
    to negative X, so we shift right by this amount.
    """
    return (rows - 1) * (tile_w // 2)
