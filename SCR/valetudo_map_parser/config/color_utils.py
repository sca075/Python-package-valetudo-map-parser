"""Utility functions for color operations in the map parser."""

from typing import Optional, Tuple

from .colors import ColorsManagement
from .types import Color, NumpyArray


def get_blended_color(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    arr: Optional[NumpyArray],
    color: Color,
) -> Color:
    """
    Get a blended color for a pixel based on the current element map and the new element to draw.

    This function:
    1. Gets the background colors at the start and end points (with offset to avoid sampling already drawn pixels)
    2. Directly blends the foreground color with the background using straight alpha
    3. Returns the average of the two blended colors

    Returns:
        Blended RGBA color to use for drawing
    """
    # Extract foreground color components
    fg_r, fg_g, fg_b, fg_a = color
    fg_alpha = fg_a / 255.0  # Convert to 0-1 range

    # Fast path for fully opaque or transparent foreground
    if fg_a == 255:
        return color
    if fg_a == 0:
        # Sample background at midpoint
        mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
        if 0 <= mid_y < arr.shape[0] and 0 <= mid_x < arr.shape[1]:
            return tuple(arr[mid_y, mid_x])
        return (0, 0, 0, 0)  # Default if out of bounds

    # Calculate direction vector for offset sampling
    dx = x1 - x0
    dy = y1 - y0
    length = max(1, (dx**2 + dy**2) ** 0.5)  # Avoid division by zero
    offset = 5  # 5-pixel offset to avoid sampling already drawn pixels

    # Calculate offset coordinates for start point (move away from the line)
    offset_x0 = int(x0 - (offset * dx / length))
    offset_y0 = int(y0 - (offset * dy / length))

    # Calculate offset coordinates for end point (move away from the line)
    offset_x1 = int(x1 + (offset * dx / length))
    offset_y1 = int(y1 + (offset * dy / length))

    # Sample background at offset start point
    if 0 <= offset_y0 < arr.shape[0] and 0 <= offset_x0 < arr.shape[1]:
        bg_color_start = arr[offset_y0, offset_x0]
        # Direct straight alpha blending
        start_r = int(fg_r * fg_alpha + bg_color_start[0] * (1 - fg_alpha))
        start_g = int(fg_g * fg_alpha + bg_color_start[1] * (1 - fg_alpha))
        start_b = int(fg_b * fg_alpha + bg_color_start[2] * (1 - fg_alpha))
        start_a = int(fg_a + bg_color_start[3] * (1 - fg_alpha))
        start_blended_color = (start_r, start_g, start_b, start_a)
    else:
        # If offset point is out of bounds, try original point
        if 0 <= y0 < arr.shape[0] and 0 <= x0 < arr.shape[1]:
            bg_color_start = arr[y0, x0]
            start_r = int(fg_r * fg_alpha + bg_color_start[0] * (1 - fg_alpha))
            start_g = int(fg_g * fg_alpha + bg_color_start[1] * (1 - fg_alpha))
            start_b = int(fg_b * fg_alpha + bg_color_start[2] * (1 - fg_alpha))
            start_a = int(fg_a + bg_color_start[3] * (1 - fg_alpha))
            start_blended_color = (start_r, start_g, start_b, start_a)
        else:
            start_blended_color = color

    # Sample background at offset end point
    if 0 <= offset_y1 < arr.shape[0] and 0 <= offset_x1 < arr.shape[1]:
        bg_color_end = arr[offset_y1, offset_x1]
        # Direct straight alpha blending
        end_r = int(fg_r * fg_alpha + bg_color_end[0] * (1 - fg_alpha))
        end_g = int(fg_g * fg_alpha + bg_color_end[1] * (1 - fg_alpha))
        end_b = int(fg_b * fg_alpha + bg_color_end[2] * (1 - fg_alpha))
        end_a = int(fg_a + bg_color_end[3] * (1 - fg_alpha))
        end_blended_color = (end_r, end_g, end_b, end_a)
    else:
        # If offset point is out of bounds, try original point
        if 0 <= y1 < arr.shape[0] and 0 <= x1 < arr.shape[1]:
            bg_color_end = arr[y1, x1]
            end_r = int(fg_r * fg_alpha + bg_color_end[0] * (1 - fg_alpha))
            end_g = int(fg_g * fg_alpha + bg_color_end[1] * (1 - fg_alpha))
            end_b = int(fg_b * fg_alpha + bg_color_end[2] * (1 - fg_alpha))
            end_a = int(fg_a + bg_color_end[3] * (1 - fg_alpha))
            end_blended_color = (end_r, end_g, end_b, end_a)
        else:
            end_blended_color = color

    # Use the average of the two blended colors
    blended_color = (
        (start_blended_color[0] + end_blended_color[0]) // 2,
        (start_blended_color[1] + end_blended_color[1]) // 2,
        (start_blended_color[2] + end_blended_color[2]) // 2,
        (start_blended_color[3] + end_blended_color[3]) // 2,
    )
    return blended_color
