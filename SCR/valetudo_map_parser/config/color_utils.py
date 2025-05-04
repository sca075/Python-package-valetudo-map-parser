"""Utility functions for color operations in the map parser."""

from typing import Tuple, Optional

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.types import NumpyArray, Color


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
    1. Gets the current element at position (x,y) from the element map
    2. Gets the color for that element from the colors manager
    3. Blends the new color with the existing color based on alpha values

    Returns:
        Blended RGBA color to use for drawing
    """
    # Sample background color at the endpoints and blend with foreground color
    # This is more efficient than sampling at every pixel
    if 0 <= y0 < arr.shape[0] and 0 <= x0 < arr.shape[1]:
        start_blended_color = ColorsManagement.sample_and_blend_color(
            arr, x0, y0, color
        )
    else:
        start_blended_color = color

    if 0 <= y1 < arr.shape[0] and 0 <= x1 < arr.shape[1]:
        end_blended_color = ColorsManagement.sample_and_blend_color(
            arr, x1, y1, color
        )
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
