"""Utility functions for color operations in the map parser."""

from typing import Tuple, Optional

from ..config.colors import ColorsManagment
from ..config.drawable_elements import ElementMapGenerator, DrawableElement


def get_blended_color(
    element_map_generator: ElementMapGenerator,
    colors_manager: ColorsManagment,
    x: int,
    y: int,
    new_element: DrawableElement,
    new_color: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Get a blended color for a pixel based on the current element map and the new element to draw.
    
    This function:
    1. Gets the current element at position (x,y) from the element map
    2. Gets the color for that element from the colors manager
    3. Blends the new color with the existing color based on alpha values
    
    Args:
        element_map_generator: The element map generator containing the current element map
        colors_manager: The colors manager to get colors for elements
        x: X coordinate in the element map
        y: Y coordinate in the element map
        new_element: The new element to draw at this position
        new_color: The RGBA color of the new element
        
    Returns:
        Blended RGBA color to use for drawing
    """
    # Get current element at this position
    current_element = element_map_generator.get_element_at_position(x, y)
    
    # If no current element or it's the same as the new element, just return the new color
    if current_element is None or current_element == new_element:
        return new_color
    
    # Get color for the current element
    current_color = None
    
    # Handle different element types
    if current_element == DrawableElement.FLOOR:
        # Floor is the background color
        current_color = colors_manager.get_colour("color_background")
    elif current_element == DrawableElement.WALL:
        # Wall color
        current_color = colors_manager.get_colour("color_wall")
    elif DrawableElement.ROOM_1 <= current_element <= DrawableElement.ROOM_15:
        # Room colors (ROOM_1 = 16, ROOM_2 = 17, etc.)
        room_index = current_element - DrawableElement.ROOM_1
        current_color = colors_manager.get_colour(f"color_room_{room_index}")
    else:
        # Default for unknown elements
        current_color = (100, 100, 100, 255)
    
    # Blend the colors
    return colors_manager.blend_colors(current_color, new_color)
