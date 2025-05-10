"""
Drawable Elements Configuration.
Defines the elements that can be drawn on the map and their properties.
Version: 0.1.9
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple, Union

import numpy as np

from .colors import DefaultColors, SupportedColor
from .types import LOGGER


# Type aliases
Color = Tuple[int, int, int, int]  # RGBA color
PropertyDict = Dict[str, Union[Color, float, int]]


class DrawableElement(IntEnum):
    """Enumeration of drawable map elements with unique integer codes."""

    # Base elements
    FLOOR = 1
    WALL = 2
    ROBOT = 3
    CHARGER = 4
    VIRTUAL_WALL = 5
    RESTRICTED_AREA = 6
    NO_MOP_AREA = 7
    OBSTACLE = 8
    PATH = 9
    PREDICTED_PATH = 10
    GO_TO_TARGET = 11

    # Rooms (101-115 for up to 15 rooms)
    ROOM_1 = 101
    ROOM_2 = 102
    ROOM_3 = 103
    ROOM_4 = 104
    ROOM_5 = 105
    ROOM_6 = 106
    ROOM_7 = 107
    ROOM_8 = 108
    ROOM_9 = 109
    ROOM_10 = 110
    ROOM_11 = 111
    ROOM_12 = 112
    ROOM_13 = 113
    ROOM_14 = 114
    ROOM_15 = 115


class DrawingConfig:
    """Configuration for which elements to draw and their properties."""

    def __init__(self):
        """Initialize with all elements enabled by default."""
        # Dictionary of element_code -> enabled status
        self._enabled_elements = {element: True for element in DrawableElement}

        # Dictionary of element_code -> drawing properties (color, opacity, etc.)
        self._element_properties: Dict[DrawableElement, PropertyDict] = {}

        # Initialize default properties
        self._set_default_properties()

    def _set_default_properties(self):
        """Set default drawing properties for each element."""
        # Set properties for rooms using DefaultColors
        for room_id in range(1, 16):
            room_element = getattr(DrawableElement, f"ROOM_{room_id}")
            room_key = SupportedColor.room_key(room_id - 1)
            rgb = DefaultColors.DEFAULT_ROOM_COLORS.get(room_key, (135, 206, 250))
            alpha = DefaultColors.DEFAULT_ALPHA.get(f"alpha_room_{room_id - 1}", 255.0)

            self._element_properties[room_element] = {
                "color": (*rgb, int(alpha)),
                "opacity": alpha / 255.0,
                "z_index": 10,  # Drawing order
            }

        # Map DrawableElement to SupportedColor
        element_color_mapping = {
            DrawableElement.FLOOR: SupportedColor.MAP_BACKGROUND,
            DrawableElement.WALL: SupportedColor.WALLS,
            DrawableElement.ROBOT: SupportedColor.ROBOT,
            DrawableElement.CHARGER: SupportedColor.CHARGER,
            DrawableElement.VIRTUAL_WALL: SupportedColor.NO_GO,
            DrawableElement.RESTRICTED_AREA: SupportedColor.NO_GO,
            DrawableElement.PATH: SupportedColor.PATH,
            DrawableElement.PREDICTED_PATH: SupportedColor.PREDICTED_PATH,
            DrawableElement.GO_TO_TARGET: SupportedColor.GO_TO,
            DrawableElement.NO_MOP_AREA: SupportedColor.NO_GO,  # Using NO_GO for no-mop areas
            DrawableElement.OBSTACLE: SupportedColor.NO_GO,  # Using NO_GO for obstacles
        }

        # Set z-index for each element type
        z_indices = {
            DrawableElement.FLOOR: 0,
            DrawableElement.WALL: 20,
            DrawableElement.ROBOT: 50,
            DrawableElement.CHARGER: 40,
            DrawableElement.VIRTUAL_WALL: 30,
            DrawableElement.RESTRICTED_AREA: 25,
            DrawableElement.NO_MOP_AREA: 25,
            DrawableElement.OBSTACLE: 15,
            DrawableElement.PATH: 35,
            DrawableElement.PREDICTED_PATH: 34,
            DrawableElement.GO_TO_TARGET: 45,
        }

        # Set properties for other elements using DefaultColors
        for element, color_key in element_color_mapping.items():
            rgb = DefaultColors.COLORS_RGB.get(color_key, (0, 0, 0))
            alpha_key = f"alpha_{color_key}"
            alpha = DefaultColors.DEFAULT_ALPHA.get(alpha_key, 255.0)

            # Special case for semi-transparent elements
            if element in [
                DrawableElement.RESTRICTED_AREA,
                DrawableElement.NO_MOP_AREA,
                DrawableElement.PREDICTED_PATH,
            ]:
                alpha = 125.0  # Semi-transparent by default

            self._element_properties[element] = {
                "color": (*rgb, int(alpha)),
                "opacity": alpha / 255.0,
                "z_index": z_indices.get(element, 0),
            }

    def enable_element(self, element_code: DrawableElement) -> None:
        """Enable drawing of a specific element."""
        if element_code in self._enabled_elements:
            self._enabled_elements[element_code] = True
            LOGGER.info(
                "Enabled element %s (%s)", element_code.name, element_code.value
            )
            LOGGER.info(
                "Element %s is now enabled: %s",
                element_code.name,
                self._enabled_elements[element_code],
            )

    def disable_element(self, element_code: DrawableElement) -> None:
        """Disable drawing of a specific element."""
        if element_code in self._enabled_elements:
            self._enabled_elements[element_code] = False
            LOGGER.info(
                "Disabled element %s (%s)", element_code.name, element_code.value
            )
            LOGGER.info(
                "Element %s is now enabled: %s",
                element_code.name,
                self._enabled_elements[element_code],
            )

    def set_elements(self, element_codes: List[DrawableElement]) -> None:
        """Enable only the specified elements, disable all others."""
        # First disable all
        for element in self._enabled_elements:
            self._enabled_elements[element] = False

        # Then enable specified ones
        for element in element_codes:
            if element in self._enabled_elements:
                self._enabled_elements[element] = True

    def is_enabled(self, element_code: DrawableElement) -> bool:
        """Check if an element is enabled for drawing."""
        return self._enabled_elements.get(element_code, False)

    def set_property(
        self, element_code: DrawableElement, property_name: str, value
    ) -> None:
        """Set a drawing property for an element."""
        if element_code in self._element_properties:
            self._element_properties[element_code][property_name] = value

    def get_property(
        self, element_code: DrawableElement, property_name: str, default=None
    ):
        """Get a drawing property for an element."""
        if element_code in self._element_properties:
            return self._element_properties[element_code].get(property_name, default)
        return default

    def get_enabled_elements(self) -> List[DrawableElement]:
        """Get list of enabled element codes."""
        return [
            element for element, enabled in self._enabled_elements.items() if enabled
        ]

    def get_drawing_order(self) -> List[DrawableElement]:
        """Get list of enabled elements in drawing order (by z_index)."""
        enabled = self.get_enabled_elements()
        return sorted(enabled, key=lambda e: self.get_property(e, "z_index", 0))

    def update_from_device_info(self, device_info: dict) -> None:
        """Update configuration based on device info dictionary."""
        # Map DrawableElement to SupportedColor
        element_color_mapping = {
            DrawableElement.FLOOR: SupportedColor.MAP_BACKGROUND,
            DrawableElement.WALL: SupportedColor.WALLS,
            DrawableElement.ROBOT: SupportedColor.ROBOT,
            DrawableElement.CHARGER: SupportedColor.CHARGER,
            DrawableElement.VIRTUAL_WALL: SupportedColor.NO_GO,
            DrawableElement.RESTRICTED_AREA: SupportedColor.NO_GO,
            DrawableElement.PATH: SupportedColor.PATH,
            DrawableElement.PREDICTED_PATH: SupportedColor.PREDICTED_PATH,
            DrawableElement.GO_TO_TARGET: SupportedColor.GO_TO,
            DrawableElement.NO_MOP_AREA: SupportedColor.NO_GO,
            DrawableElement.OBSTACLE: SupportedColor.NO_GO,
        }

        # Update room colors from device info
        for room_id in range(1, 16):
            room_element = getattr(DrawableElement, f"ROOM_{room_id}")
            color_key = SupportedColor.room_key(room_id - 1)
            alpha_key = f"alpha_room_{room_id - 1}"

            if color_key in device_info:
                rgb = device_info[color_key]
                alpha = device_info.get(alpha_key, 255.0)

                # Create RGBA color
                rgba = (*rgb, int(alpha))

                # Update color and opacity
                self.set_property(room_element, "color", rgba)
                self.set_property(room_element, "opacity", alpha / 255.0)

        # Update other element colors
        for element, color_key in element_color_mapping.items():
            if color_key in device_info:
                rgb = device_info[color_key]
                alpha_key = f"alpha_{color_key}"
                alpha = device_info.get(alpha_key, 255.0)

                # Special case for semi-transparent elements
                if element in [
                    DrawableElement.RESTRICTED_AREA,
                    DrawableElement.NO_MOP_AREA,
                    DrawableElement.PREDICTED_PATH,
                ]:
                    if alpha > 200:  # If alpha is too high for these elements
                        alpha = 125.0  # Use a more appropriate default

                # Create RGBA color
                rgba = (*rgb, int(alpha))

                # Update color and opacity
                self.set_property(element, "color", rgba)
                self.set_property(element, "opacity", alpha / 255.0)

        # Check for disabled elements using specific boolean flags
        # Map element disable flags to DrawableElement enum values
        element_disable_mapping = {
            "disable_floor": DrawableElement.FLOOR,
            "disable_wall": DrawableElement.WALL,
            "disable_robot": DrawableElement.ROBOT,
            "disable_charger": DrawableElement.CHARGER,
            "disable_virtual_walls": DrawableElement.VIRTUAL_WALL,
            "disable_restricted_areas": DrawableElement.RESTRICTED_AREA,
            "disable_no_mop_areas": DrawableElement.NO_MOP_AREA,
            "disable_obstacles": DrawableElement.OBSTACLE,
            "disable_path": DrawableElement.PATH,
            "disable_predicted_path": DrawableElement.PREDICTED_PATH,
            "disable_go_to_target": DrawableElement.GO_TO_TARGET,
        }

        # Process base element disable flags
        for disable_key, element in element_disable_mapping.items():
            if device_info.get(disable_key, False):
                self.disable_element(element)
                LOGGER.info(
                    "Disabled %s element from device_info setting", element.name
                )

        # Process room disable flags (1-15)
        for room_id in range(1, 16):
            disable_key = f"disable_room_{room_id}"
            if device_info.get(disable_key, False):
                room_element = getattr(DrawableElement, f"ROOM_{room_id}")
                self.disable_element(room_element)
                LOGGER.info(
                    "Disabled ROOM_%d element from device_info setting", room_id
                )
