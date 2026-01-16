"""Colors for the maps Elements."""

from __future__ import annotations

from enum import StrEnum
from typing import Dict, List, Tuple

import numpy as np

from ..const import (
    ALPHA_BACKGROUND,
    ALPHA_CARPET,
    ALPHA_CHARGER,
    ALPHA_GO_TO,
    ALPHA_MATERIAL_TILE,
    ALPHA_MATERIAL_WOOD,
    ALPHA_MOVE,
    ALPHA_NO_GO,
    ALPHA_ROBOT,
    ALPHA_ROOM_0,
    ALPHA_ROOM_1,
    ALPHA_ROOM_2,
    ALPHA_ROOM_3,
    ALPHA_ROOM_4,
    ALPHA_ROOM_5,
    ALPHA_ROOM_6,
    ALPHA_ROOM_7,
    ALPHA_ROOM_8,
    ALPHA_ROOM_9,
    ALPHA_ROOM_10,
    ALPHA_ROOM_11,
    ALPHA_ROOM_12,
    ALPHA_ROOM_13,
    ALPHA_ROOM_14,
    ALPHA_ROOM_15,
    ALPHA_TEXT,
    ALPHA_WALL,
    ALPHA_ZONE_CLEAN,
    COLOR_BACKGROUND,
    COLOR_CARPET,
    COLOR_CHARGER,
    COLOR_GO_TO,
    COLOR_MATERIAL_TILE,
    COLOR_MATERIAL_WOOD,
    COLOR_MOVE,
    COLOR_NO_GO,
    COLOR_ROBOT,
    COLOR_ROOM_0,
    COLOR_ROOM_1,
    COLOR_ROOM_2,
    COLOR_ROOM_3,
    COLOR_ROOM_4,
    COLOR_ROOM_5,
    COLOR_ROOM_6,
    COLOR_ROOM_7,
    COLOR_ROOM_8,
    COLOR_ROOM_9,
    COLOR_ROOM_10,
    COLOR_ROOM_11,
    COLOR_ROOM_12,
    COLOR_ROOM_13,
    COLOR_ROOM_14,
    COLOR_ROOM_15,
    COLOR_TEXT,
    COLOR_WALL,
    COLOR_ZONE_CLEAN,
)
from .types import LOGGER, Color


color_transparent = (0, 0, 0, 0)
color_charger = (0, 128, 0, 255)
color_carpet = (67, 103, 125, 255)
color_move = (238, 247, 255, 255)
color_robot = (255, 255, 204, 255)
color_no_go = (255, 0, 0, 255)
color_go_to = (0, 255, 0, 255)
color_background = (0, 125, 255, 255)
color_zone_clean = (255, 255, 255, 125)
color_wall = (255, 255, 0, 255)
color_text = (255, 255, 255, 255)
color_grey = (125, 125, 125, 255)
color_black = (0, 0, 0, 255)
color_material_wood = (40, 40, 40, 38)
color_material_tile = (40, 40, 40, 45)
color_room_0 = (135, 206, 250, 255)
color_room_1 = (176, 226, 255, 255)
color_room_2 = (164, 211, 238, 255)
color_room_3 = (141, 182, 205, 255)
color_room_4 = (96, 123, 139, 255)
color_room_5 = (224, 255, 255, 255)
color_room_6 = (209, 238, 238, 255)
color_room_7 = (180, 205, 205, 255)
color_room_8 = (122, 139, 139, 255)
color_room_9 = (175, 238, 238, 255)
color_room_10 = (84, 153, 199, 255)
color_room_11 = (133, 193, 233, 255)
color_room_12 = (245, 176, 65, 255)
color_room_13 = (82, 190, 128, 255)
color_room_14 = (72, 201, 176, 255)
color_room_15 = (165, 105, 18, 255)

rooms_color = [
    color_room_0,
    color_room_1,
    color_room_2,
    color_room_3,
    color_room_4,
    color_room_5,
    color_room_6,
    color_room_7,
    color_room_8,
    color_room_9,
    color_room_10,
    color_room_11,
    color_room_12,
    color_room_13,
    color_room_14,
    color_room_15,
]

base_colors_array = [
    color_wall,
    color_zone_clean,
    color_carpet,
    color_material_wood,
    color_material_tile,
    color_robot,
    color_background,
    color_move,
    color_charger,
    color_no_go,
    color_go_to,
    color_text,
]

color_array = [
    base_colors_array[0],  # color_wall
    base_colors_array[6],  # color_no_go
    base_colors_array[7],  # color_go_to
    base_colors_array[8],  # color_predicted_path
    base_colors_array[9],  # color_obstacle
    color_black,
    base_colors_array[2],  # color_robot
    base_colors_array[5],  # color_charger
    color_text,
    base_colors_array[4],  # color_move
    base_colors_array[3],  # color_background
    base_colors_array[1],  # color_zone_clean
    color_transparent,
    rooms_color,
]


class SupportedColor(StrEnum):
    """Color of a supported map element."""

    CHARGER = "color_charger"
    PATH = "color_move"
    PREDICTED_PATH = "color_predicted_move"
    WALLS = "color_wall"
    ROBOT = "color_robot"
    GO_TO = "color_go_to"
    NO_GO = "color_no_go"
    ZONE_CLEAN = "color_zone_clean"
    CARPET = "color_carpet"
    OBSTACLE = "color_obstacle"
    TILE = "color_material_tile"
    WOOD = "color_material_wood"
    MAP_BACKGROUND = "color_background"
    TEXT = "color_text"
    TRANSPARENT = "color_transparent"
    COLOR_ROOM_PREFIX = "color_room_"

    @staticmethod
    def room_key(index: int) -> str:
        return f"{SupportedColor.COLOR_ROOM_PREFIX}{index}"


class DefaultColors:
    """Container that simplifies retrieving default RGB and RGBA colors."""

    COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
        SupportedColor.CHARGER: (255, 128, 0),
        SupportedColor.PATH: (50, 150, 255),  # More vibrant blue for better visibility
        SupportedColor.PREDICTED_PATH: (93, 109, 126),
        SupportedColor.WALLS: (255, 255, 0),
        SupportedColor.ROBOT: (255, 255, 204),
        SupportedColor.GO_TO: (0, 255, 0),
        SupportedColor.NO_GO: (255, 0, 0),
        SupportedColor.ZONE_CLEAN: (255, 255, 255),
        SupportedColor.CARPET: (67, 103, 125),
        SupportedColor.OBSTACLE: (255, 0, 0),
        SupportedColor.TILE: (40, 40, 40),
        SupportedColor.WOOD: (40, 40, 40),
        SupportedColor.MAP_BACKGROUND: (0, 125, 255),
        SupportedColor.TEXT: (0, 0, 0),
        SupportedColor.TRANSPARENT: (0, 0, 0),
    }

    DEFAULT_ROOM_COLORS: Dict[str, Tuple[int, int, int]] = {
        SupportedColor.room_key(i): color
        for i, color in enumerate(
            [
                (135, 206, 250),
                (176, 226, 255),
                (165, 105, 18),
                (164, 211, 238),
                (141, 182, 205),
                (96, 123, 139),
                (224, 255, 255),
                (209, 238, 238),
                (180, 205, 205),
                (122, 139, 139),
                (175, 238, 238),
                (84, 153, 199),
                (133, 193, 233),
                (245, 176, 65),
                (82, 190, 128),
                (72, 201, 176),
            ]
        )
    }

    DEFAULT_ALPHA: Dict[str, float] = {
        f"alpha_{key}": 255.0 for key in COLORS_RGB.keys()
    }
    # Override specific alpha values
    DEFAULT_ALPHA.update(
        {
            "alpha_color_path": 200.0,  # Make path slightly transparent but still very visible
            "alpha_color_wall": 150.0,  # Keep walls semi-transparent
        }
    )
    DEFAULT_ALPHA.update({f"alpha_room_{i}": 255.0 for i in range(16)})

    @classmethod
    def get_rgba(cls, key: str, alpha: float) -> Color:
        rgb = cls.COLORS_RGB.get(key, (0, 0, 0))
        r, g, b = rgb  # Explicitly unpack the RGB values
        return r, g, b, int(alpha)


class ColorsManagement:
    """Manages user-defined and default colors for map elements."""

    def __init__(self, shared_var) -> None:
        """
        Initialize ColorsManagement for Home Assistant.
        Uses optimized initialization for better performance.
        """
        self.shared_var = shared_var
        self.color_cache = {}  # Cache for frequently used color blends

        # Initialize colors efficiently
        self.user_colors = self.initialize_user_colors(self.shared_var.device_info)
        self.rooms_colors = self.initialize_rooms_colors(self.shared_var.device_info)

    @staticmethod
    def add_alpha_to_rgb(alpha_channels, rgb_colors):
        """
        Add alpha channel to RGB colors using corresponding alpha channels.
        Uses NumPy for vectorized operations when possible for better performance.

        Args:
            alpha_channels (List[Optional[float]]): List of alpha channel values (0.0-255.0).
            rgb_colors (List[Tuple[int, int, int]]): List of RGB colors.

        Returns:
            List[Tuple[int, int, int, int]]: List of RGBA colors with alpha channel added.
        """
        if len(alpha_channels) != len(rgb_colors):
            LOGGER.warning("Input lists must have the same length.")
            return []

        # Fast path for empty lists
        if not rgb_colors:
            return []

        # Try to use NumPy for vectorized operations
        try:
            # Convert inputs to NumPy arrays for vectorized processing
            alphas = np.array(alpha_channels, dtype=np.float32)

            # Clip alpha values to valid range [0, 255]
            alphas = np.clip(alphas, 0, 255).astype(np.int32)

            # Process RGB colors
            result = []
            for _, (alpha, rgb) in enumerate(zip(alphas, rgb_colors)):
                if rgb is None:
                    result.append((0, 0, 0, int(alpha)))
                else:
                    result.append((rgb[0], rgb[1], rgb[2], int(alpha)))

            return result

        except (ValueError, TypeError, AttributeError):
            # Fallback to non-vectorized method if NumPy processing fails
            result = []
            for alpha, rgb in zip(alpha_channels, rgb_colors):
                try:
                    alpha_int = int(alpha)
                    alpha_int = max(0, min(255, alpha_int))  # Clip to valid range

                    if rgb is None:
                        result.append((0, 0, 0, alpha_int))
                    else:
                        result.append((rgb[0], rgb[1], rgb[2], alpha_int))
                except (ValueError, TypeError):
                    result.append(None)

            return result

    def set_initial_colours(self, device_info: dict) -> None:
        """Set the initial colours for the map using optimized methods."""
        try:
            # Define color keys and default values
            base_color_keys = [
                (COLOR_WALL, color_wall, ALPHA_WALL),
                (COLOR_ZONE_CLEAN, color_zone_clean, ALPHA_ZONE_CLEAN),
                (COLOR_ROBOT, color_robot, ALPHA_ROBOT),
                (COLOR_BACKGROUND, color_background, ALPHA_BACKGROUND),
                (COLOR_MOVE, color_move, ALPHA_MOVE),
                (COLOR_CHARGER, color_charger, ALPHA_CHARGER),
                (COLOR_CARPET, color_carpet, ALPHA_CARPET),
                (COLOR_NO_GO, color_no_go, ALPHA_NO_GO),
                (COLOR_GO_TO, color_go_to, ALPHA_GO_TO),
                (COLOR_TEXT, color_text, ALPHA_TEXT),
                (COLOR_MATERIAL_WOOD, color_material_wood, ALPHA_MATERIAL_WOOD),
                (COLOR_MATERIAL_TILE, color_material_tile, ALPHA_MATERIAL_TILE),
            ]

            room_color_keys = [
                (COLOR_ROOM_0, color_room_0, ALPHA_ROOM_0),
                (COLOR_ROOM_1, color_room_1, ALPHA_ROOM_1),
                (COLOR_ROOM_2, color_room_2, ALPHA_ROOM_2),
                (COLOR_ROOM_3, color_room_3, ALPHA_ROOM_3),
                (COLOR_ROOM_4, color_room_4, ALPHA_ROOM_4),
                (COLOR_ROOM_5, color_room_5, ALPHA_ROOM_5),
                (COLOR_ROOM_6, color_room_6, ALPHA_ROOM_6),
                (COLOR_ROOM_7, color_room_7, ALPHA_ROOM_7),
                (COLOR_ROOM_8, color_room_8, ALPHA_ROOM_8),
                (COLOR_ROOM_9, color_room_9, ALPHA_ROOM_9),
                (COLOR_ROOM_10, color_room_10, ALPHA_ROOM_10),
                (COLOR_ROOM_11, color_room_11, ALPHA_ROOM_11),
                (COLOR_ROOM_12, color_room_12, ALPHA_ROOM_12),
                (COLOR_ROOM_13, color_room_13, ALPHA_ROOM_13),
                (COLOR_ROOM_14, color_room_14, ALPHA_ROOM_14),
                (COLOR_ROOM_15, color_room_15, ALPHA_ROOM_15),
            ]

            # Extract user colors and alphas efficiently
            user_colors = [
                device_info.get(color_key, default_color)
                for color_key, default_color, _ in base_color_keys
            ]
            user_alpha = [
                device_info.get(alpha_key, 255) for _, _, alpha_key in base_color_keys
            ]

            # Extract room colors and alphas efficiently
            rooms_colors = [
                device_info.get(color_key, default_color)
                for color_key, default_color, _ in room_color_keys
            ]
            rooms_alpha = [
                device_info.get(alpha_key, 255) for _, _, alpha_key in room_color_keys
            ]

            # Use our optimized add_alpha_to_rgb method
            self.shared_var.update_user_colors(
                self.add_alpha_to_rgb(user_alpha, user_colors)
            )
            self.shared_var.update_rooms_colors(
                self.add_alpha_to_rgb(rooms_alpha, rooms_colors)
            )

            # Clear the color cache after initialization
            self.color_cache.clear()

        except (ValueError, IndexError, UnboundLocalError) as e:
            LOGGER.warning("Error while populating colors: %s", e)

    def initialize_user_colors(self, device_info: dict) -> List[Color]:
        """
        Initialize user-defined colors with defaults as fallback.
        :param device_info: Dictionary containing user-defined colors.
        :return: List of RGBA colors for map elements.
        """
        colors = []
        for key in SupportedColor:
            if key.startswith(SupportedColor.COLOR_ROOM_PREFIX):
                continue  # Skip room colors for user_colors
            rgb = device_info.get(key, DefaultColors.COLORS_RGB.get(key))
            alpha = device_info.get(
                f"alpha_{key}", DefaultColors.DEFAULT_ALPHA.get(f"alpha_{key}")
            )
            colors.append(self.add_alpha_to_color(rgb, alpha))
        return colors

    def initialize_rooms_colors(self, device_info: dict) -> List[Color]:
        """
        Initialize room colors with defaults as fallback.
        :param device_info: Dictionary containing user-defined room colors.
        :return: List of RGBA colors for rooms.
        """
        colors = []
        for i in range(16):
            rgb = device_info.get(
                SupportedColor.room_key(i),
                DefaultColors.DEFAULT_ROOM_COLORS.get(SupportedColor.room_key(i)),
            )
            alpha = device_info.get(
                f"alpha_room_{i}", DefaultColors.DEFAULT_ALPHA.get(f"alpha_room_{i}")
            )
            colors.append(self.add_alpha_to_color(rgb, alpha))
        return colors

    @staticmethod
    def add_alpha_to_color(rgb: Tuple[int, int, int], alpha: float) -> Color:
        """
        Convert RGB to RGBA by appending the alpha value.
        :param rgb: RGB values.
        :param alpha: Alpha value (0.0 to 255.0).
        :return: RGBA color.
        """
        return (*rgb, int(alpha)) if rgb else (0, 0, 0, int(alpha))

    def get_user_colors(self) -> List[Color]:
        """Return the list of RGBA colors for user-defined map elements."""
        return self.user_colors

    def get_rooms_colors(self) -> List[Color]:
        """Return the list of RGBA colors for rooms."""
        return self.rooms_colors
