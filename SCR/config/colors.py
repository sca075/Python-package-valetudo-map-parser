"""Colors for the maps Elements."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar, Union
from .types import Color
import logging

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class SupportedColor(StrEnum):
    """Color of a supported map element."""

    DEFAULT_ROOM_COLORS = {
        f"color_room_{i}": color
        for i, color in enumerate(
            [
                [135, 206, 250],
                [176, 226, 255],
                [165, 105, 18],
                [164, 211, 238],
                [141, 182, 205],
                [96, 123, 139],
                [224, 255, 255],
                [209, 238, 238],
                [180, 205, 205],
                [122, 139, 139],
                [175, 238, 238],
                [84, 153, 199],
                [133, 193, 233],
                [245, 176, 65],
                [82, 190, 128],
                [72, 201, 176],
            ]
        )
    }

    DEFAULT_ALPHA = {f"alpha_room_{i}": 255.0 for i in range(16)}

    CHARGER = "color_charger"
    PATH = "color_move"
    PREDICTED_PATH = "color_predicted_move"
    WALLS = "color_wall"
    ROBOT = "color_robot"
    GO_TO = "color_go_to"
    NO_GO = "color_no_go"
    ZONE_CLEAN = "color_zone_clean"
    MAP_BACKGROUND = "color_background"
    TEXT = "color_text"
    TRANSPARENT = "color_transparent"
    ROOMS_LIST_KEYS = list(dict(DEFAULT_ROOM_COLORS).keys())
    DEF_ALPHA_LIST_KEYS = list(dict(DEFAULT_ALPHA).keys())

    "Rooms Colours RGB"
    COLOR_ROOM_0 = "color_room_0"
    COLOR_ROOM_1 = "color_room_1"
    COLOR_ROOM_2 = "color_room_2"
    COLOR_ROOM_3 = "color_room_3"
    COLOR_ROOM_4 = "color_room_4"
    COLOR_ROOM_5 = "color_room_5"
    COLOR_ROOM_6 = "color_room_6"
    COLOR_ROOM_7 = "color_room_7"
    COLOR_ROOM_8 = "color_room_8"
    COLOR_ROOM_9 = "color_room_9"
    COLOR_ROOM_10 = "color_room_10"
    COLOR_ROOM_11 = "color_room_11"
    COLOR_ROOM_12 = "color_room_12"
    COLOR_ROOM_13 = "color_room_13"
    COLOR_ROOM_14 = "color_room_14"
    COLOR_ROOM_15 = "color_room_15"

    """Alpha for RGBA Colours"""
    ALPHA_CHARGER = "alpha_charger"
    ALPHA_MOVE = "alpha_move"
    ALPHA_ROBOT = "alpha_robot"
    ALPHA_NO_GO = "alpha_no_go"
    ALPHA_GO_TO = "alpha_go_to"
    ALPHA_BACKGROUND = "alpha_background"
    ALPHA_ZONE_CLEAN = "alpha_zone_clean"
    ALPHA_WALL = "alpha_wall"
    ALPHA_TEXT = "alpha_text"
    ALPHA_ROOM_0 = "alpha_room_0"
    ALPHA_ROOM_1 = "alpha_room_1"
    ALPHA_ROOM_2 = "alpha_room_2"
    ALPHA_ROOM_3 = "alpha_room_3"
    ALPHA_ROOM_4 = "alpha_room_4"
    ALPHA_ROOM_5 = "alpha_room_5"
    ALPHA_ROOM_6 = "alpha_room_6"
    ALPHA_ROOM_7 = "alpha_room_7"
    ALPHA_ROOM_8 = "alpha_room_8"
    ALPHA_ROOM_9 = "alpha_room_9"
    ALPHA_ROOM_10 = "alpha_room_10"
    ALPHA_ROOM_11 = "alpha_room_11"
    ALPHA_ROOM_12 = "alpha_room_12"
    ALPHA_ROOM_13 = "alpha_room_13"
    ALPHA_ROOM_14 = "alpha_room_14"
    ALPHA_ROOM_15 = "alpha_room_15"


class DefaultColors:
    """Container that simplifies retrieving desired color."""

    COLORS_RGBA: dict[SupportedColor, Color] = {
        SupportedColor.CHARGER: [
            255,
            128,
            0,
            SupportedColor.DEFAULT_ALPHA["alpha_charger"],
        ],
        SupportedColor.PATH: [
            238,
            247,
            255,
            SupportedColor.DEFAULT_ALPHA["alpha_move"],
        ],
        SupportedColor.PREDICTED_PATH: [
            93,
            109,
            126,
            SupportedColor.DEFAULT_ALPHA["alpha_move"],
        ],
        SupportedColor.WALLS: [255, 255, 0, SupportedColor.DEFAULT_ALPHA["alpha_wall"]],
        SupportedColor.ROBOT: [
            255,
            255,
            204,
            SupportedColor.DEFAULT_ALPHA["alpha_robot"],
        ],
        SupportedColor.GO_TO: [0, 255, 0, SupportedColor.DEFAULT_ALPHA["alpha_go_to"]],
        SupportedColor.NO_GO: [255, 0, 0, SupportedColor.DEFAULT_ALPHA["alpha_no_go"]],
        SupportedColor.ZONE_CLEAN: [
            255,
            255,
            255,
            SupportedColor.DEFAULT_ALPHA["alpha_zone_clean"],
        ],
        SupportedColor.MAP_BACKGROUND: [
            0,
            125,
            255,
            SupportedColor.DEFAULT_ALPHA["alpha_background"],
        ],
        SupportedColor.TEXT: [0, 0, 0, SupportedColor.DEFAULT_ALPHA["alpha_text"]],
        SupportedColor.TRANSPARENT: [0, 0, 0, 0],
    }
    COLORS_RGB: dict[SupportedColor, Color] = {
        SupportedColor.CHARGER: [255, 128, 0],
        SupportedColor.PATH: [238, 247, 255],
        SupportedColor.PREDICTED_PATH: [93, 109, 126],
        SupportedColor.WALLS: [255, 255, 0, SupportedColor.DEFAULT_ALPHA["alpha_wall"]],
        SupportedColor.ROBOT: [255, 255, 204],
        SupportedColor.GO_TO: [0, 255, 0],
        SupportedColor.NO_GO: [255, 0, 0],
        SupportedColor.ZONE_CLEAN: [
            255,
            255,
            255,
        ],
        SupportedColor.MAP_BACKGROUND: [
            0,
            125,
            255,
            SupportedColor.DEFAULT_ALPHA["alpha_background"],
        ],
        SupportedColor.TEXT: [0, 0, 0],
        SupportedColor.TRANSPARENT: [0, 0, 0],
    }


class ColorsManagment:
    """Class to manage colors for the map elements.
    Imports and updates colors dynamically based on user configuration.
    """

    def __init__(self, device_info: dict = None):
        """
        Initialize ColorsManagment with optional user configuration.

        :param device_info: Optional dictionary containing user-defined colors and alphas.
        """
        self.user_colors = {}
        self.rooms_colors = []
        if device_info:
            self.set_initial_colours(device_info)

    @staticmethod
    def add_alpha_to_rgb(alpha_channels, rgb_colors):
        """
        Add alpha channel to RGB colors using corresponding alpha channels.

        Args:
            alpha_channels (List[Optional[float]]): List of alpha channel values (0.0-255.0).
            rgb_colors (List[Tuple[int, int, int]]): List of RGB colors.

        Returns:
            List[Tuple[int, int, int, int]]: List of RGBA colors with alpha channel added.
        """
        if (
            not alpha_channels
            or not rgb_colors
            or len(alpha_channels) != len(rgb_colors)
        ):
            raise ValueError(
                "Alpha channels and RGB colors must have the same non-zero length."
            )

        result = []
        for alpha, rgb in zip(alpha_channels, rgb_colors):
            try:
                alpha_int = max(
                    0, min(255, int(alpha))
                )  # Clamp alpha between 0 and 255
                if rgb is None:
                    result.append((0, 0, 0, alpha_int))
                else:
                    result.append((rgb[0], rgb[1], rgb[2], alpha_int))
            except (ValueError, TypeError):
                result.append((0, 0, 0, 0))  # Transparent fallback for invalid input

        return result

    def set_initial_colours(self, device_info: dict) -> None:
        """Set the initial colors and alphas for the map elements."""
        try:
            # User-defined map element colors
            user_colors = [
                device_info.get(SupportedColor.WALLS),
                device_info.get(SupportedColor.ZONE_CLEAN),
                device_info.get(SupportedColor.ROBOT),
                device_info.get(SupportedColor.MAP_BACKGROUND),
                device_info.get(SupportedColor.PATH),
                device_info.get(SupportedColor.CHARGER),
                device_info.get(SupportedColor.NO_GO),
                device_info.get(SupportedColor.GO_TO),
                device_info.get(SupportedColor.TEXT),
            ]
            user_alpha = [
                device_info.get(SupportedColor.ALPHA_WALL),
                device_info.get(SupportedColor.ALPHA_ZONE_CLEAN),
                device_info.get(SupportedColor.ALPHA_ROBOT),
                device_info.get(SupportedColor.ALPHA_BACKGROUND),
                device_info.get(SupportedColor.ALPHA_MOVE),
                device_info.get(SupportedColor.ALPHA_CHARGER),
                device_info.get(SupportedColor.ALPHA_NO_GO),
                device_info.get(SupportedColor.ALPHA_GO_TO),
                device_info.get(SupportedColor.ALPHA_TEXT),
            ]
            # User-defined room colors
            rooms_colors = [device_info.get(f"color_room_{i}") for i in range(16)]
            rooms_alpha = [device_info.get(f"alpha_room_{i}") for i in range(16)]

            # Initialize internal state
            self.user_colors = self.add_alpha_to_rgb(user_alpha, user_colors)
            self.rooms_colors = self.add_alpha_to_rgb(rooms_alpha, rooms_colors)

        except (ValueError, IndexError, UnboundLocalError) as e:
            _LOGGER.error("Error while populating colors: %s", e)
            self.user_colors = []
            self.rooms_colors = []

    def get_colour(self, supported_color: SupportedColor) -> Color:
        """
        Retrieve the color for a specific map element, prioritizing user-defined values.

        :param supported_color: The SupportedColor key for the desired color.
        :return: The RGBA color for the given map element.
        """
        # Handle room-specific colors
        if supported_color.startswith("color_room_"):
            room_index = int(supported_color.split("_")[-1])
            try:
                return self.rooms_colors[room_index]
            except (IndexError, KeyError):
                _LOGGER.warning(f"Room index {room_index} not found, using default.")
                return SupportedColor.DEFAULT_ROOM_COLORS[f"color_room_{room_index}"]

        # Handle general map element colors
        try:
            index = list(SupportedColor).index(supported_color)
            return self.user_colors[index]
        except (IndexError, KeyError, ValueError):
            _LOGGER.warning(
                f"Color for {supported_color} not found. Returning default."
            )
            return DefaultColors.COLORS_RGBA.get(
                supported_color, [0, 0, 0, 255]
            )  # Transparent fallback

    def get_rooms_colours(self) -> list[Color]:
        """
        Retrieve the list of colors for all rooms, prioritizing user-defined values.
        """
        return self.rooms_colors or [
            color for color in SupportedColor.DEFAULT_ROOM_COLORS.values()
        ]
