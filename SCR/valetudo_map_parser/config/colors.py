"""Colors for the maps Elements."""

from __future__ import annotations

from enum import StrEnum
from typing import Dict, List, Tuple

from .types import LOGGER, Color


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
    DEFAULT_ALPHA.update({
        "alpha_color_path": 200.0,  # Make path slightly transparent but still very visible
        "alpha_color_wall": 150.0,  # Keep walls semi-transparent
    })
    DEFAULT_ALPHA.update({f"alpha_room_{i}": 255.0 for i in range(16)})

    @classmethod
    def get_rgba(cls, key: str, alpha: float) -> Color:
        rgb = cls.COLORS_RGB.get(key, (0, 0, 0))
        r, g, b = rgb  # Explicitly unpack the RGB values
        return r, g, b, int(alpha)


class ColorsManagment:
    """Manages user-defined and default colors for map elements."""

    def __init__(self, device_info: dict) -> None:
        """
        Initialize ColorsManagment with optional device_info from Home Assistant.
        :param device_info: Dictionary containing user-defined RGB colors and alpha values.
        """
        self.user_colors = self.initialize_user_colors(device_info)
        self.rooms_colors = self.initialize_rooms_colors(device_info)

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

    @staticmethod
    def blend_colors(background: Color, foreground: Color) -> Color:
        """
        Blend foreground color with background color based on alpha values.

        This is used when drawing elements that overlap on the map.
        The alpha channel determines how much of the foreground color is visible.

        :param background: Background RGBA color (r,g,b,a)
        :param foreground: Foreground RGBA color (r,g,b,a) to blend on top
        :return: Blended RGBA color
        """
        # Extract components
        bg_r, bg_g, bg_b, bg_a = background
        fg_r, fg_g, fg_b, fg_a = foreground

        # If foreground is fully opaque, just return it
        if fg_a == 255:
            return foreground

        # If foreground is fully transparent, return background
        if fg_a == 0:
            return background

        # Calculate alpha blending
        # Convert alpha from [0-255] to [0-1] for calculations
        fg_alpha = fg_a / 255.0
        bg_alpha = bg_a / 255.0

        # Calculate resulting alpha
        out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

        # Avoid division by zero
        if out_alpha < 0.0001:
            return (0, 0, 0, 0)  # Fully transparent result

        # Calculate blended RGB components
        out_r = int((fg_r * fg_alpha + bg_r * bg_alpha * (1 - fg_alpha)) / out_alpha)
        out_g = int((fg_g * fg_alpha + bg_g * bg_alpha * (1 - fg_alpha)) / out_alpha)
        out_b = int((fg_b * fg_alpha + bg_b * bg_alpha * (1 - fg_alpha)) / out_alpha)

        # Convert alpha back to [0-255] range
        out_a = int(out_alpha * 255)

        # Ensure values are in valid range
        out_r = max(0, min(255, out_r))
        out_g = max(0, min(255, out_g))
        out_b = max(0, min(255, out_b))

        return (out_r, out_g, out_b, out_a)

    @staticmethod
    def sample_and_blend_color(array, x: int, y: int, foreground: Color) -> Color:
        """
        Sample the background color from the array at coordinates (x,y) and blend with foreground color.

        Args:
            array: The RGBA numpy array representing the image
            x, y: Coordinates to sample the background color from
            foreground: Foreground RGBA color (r,g,b,a) to blend on top

        Returns:
            Blended RGBA color
        """
        # Ensure coordinates are within bounds
        if array is None:
            return foreground

        height, width = array.shape[:2]
        if not (0 <= y < height and 0 <= x < width):
            return foreground  # Return foreground if coordinates are out of bounds

        # Sample background color from the array
        # The array is in RGBA format with shape (height, width, 4)
        background = tuple(array[y, x])

        # Blend the colors
        return ColorsManagment.blend_colors(background, foreground)

    def get_user_colors(self) -> List[Color]:
        """Return the list of RGBA colors for user-defined map elements."""
        return self.user_colors

    def get_rooms_colors(self) -> List[Color]:
        """Return the list of RGBA colors for rooms."""
        return self.rooms_colors

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
                LOGGER.warning("Room index %s not found, using default.", room_index)
                r, g, b = DefaultColors.DEFAULT_ROOM_COLORS[f"color_room_{room_index}"]
                a = DefaultColors.DEFAULT_ALPHA[f"alpha_room_{room_index}"]
                return r, g, b, int(a)

        # Handle general map element colors
        try:
            index = list(SupportedColor).index(supported_color)
            return self.user_colors[index]
        except (IndexError, KeyError, ValueError):
            LOGGER.warning(
                "Color for %s not found. Returning default.", supported_color
            )
            return DefaultColors.get_rgba(supported_color, 255)  # Transparent fallback
