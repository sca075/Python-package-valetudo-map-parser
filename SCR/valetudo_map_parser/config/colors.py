"""Colors for the maps Elements."""

from __future__ import annotations

from enum import StrEnum
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .types import (
    ALPHA_BACKGROUND,
    ALPHA_CHARGER,
    ALPHA_GO_TO,
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
    COLOR_CHARGER,
    COLOR_GO_TO,
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
    LOGGER,
    Color,
)


color_transparent = (0, 0, 0, 0)
color_charger = (0, 128, 0, 255)
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
            LOGGER.error("Input lists must have the same length.")
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
                (COLOR_NO_GO, color_no_go, ALPHA_NO_GO),
                (COLOR_GO_TO, color_go_to, ALPHA_GO_TO),
                (COLOR_TEXT, color_text, ALPHA_TEXT),
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
            LOGGER.error("Error while populating colors: %s", e)

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
        Optimized version with more fast paths and simplified calculations.

        :param background: Background RGBA color (r,g,b,a)
        :param foreground: Foreground RGBA color (r,g,b,a) to blend on top
        :return: Blended RGBA color
        """
        # Fast paths for common cases
        fg_a = foreground[3]

        if fg_a == 255:  # Fully opaque foreground
            return foreground

        if fg_a == 0:  # Fully transparent foreground
            return background

        bg_a = background[3]
        if bg_a == 0:  # Fully transparent background
            return foreground

        # Extract components (only after fast paths)
        bg_r, bg_g, bg_b = background[:3]
        fg_r, fg_g, fg_b = foreground[:3]

        # Pre-calculate the blend factor once (avoid repeated division)
        blend = fg_a / 255.0
        inv_blend = 1.0 - blend

        # Simple linear interpolation for RGB channels
        # This is faster than the previous implementation
        out_r = int(fg_r * blend + bg_r * inv_blend)
        out_g = int(fg_g * blend + bg_g * inv_blend)
        out_b = int(fg_b * blend + bg_b * inv_blend)

        # Alpha blending - simplified calculation
        out_a = int(fg_a + bg_a * inv_blend)

        # No need for min/max checks as the blend math keeps values in range
        # when input values are valid (0-255)

        return [out_r, out_g, out_b, out_a]

    # Cache for recently sampled background colors
    _bg_color_cache = {}
    _cache_size = 1024  # Limit cache size to avoid memory issues

    @staticmethod
    def sample_and_blend_color(array, x: int, y: int, foreground: Color) -> Color:
        """
        Sample the background color from the array at coordinates (x,y) and blend with foreground color.
        Optimized version with caching and faster sampling.

        Args:
            array: The RGBA numpy array representing the image
            x: Coordinate X to sample the background color from
            y: Coordinate Y to sample the background color from
            foreground: Foreground RGBA color (r,g,b,a) to blend on top

        Returns:
            Blended RGBA color
        """
        # Fast path for fully opaque foreground - no need to sample or blend
        if foreground[3] == 255:
            return foreground

        # Ensure array exists
        if array is None:
            return foreground

        # Check if coordinates are within bounds
        height, width = array.shape[:2]
        if not (0 <= y < height and 0 <= x < width):
            return foreground

        # Check cache for this coordinate
        cache_key = (id(array), x, y)
        cache = ColorsManagement._bg_color_cache

        if cache_key in cache:
            background = cache[cache_key]
        else:
            # Sample the background color using direct indexing (fastest method)
            try:
                background = tuple(map(int, array[y, x]))

                # Update cache (with simple LRU-like behavior)
                try:
                    if len(cache) >= ColorsManagement._cache_size:
                        # Remove a random entry if cache is full
                        if cache:  # Make sure cache is not empty
                            cache.pop(next(iter(cache)))
                        else:
                            # If cache is somehow empty but len reported >= _cache_size
                            # This is an edge case that shouldn't happen but we handle it
                            pass
                    cache[cache_key] = background
                except KeyError:
                    # If we encounter a KeyError, reset the cache
                    # This is a rare edge case that might happen in concurrent access
                    ColorsManagement._bg_color_cache = {cache_key: background}

            except (IndexError, ValueError):
                return foreground

        # Fast path for fully transparent foreground
        if foreground[3] == 0:
            return background

        # Blend the colors
        return ColorsManagement.blend_colors(background, foreground)

    def get_user_colors(self) -> List[Color]:
        """Return the list of RGBA colors for user-defined map elements."""
        return self.user_colors

    def get_rooms_colors(self) -> List[Color]:
        """Return the list of RGBA colors for rooms."""
        return self.rooms_colors

    @staticmethod
    def batch_blend_colors(image_array, mask, foreground_color):
        """
        Blend a foreground color with all pixels in an image where the mask is True.
        Uses scipy.ndimage for efficient batch processing.

        Args:
            image_array: NumPy array of shape (height, width, 4) containing RGBA image data
            mask: Boolean mask of shape (height, width) indicating pixels to blend
            foreground_color: RGBA color tuple to blend with the masked pixels

        Returns:
            Modified image array with blended colors
        """
        if not np.any(mask):
            return image_array  # No pixels to blend

        # Extract foreground components
        fg_r, fg_g, fg_b, fg_a = foreground_color

        # Fast path for fully opaque foreground
        if fg_a == 255:
            # Just set the color directly where mask is True
            image_array[mask, 0] = fg_r
            image_array[mask, 1] = fg_g
            image_array[mask, 2] = fg_b
            image_array[mask, 3] = fg_a
            return image_array

        # Fast path for fully transparent foreground
        if fg_a == 0:
            return image_array  # No change needed

        # For semi-transparent foreground, we need to blend
        # Extract background components where mask is True
        bg_pixels = image_array[mask]

        # Convert alpha from [0-255] to [0-1] for calculations
        fg_alpha = fg_a / 255.0
        bg_alpha = bg_pixels[:, 3] / 255.0

        # Calculate resulting alpha
        out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

        # Calculate alpha ratios for blending
        # Handle division by zero by setting ratio to 0 where out_alpha is near zero
        alpha_ratio = np.zeros_like(out_alpha)
        valid_alpha = out_alpha > 0.0001
        alpha_ratio[valid_alpha] = fg_alpha / out_alpha[valid_alpha]
        inv_alpha_ratio = 1.0 - alpha_ratio

        # Calculate blended RGB components
        out_r = np.clip(
            (fg_r * alpha_ratio + bg_pixels[:, 0] * inv_alpha_ratio), 0, 255
        ).astype(np.uint8)
        out_g = np.clip(
            (fg_g * alpha_ratio + bg_pixels[:, 1] * inv_alpha_ratio), 0, 255
        ).astype(np.uint8)
        out_b = np.clip(
            (fg_b * alpha_ratio + bg_pixels[:, 2] * inv_alpha_ratio), 0, 255
        ).astype(np.uint8)
        out_a = np.clip((out_alpha * 255), 0, 255).astype(np.uint8)

        # Update the image array with blended values
        image_array[mask, 0] = out_r
        image_array[mask, 1] = out_g
        image_array[mask, 2] = out_b
        image_array[mask, 3] = out_a

        return image_array

    @staticmethod
    def process_regions_with_colors(image_array, regions_mask, colors):
        """
        Process multiple regions in an image with different colors using scipy.ndimage.
        This is much faster than processing each region separately.

        Args:
            image_array: NumPy array of shape (height, width, 4) containing RGBA image data
            regions_mask: NumPy array of shape (height, width) with integer labels for different regions
            colors: List of RGBA color tuples corresponding to each region label

        Returns:
            Modified image array with all regions colored and blended
        """
        # Skip processing if no regions or colors
        if regions_mask is None or not np.any(regions_mask) or not colors:
            return image_array

        # Get unique region labels (excluding 0 which is typically background)
        unique_labels = np.unique(regions_mask)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background (0)

        if len(unique_labels) == 0:
            return image_array  # No regions to process

        # Process each region with its corresponding color
        for label in unique_labels:
            if label <= len(colors):
                # Create mask for this region
                region_mask = regions_mask == label

                # Get color for this region
                color = colors[label - 1] if label - 1 < len(colors) else colors[0]

                # Apply color to this region
                image_array = ColorsManagement.batch_blend_colors(
                    image_array, region_mask, color
                )

        return image_array

    @staticmethod
    def apply_color_to_shapes(image_array, shapes, color, thickness=1):
        """
        Apply a color to multiple shapes (lines, circles, etc.) using scipy.ndimage.

        Args:
            image_array: NumPy array of shape (height, width, 4) containing RGBA image data
            shapes: List of shape definitions (each a list of points or parameters)
            color: RGBA color tuple to apply to the shapes
            thickness: Line thickness for shapes

        Returns:
            Modified image array with shapes drawn and blended
        """
        height, width = image_array.shape[:2]

        # Create a mask for all shapes
        shapes_mask = np.zeros((height, width), dtype=bool)

        # Draw all shapes into the mask
        for shape in shapes:
            if len(shape) >= 2:  # At least two points for a line
                # Draw line into mask
                for i in range(len(shape) - 1):
                    x1, y1 = shape[i]
                    x2, y2 = shape[i + 1]

                    # Use Bresenham's line algorithm via scipy.ndimage.map_coordinates
                    # Create coordinates for the line
                    length = int(np.hypot(x2 - x1, y2 - y1))
                    if length == 0:
                        continue

                    t = np.linspace(0, 1, length * 2)
                    x = np.round(x1 * (1 - t) + x2 * t).astype(int)
                    y = np.round(y1 * (1 - t) + y2 * t).astype(int)

                    # Filter points outside the image
                    valid = (0 <= x) & (x < width) & (0 <= y) & (y < height)
                    x, y = x[valid], y[valid]

                    # Add points to mask
                    if thickness == 1:
                        shapes_mask[y, x] = True
                    else:
                        # For thicker lines, use a disk structuring element
                        # Create a disk structuring element once
                        disk_radius = thickness
                        disk_size = 2 * disk_radius + 1
                        disk_struct = np.zeros((disk_size, disk_size), dtype=bool)
                        y_grid, x_grid = np.ogrid[
                            -disk_radius : disk_radius + 1,
                            -disk_radius : disk_radius + 1,
                        ]
                        mask = x_grid**2 + y_grid**2 <= disk_radius**2
                        disk_struct[mask] = True

                        # Use scipy.ndimage.binary_dilation for efficient dilation
                        # Create a temporary mask for this line segment
                        line_mask = np.zeros_like(shapes_mask)
                        line_mask[y, x] = True
                        # Dilate the line with the disk structuring element
                        dilated_line = ndimage.binary_dilation(
                            line_mask, structure=disk_struct
                        )
                        # Add to the overall shapes mask
                        shapes_mask |= dilated_line

        # Apply color to all shapes at once
        return ColorsManagement.batch_blend_colors(image_array, shapes_mask, color)

    @staticmethod
    def batch_sample_colors(image_array, coordinates):
        """
        Efficiently sample colors from multiple coordinates in an image using scipy.ndimage.

        Args:
            image_array: NumPy array of shape (height, width, 4) containing RGBA image data
            coordinates: List of (x,y) tuples or numpy array of shape (N,2) with coordinates to sample

        Returns:
            NumPy array of shape (N,4) containing the RGBA colors at each coordinate
        """
        if len(coordinates) == 0:
            return np.array([])

        height, width = image_array.shape[:2]

        # Convert coordinates to numpy array if not already
        coords = np.array(coordinates)

        # Separate x and y coordinates
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # Create a mask for valid coordinates (within image bounds)
        valid_mask = (
            (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
        )

        # Initialize result array with zeros
        result = np.zeros((len(coordinates), 4), dtype=np.uint8)

        if not np.any(valid_mask):
            return result  # No valid coordinates

        # Filter valid coordinates
        valid_x = x_coords[valid_mask].astype(int)
        valid_y = y_coords[valid_mask].astype(int)

        # Use scipy.ndimage.map_coordinates for efficient sampling
        # This is much faster than looping through coordinates
        for channel in range(4):
            # Sample this color channel for all valid coordinates at once
            channel_values = ndimage.map_coordinates(
                image_array[..., channel],
                np.vstack((valid_y, valid_x)),
                order=0,  # Use nearest-neighbor interpolation
                mode="nearest",
            )

            # Assign sampled values to result array
            result[valid_mask, channel] = channel_values

        return result

    def cached_blend_colors(self, background: Color, foreground: Color) -> Color:
        """
        Cached version of blend_colors that stores frequently used combinations.
        This improves performance when the same color combinations are used repeatedly.

        Args:
            background: Background RGBA color tuple
            foreground: Foreground RGBA color tuple

        Returns:
            Blended RGBA color tuple
        """
        # Fast paths for common cases
        if foreground[3] == 255:
            return foreground
        if foreground[3] == 0:
            return background

        # Create a cache key from the color tuples
        cache_key = (background, foreground)

        # Check if this combination is in the cache
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]

        # Calculate the blended color
        result = ColorsManagement.blend_colors(background, foreground)

        # Store in cache (with a maximum cache size to prevent memory issues)
        if len(self.color_cache) < 1000:  # Limit cache size
            self.color_cache[cache_key] = result

        return result

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
