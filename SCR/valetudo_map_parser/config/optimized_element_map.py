"""
Optimized Element Map Generator.
Uses scipy for efficient element map generation and processing.
Version: 0.1.9
"""

from __future__ import annotations

import logging
import numpy as np
from scipy import ndimage

from .drawable_elements import DrawableElement, DrawingConfig
from .types import LOGGER


class OptimizedElementMapGenerator:
    """Class for generating 2D element maps from JSON data with optimized performance.

    This class creates a 2D array where each cell contains an integer code
    representing the element at that position (floor, wall, room, etc.).
    It uses scipy for efficient processing and supports sparse matrices for memory efficiency.
    """

    def __init__(self, drawing_config: DrawingConfig = None, shared_data=None):
        """Initialize the optimized element map generator.

        Args:
            drawing_config: Optional drawing configuration for element properties
            shared_data: Shared data object for accessing common resources
        """
        self.drawing_config = drawing_config or DrawingConfig()
        self.shared = shared_data
        self.element_map = None
        self.element_map_shape = None
        self.scale_info = None
        self.file_name = (
            getattr(shared_data, "file_name", "ElementMap")
            if shared_data
            else "ElementMap"
        )

    async def async_generate_from_json(self, json_data, existing_element_map=None):
        """Generate a 2D element map from JSON data with optimized performance.

        Args:
            json_data: The JSON data from the vacuum
            existing_element_map: Optional pre-created element map to populate

        Returns:
            numpy.ndarray: The 2D element map
        """
        if not self.shared:
            LOGGER.warning("Shared data not provided, some features may not work.")
            return None

        # Use existing element map if provided
        if existing_element_map is not None:
            self.element_map = existing_element_map
            return existing_element_map

        # Detect JSON format
        is_valetudo = "layers" in json_data and "pixelSize" in json_data
        is_rand256 = "map_data" in json_data

        if not (is_valetudo or is_rand256):
            LOGGER.error("Unknown JSON format, cannot generate element map")
            return None

        if is_valetudo:
            return await self._generate_valetudo_element_map(json_data)
        elif is_rand256:
            return await self._generate_rand256_element_map(json_data)

    async def _generate_valetudo_element_map(self, json_data):
        """Generate an element map from Valetudo format JSON data."""
        # Get map dimensions from the JSON data
        size_x = json_data["size"]["x"]
        size_y = json_data["size"]["y"]
        pixel_size = json_data["pixelSize"]

        # Calculate downscale factor based on pixel size
        # Standard pixel size is 5mm, so adjust accordingly
        downscale_factor = max(1, pixel_size // 5 * 2)  # More aggressive downscaling

        # Calculate dimensions for the downscaled map
        map_width = max(100, size_x // (pixel_size * downscale_factor))
        map_height = max(100, size_y // (pixel_size * downscale_factor))

        LOGGER.info(
            "%s: Creating optimized element map with dimensions: %dx%d (downscale factor: %d)",
            self.file_name,
            map_width,
            map_height,
            downscale_factor,
        )

        # Create the element map at the reduced size
        element_map = np.zeros((map_height, map_width), dtype=np.int32)
        element_map[:] = DrawableElement.FLOOR

        # Store scaling information for coordinate conversion
        self.scale_info = {
            "original_size": (size_x, size_y),
            "map_size": (map_width, map_height),
            "scale_factor": downscale_factor * pixel_size,
            "pixel_size": pixel_size,
        }

        # Process layers at the reduced resolution
        for layer in json_data.get("layers", []):
            layer_type = layer.get("type")

            # Process rooms (segments)
            if layer_type == "segment":
                # Get room ID
                meta_data = layer.get("metaData", {})
                segment_id = meta_data.get("segmentId")

                if segment_id is not None:
                    # Convert segment_id to int if it's a string
                    segment_id_int = (
                        int(segment_id) if isinstance(segment_id, str) else segment_id
                    )
                    if 1 <= segment_id_int <= 15:
                        room_element = getattr(
                            DrawableElement, f"ROOM_{segment_id_int}", None
                        )

                        # Skip if room is disabled
                        if room_element is None or not self.drawing_config.is_enabled(
                            room_element
                        ):
                            continue

                        # Create a temporary high-resolution mask for this room
                        temp_mask = np.zeros(
                            (size_y // pixel_size, size_x // pixel_size), dtype=np.uint8
                        )

                        # Process pixels for this room
                        compressed_pixels = layer.get("compressedPixels", [])
                        if compressed_pixels:
                            # Process in chunks of 3 (x, y, count)
                            for i in range(0, len(compressed_pixels), 3):
                                if i + 2 < len(compressed_pixels):
                                    x = compressed_pixels[i]
                                    y = compressed_pixels[i + 1]
                                    count = compressed_pixels[i + 2]

                                    # Set pixels in the high-resolution mask
                                    for j in range(count):
                                        px = x + j
                                        if (
                                            0 <= y < temp_mask.shape[0]
                                            and 0 <= px < temp_mask.shape[1]
                                        ):
                                            temp_mask[y, px] = 1

                        # Use scipy to downsample the mask efficiently
                        # This preserves the room shape better than simple decimation
                        downsampled_mask = ndimage.zoom(
                            temp_mask,
                            (
                                map_height / temp_mask.shape[0],
                                map_width / temp_mask.shape[1],
                            ),
                            order=0,  # Nearest neighbor interpolation
                        )

                        # Apply the downsampled mask to the element map
                        element_map[downsampled_mask > 0] = room_element

                        # Clean up
                        del temp_mask, downsampled_mask

            # Process walls similarly
            elif layer_type == "wall" and self.drawing_config.is_enabled(
                DrawableElement.WALL
            ):
                # Create a temporary high-resolution mask for walls
                temp_mask = np.zeros(
                    (size_y // pixel_size, size_x // pixel_size), dtype=np.uint8
                )

                # Process compressed pixels for walls
                compressed_pixels = layer.get("compressedPixels", [])
                if compressed_pixels:
                    # Process in chunks of 3 (x, y, count)
                    for i in range(0, len(compressed_pixels), 3):
                        if i + 2 < len(compressed_pixels):
                            x = compressed_pixels[i]
                            y = compressed_pixels[i + 1]
                            count = compressed_pixels[i + 2]

                            # Set pixels in the high-resolution mask
                            for j in range(count):
                                px = x + j
                                if (
                                    0 <= y < temp_mask.shape[0]
                                    and 0 <= px < temp_mask.shape[1]
                                ):
                                    temp_mask[y, px] = 1

                # Use scipy to downsample the mask efficiently
                downsampled_mask = ndimage.zoom(
                    temp_mask,
                    (map_height / temp_mask.shape[0], map_width / temp_mask.shape[1]),
                    order=0,
                )

                # Apply the downsampled mask to the element map
                # Only overwrite floor pixels, not room pixels
                wall_mask = (downsampled_mask > 0) & (
                    element_map == DrawableElement.FLOOR
                )
                element_map[wall_mask] = DrawableElement.WALL

                # Clean up
                del temp_mask, downsampled_mask

        # Store the element map
        self.element_map = element_map
        self.element_map_shape = element_map.shape

        LOGGER.info(
            "%s: Element map generation complete with shape: %s",
            self.file_name,
            element_map.shape,
        )
        return element_map

    async def _generate_rand256_element_map(self, json_data):
        """Generate an element map from Rand256 format JSON data."""
        # Get map dimensions from the Rand256 JSON data
        map_data = json_data["map_data"]
        size_x = map_data["dimensions"]["width"]
        size_y = map_data["dimensions"]["height"]

        # Calculate downscale factor
        downscale_factor = max(
            1, min(size_x, size_y) // 500
        )  # Target ~500px in smallest dimension

        # Calculate dimensions for the downscaled map
        map_width = max(100, size_x // downscale_factor)
        map_height = max(100, size_y // downscale_factor)

        LOGGER.info(
            "%s: Creating optimized Rand256 element map with dimensions: %dx%d (downscale factor: %d)",
            self.file_name,
            map_width,
            map_height,
            downscale_factor,
        )

        # Create the element map at the reduced size
        element_map = np.zeros((map_height, map_width), dtype=np.int32)
        element_map[:] = DrawableElement.FLOOR

        # Store scaling information for coordinate conversion
        self.scale_info = {
            "original_size": (size_x, size_y),
            "map_size": (map_width, map_height),
            "scale_factor": downscale_factor,
            "pixel_size": 1,  # Rand256 uses 1:1 pixel mapping
        }

        # Process rooms
        if "rooms" in map_data and map_data["rooms"]:
            for room in map_data["rooms"]:
                # Get room ID and check if it's enabled
                room_id_int = room["id"]

                # Get room element code (ROOM_1, ROOM_2, etc.)
                room_element = None
                if 0 < room_id_int <= 15:
                    room_element = getattr(DrawableElement, f"ROOM_{room_id_int}", None)

                # Skip if room is disabled
                if room_element is None or not self.drawing_config.is_enabled(
                    room_element
                ):
                    continue

                if "coordinates" in room:
                    # Create a high-resolution mask for this room
                    temp_mask = np.zeros((size_y, size_x), dtype=np.uint8)

                    # Fill the mask with room coordinates
                    for coord in room["coordinates"]:
                        x, y = coord
                        if 0 <= y < size_y and 0 <= x < size_x:
                            temp_mask[y, x] = 1

                    # Use scipy to downsample the mask efficiently
                    downsampled_mask = ndimage.zoom(
                        temp_mask,
                        (map_height / size_y, map_width / size_x),
                        order=0,  # Nearest neighbor interpolation
                    )

                    # Apply the downsampled mask to the element map
                    element_map[downsampled_mask > 0] = room_element

                    # Clean up
                    del temp_mask, downsampled_mask

        # Process walls
        if (
            "walls" in map_data
            and map_data["walls"]
            and self.drawing_config.is_enabled(DrawableElement.WALL)
        ):
            # Create a high-resolution mask for walls
            temp_mask = np.zeros((size_y, size_x), dtype=np.uint8)

            # Fill the mask with wall coordinates
            for coord in map_data["walls"]:
                x, y = coord
                if 0 <= y < size_y and 0 <= x < size_x:
                    temp_mask[y, x] = 1

            # Use scipy to downsample the mask efficiently
            downsampled_mask = ndimage.zoom(
                temp_mask, (map_height / size_y, map_width / size_x), order=0
            )

            # Apply the downsampled mask to the element map
            # Only overwrite floor pixels, not room pixels
            wall_mask = (downsampled_mask > 0) & (element_map == DrawableElement.FLOOR)
            element_map[wall_mask] = DrawableElement.WALL

            # Clean up
            del temp_mask, downsampled_mask

        # Store the element map
        self.element_map = element_map
        self.element_map_shape = element_map.shape

        LOGGER.info(
            "%s: Rand256 element map generation complete with shape: %s",
            self.file_name,
            element_map.shape,
        )
        return element_map

    def map_to_element_coordinates(self, x, y):
        """Convert map coordinates to element map coordinates."""
        if not hasattr(self, "scale_info"):
            return x, y

        scale = self.scale_info["scale_factor"]
        return int(x / scale), int(y / scale)

    def element_to_map_coordinates(self, x, y):
        """Convert element map coordinates to map coordinates."""
        if not hasattr(self, "scale_info"):
            return x, y

        scale = self.scale_info["scale_factor"]
        return int(x * scale), int(y * scale)

    def get_element_at_position(self, x, y):
        """Get the element at the specified position."""
        if not hasattr(self, "element_map") or self.element_map is None:
            return None

        if not (
            0 <= y < self.element_map.shape[0] and 0 <= x < self.element_map.shape[1]
        ):
            return None

        return self.element_map[y, x]

    def get_room_at_position(self, x, y):
        """Get the room ID at a specific position, or None if not a room."""
        element_code = self.get_element_at_position(x, y)
        if element_code is None:
            return None

        # Check if it's a room (codes 101-115)
        if 101 <= element_code <= 115:
            return element_code
        return None

    def get_element_name(self, element_code):
        """Get the name of the element from its code."""
        if element_code is None:
            return "NONE"

        # Check if it's a room
        if element_code >= 100:
            room_number = element_code - 100
            return f"ROOM_{room_number}"

        # Check standard elements
        for name, code in vars(DrawableElement).items():
            if (
                not name.startswith("_")
                and isinstance(code, int)
                and code == element_code
            ):
                return name

        return f"UNKNOWN_{element_code}"
