"""
Hipfer Rooms Handler Module.
Handles room data extraction and processing for Valetudo Hipfer vacuum maps.
Provides async methods for room outline extraction and properties management.
Version: 0.1.9
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import ConvexHull

from .config.drawable_elements import DrawableElement, DrawingConfig
from .config.types import LOGGER, RoomsProperties


class RoomsHandler:
    """
    Handler for extracting and managing room data from Hipfer vacuum maps.

    This class provides methods to:
    - Extract room outlines using the Ramer-Douglas-Peucker algorithm
    - Process room properties from JSON data
    - Generate room masks and extract contours

    All methods are async for better integration with the rest of the codebase.
    """

    def __init__(self, vacuum_id: str, drawing_config: Optional[DrawingConfig] = None):
        """
        Initialize the HipferRoomsHandler.

        Args:
            vacuum_id: Identifier for the vacuum
            drawing_config: Configuration for which elements to draw (optional)
        """
        self.vacuum_id = vacuum_id
        self.drawing_config = drawing_config
        self.current_json_data = (
            None  # Will store the current JSON data being processed
        )

    @staticmethod
    def sublist(data: list, chunk_size: int) -> list:
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    @staticmethod
    def convex_hull_outline(mask: np.ndarray) -> list[tuple[int, int]]:
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return []

        points = np.column_stack((x_indices, y_indices))
        if len(points) < 3:
            return [(int(x), int(y)) for x, y in points]

        hull = ConvexHull(points)
        # Convert numpy.int64 values to regular Python integers
        hull_points = [
            (int(points[vertex][0]), int(points[vertex][1])) for vertex in hull.vertices
        ]
        if hull_points[0] != hull_points[-1]:
            hull_points.append(hull_points[0])
        return hull_points

    async def _process_room_layer(
        self, layer: Dict[str, Any], width: int, height: int, pixel_size: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Process a single room layer and extract its outline.

        Args:
            layer: The layer data from the JSON
            width: The width of the map
            height: The height of the map
            pixel_size: The size of each pixel

        Returns:
            Tuple of (room_id, room_data) or (None, None) if processing failed
        """
        meta_data = layer.get("metaData", {})
        segment_id = meta_data.get("segmentId")
        name = meta_data.get("name", "Room {}".format(segment_id))
        compressed_pixels = layer.get("compressedPixels", [])
        pixels = self.sublist(compressed_pixels, 3)

        # Check if this room is enabled in the drawing configuration
        if self.drawing_config is not None:
            # Convert segment_id to room element (ROOM_1 to ROOM_15)
            try:
                # Segment IDs might not be sequential, so we need to map them to room elements
                # We'll use a simple approach: if segment_id is an integer, use it directly
                room_element_id = int(segment_id)
                if 1 <= room_element_id <= 15:
                    room_element = getattr(
                        DrawableElement, f"ROOM_{room_element_id}", None
                    )
                    if room_element:
                        is_enabled = self.drawing_config.is_enabled(room_element)
                        if not is_enabled:
                            # Skip this room if it's disabled
                            LOGGER.debug("Skipping disabled room %s", segment_id)
                            return None, None
            except (ValueError, TypeError):
                # If segment_id is not a valid integer, we can't map it to a room element
                # In this case, we'll include the room (fail open)
                LOGGER.debug(
                    "Could not convert segment_id %s to room element", segment_id
                )

        # Optimization: Create a smaller mask for just the room area
        if not pixels:
            # Skip if no pixels
            return None, None

        # Convert to numpy arrays for vectorized operations
        pixel_data = np.array(pixels)

        if pixel_data.size == 0:
            return None, None

        # Find the actual bounds of the room to create a smaller mask
        # Add padding to ensure we don't lose edge details
        padding = 10  # Add padding pixels around the room
        min_x = max(0, int(np.min(pixel_data[:, 0])) - padding)
        max_x = min(
            width, int(np.max(pixel_data[:, 0]) + np.max(pixel_data[:, 2])) + padding
        )
        min_y = max(0, int(np.min(pixel_data[:, 1])) - padding)
        max_y = min(height, int(np.max(pixel_data[:, 1]) + 1) + padding)

        # Create a smaller mask for just the room area (much faster)
        local_width = max_x - min_x
        local_height = max_y - min_y

        # Skip if dimensions are invalid
        if local_width <= 0 or local_height <= 0:
            return None, None

        # Create a smaller mask
        local_mask = np.zeros((local_height, local_width), dtype=np.uint8)

        # Fill the mask efficiently
        for x, y, length in pixel_data:
            x, y, length = int(x), int(y), int(length)
            # Adjust coordinates to local mask
            local_x = x - min_x
            local_y = y - min_y

            # Ensure we're within bounds
            if 0 <= local_y < local_height and 0 <= local_x < local_width:
                # Calculate the end point, clamping to mask width
                end_x = min(local_x + length, local_width)
                if end_x > local_x:  # Only process if there's a valid segment
                    local_mask[local_y, local_x:end_x] = 1

        # Apply morphological operations
        struct_elem = np.ones((3, 3), dtype=np.uint8)
        eroded = binary_erosion(local_mask, structure=struct_elem, iterations=1)
        mask = binary_dilation(eroded, structure=struct_elem, iterations=1).astype(
            np.uint8
        )

        # Extract contour from the mask
        outline = self.convex_hull_outline(mask)
        if not outline:
            return None, None

        # Adjust coordinates back to global space
        outline = [(x + min_x, y + min_y) for (x, y) in outline]

        # Use coordinates as-is without flipping Y coordinates
        xs, ys = zip(*outline)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        room_id = str(segment_id)

        # Scale coordinates by pixel_size and convert to regular Python integers
        scaled_outline = [
            (int(x * pixel_size), int(y * pixel_size)) for x, y in outline
        ]
        room_data = {
            "number": segment_id,
            "outline": scaled_outline,
            "name": name,
            "x": int(((x_min + x_max) * pixel_size) // 2),
            "y": int(((y_min + y_max) * pixel_size) // 2),
        }

        return room_id, room_data

    async def async_extract_room_properties(self, json_data) -> RoomsProperties:
        """Extract room properties from the JSON data.

        This method processes all room layers in the JSON data and extracts their outlines.
        It respects the drawing configuration, skipping rooms that are disabled.

        Args:
            json_data: The JSON data from the vacuum

        Returns:
            Dictionary of room properties
        """
        start_total = time.time()
        room_properties = {}
        pixel_size = json_data.get("pixelSize", 5)
        height = json_data["size"]["y"]
        width = json_data["size"]["x"]

        for layer in json_data.get("layers", []):
            if layer.get("__class") == "MapLayer" and layer.get("type") == "segment":
                room_id, room_data = await self._process_room_layer(
                    layer, width, height, pixel_size
                )
                if room_id is not None and room_data is not None:
                    room_properties[room_id] = room_data

        # Log timing information
        total_time = time.time() - start_total
        LOGGER.debug("Room extraction Total time: %.3fs", total_time)
        return room_properties
