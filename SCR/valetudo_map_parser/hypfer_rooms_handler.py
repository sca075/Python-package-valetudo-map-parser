"""
Hipfer Rooms Handler Module.
Handles room data extraction and processing for Valetudo Hipfer vacuum maps.
Provides async methods for room outline extraction and properties management.
Version: 0.1.9
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

from .config.drawable_elements import DrawableElement, DrawingConfig
from .config.types import LOGGER, RoomsProperties, RoomStore


class HypferRoomsHandler:
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

    @staticmethod
    def sublist(data: list, chunk_size: int) -> list:
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    @staticmethod
    def perpendicular_distance(
        point: tuple[int, int], line_start: tuple[int, int], line_end: tuple[int, int]
    ) -> float:
        """Calculate the perpendicular distance from a point to a line."""
        if line_start == line_end:
            return sqrt(
                (point[0] - line_start[0]) ** 2 + (point[1] - line_start[1]) ** 2
            )

        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate the line length
        line_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_length == 0:
            return 0

        # Calculate the distance from the point to the line
        return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_length

    async def rdp(
        self, points: List[Tuple[int, int]], epsilon: float
    ) -> List[Tuple[int, int]]:
        """Ramer-Douglas-Peucker algorithm for simplifying a curve."""
        if len(points) <= 2:
            return points

        # Find the point with the maximum distance
        dmax = 0
        index = 0
        for i in range(1, len(points) - 1):
            d = self.perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            first_segment = await self.rdp(points[: index + 1], epsilon)
            second_segment = await self.rdp(points[index:], epsilon)

            # Build the result list (avoiding duplicating the common point)
            return first_segment[:-1] + second_segment
        else:
            return [points[0], points[-1]]

    async def async_get_corners(
        self, mask: np.ndarray, epsilon_factor: float = 0.05
    ) -> List[Tuple[int, int]]:
        """
        Get the corners of a room shape as a list of (x, y) tuples.
        Uses contour detection and Douglas-Peucker algorithm to simplify the contour.

        Args:
            mask: Binary mask of the room (1 for room, 0 for background)
            epsilon_factor: Controls the level of simplification (higher = fewer points)

        Returns:
            List of (x, y) tuples representing the corners of the room
        """
        # Find contours in the mask
        contour = await self.async_moore_neighbor_trace(mask)

        if not contour:
            # Fallback to bounding box if contour detection fails
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                return []

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            return [
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_max, y_max),  # Bottom-right
                (x_min, y_max),  # Bottom-left
                (x_min, y_min),  # Back to top-left to close the polygon
            ]

        # Calculate the perimeter of the contour
        perimeter = 0
        for i in range(len(contour) - 1):
            x1, y1 = contour[i]
            x2, y2 = contour[i + 1]
            perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Apply Douglas-Peucker algorithm to simplify the contour
        epsilon = epsilon_factor * perimeter
        simplified_contour = await self.rdp(contour, epsilon=epsilon)

        # Ensure the contour has at least 3 points to form a polygon
        if len(simplified_contour) < 3:
            # Fallback to bounding box
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

            LOGGER.debug(
                f"{self.vacuum_id}: Too few points in contour, using bounding box"
            )
            return [
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_max, y_max),  # Bottom-right
                (x_min, y_max),  # Bottom-left
                (x_min, y_min),  # Back to top-left to close the polygon
            ]

        # Ensure the contour is closed
        if simplified_contour[0] != simplified_contour[-1]:
            simplified_contour.append(simplified_contour[0])

        return simplified_contour

    @staticmethod
    async def async_moore_neighbor_trace(mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Trace the contour of a binary mask using an optimized Moore-Neighbor tracing.

        Args:
            mask: Binary mask of the room (1 for room, 0 for background)

        Returns:
            List of (x, y) tuples representing the contour
        """
        # Convert to uint8 and pad
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        height, width = padded.shape

        # Find the first non-zero point efficiently (scan row by row)
        # This is much faster than np.argwhere() for large arrays
        start = None
        for y in range(height):
            # Use NumPy's any() to quickly check if there are any 1s in this row
            if np.any(padded[y]):
                # Find the first 1 in this row
                x = np.where(padded[y] == 1)[0][0]
                start = (int(x), int(y))
                break

        if start is None:
            return []

        # Pre-compute directions
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]

        # Use a 2D array for visited tracking (faster than set)
        visited = np.zeros((height, width), dtype=bool)

        # Initialize contour
        contour = []
        contour.append((int(start[0] - 1), int(start[1] - 1)))  # Adjust for padding

        current = start
        prev_dir = 7
        visited[current[1], current[0]] = True

        # Main tracing loop
        while True:
            found = False

            # Check all 8 directions
            for i in range(8):
                dir_idx = (prev_dir + i) % 8
                dx, dy = directions[dir_idx]
                nx, ny = current[0] + dx, current[1] + dy

                # Bounds check and value check
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and padded[ny, nx] == 1
                    and not visited[ny, nx]
                ):
                    current = (nx, ny)
                    visited[ny, nx] = True
                    contour.append(
                        (int(nx - 1), int(ny - 1))
                    )  # Adjust for padding and convert to int
                    prev_dir = (dir_idx + 5) % 8
                    found = True
                    break

            # Check termination conditions
            if not found or (
                len(contour) > 3
                and (int(current[0] - 1), int(current[1] - 1)) == contour[0]
            ):
                break

        return contour

    async def async_extract_room_properties(
        self, json_data: Dict[str, Any]
    ) -> RoomsProperties:
        """
        Extract room properties from the JSON data.

        Args:
            json_data: JSON data from the vacuum

        Returns:
            Dictionary of room properties
        """
        room_properties = {}
        pixel_size = json_data.get("pixelSize", 5)
        height = json_data["size"]["y"]
        width = json_data["size"]["x"]
        vacuum_id = self.vacuum_id
        room_id_counter = 0

        for layer in json_data.get("layers", []):
            if layer.get("__class") == "MapLayer" and layer.get("type") == "segment":
                meta_data = layer.get("metaData", {})
                segment_id = meta_data.get("segmentId")
                name = meta_data.get("name", f"Room {segment_id}")

                # Check if this room is disabled in the drawing configuration
                # The room_id_counter is 0-based, but DrawableElement.ROOM_X is 1-based
                current_room_id = room_id_counter + 1
                room_id_counter = (
                    room_id_counter + 1
                ) % 16  # Cycle room_id back to 0 after 15

                if 1 <= current_room_id <= 15 and self.drawing_config is not None:
                    room_element = getattr(
                        DrawableElement, f"ROOM_{current_room_id}", None
                    )
                    if room_element and not self.drawing_config.is_enabled(
                        room_element
                    ):
                        LOGGER.debug(
                            "%s: Room %d is disabled and will be skipped",
                            self.vacuum_id,
                            current_room_id,
                        )
                        continue

                compressed_pixels = layer.get("compressedPixels", [])
                pixels = self.sublist(compressed_pixels, 3)

                # Create a binary mask for the room
                if not pixels:
                    LOGGER.warning(f"Skipping segment {segment_id}: no pixels found")
                    continue

                mask = np.zeros((height, width), dtype=np.uint8)
                for x, y, length in pixels:
                    if 0 <= y < height and 0 <= x < width and x + length <= width:
                        mask[y, x : x + length] = 1

                # Find the room outline using the improved get_corners function
                # Adjust epsilon_factor to control the level of simplification (higher = fewer points)
                outline = await self.async_get_corners(mask, epsilon_factor=0.05)

                if not outline:
                    LOGGER.warning(
                        f"Skipping segment {segment_id}: failed to generate outline"
                    )
                    continue

                # Calculate the center of the room
                xs, ys = zip(*outline)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Scale coordinates by pixel_size
                scaled_outline = [(x * pixel_size, y * pixel_size) for x, y in outline]

                room_id = str(segment_id)
                room_properties[room_id] = {
                    "number": segment_id,
                    "outline": scaled_outline,  # Already includes the closing point
                    "name": name,
                    "x": ((x_min + x_max) * pixel_size) // 2,
                    "y": ((y_min + y_max) * pixel_size) // 2,
                }

        RoomStore(vacuum_id, room_properties)
        return room_properties

    async def get_room_at_position(
        self, x: int, y: int, room_properties: Optional[RoomsProperties] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the room at a specific position.

        Args:
            x: X coordinate
            y: Y coordinate
            room_properties: Room properties dictionary (optional)

        Returns:
            Room data dictionary or None if no room at position
        """
        if room_properties is None:
            room_store = RoomStore(self.vacuum_id)
            room_properties = room_store.get_rooms()

        if not room_properties:
            return None

        for room_id, room_data in room_properties.items():
            outline = room_data.get("outline", [])
            if not outline or len(outline) < 3:
                continue

            # Check if point is inside the polygon
            if self.point_in_polygon(x, y, outline):
                return {
                    "id": room_id,
                    "name": room_data.get("name", f"Room {room_id}"),
                    "x": room_data.get("x", 0),
                    "y": room_data.get("y", 0),
                }

        return None

    @staticmethod
    def point_in_polygon(x: int, y: int, polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            polygon: List of (x, y) tuples forming the polygon

        Returns:
            True if the point is inside the polygon, False otherwise
        """
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        xinters = None  # Initialize with default value
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
