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
        self.current_json_data = None  # Will store the current JSON data being processed

    @staticmethod
    def sublist(data: list, chunk_size: int) -> list:
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Cache for RDP results
    _rdp_cache = {}

    @staticmethod
    def perpendicular_distance(
        point: tuple[int, int], line_start: tuple[int, int], line_end: tuple[int, int]
    ) -> float:
        """Calculate the perpendicular distance from a point to a line.
        Optimized for performance.
        """
        # Fast path for point-to-point distance
        if line_start == line_end:
            dx = point[0] - line_start[0]
            dy = point[1] - line_start[1]
            return sqrt(dx*dx + dy*dy)

        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Precompute differences for efficiency
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the line length squared (avoid sqrt until needed)
        line_length_sq = dx*dx + dy*dy
        if line_length_sq == 0:
            return 0

        # Calculate the distance from the point to the line
        # Using the formula: |cross_product| / |line_vector|
        # This is more efficient than the original formula
        cross_product = abs(dy * x - dx * y + x2 * y1 - y2 * x1)
        return cross_product / sqrt(line_length_sq)

    async def rdp(
        self, points: List[Tuple[int, int]], epsilon: float
    ) -> List[Tuple[int, int]]:
        """Ramer-Douglas-Peucker algorithm for simplifying a curve.
        Optimized with caching and better performance.
        """
        # Create a hashable key for caching
        # Convert points to a tuple for hashing
        points_tuple = tuple(points)
        cache_key = (points_tuple, epsilon)

        # Check cache first
        if cache_key in self._rdp_cache:
            return self._rdp_cache[cache_key]

        # Base case
        if len(points) <= 2:
            return points

        # For very small point sets, process directly without recursion
        if len(points) <= 5:
            # Find the point with the maximum distance
            dmax = 0
            index = 0
            for i in range(1, len(points) - 1):
                d = self.perpendicular_distance(points[i], points[0], points[-1])
                if d > dmax:
                    index = i
                    dmax = d

            # If max distance is greater than epsilon, keep the point
            if dmax > epsilon:
                result = [points[0]] + [points[index]] + [points[-1]]
            else:
                result = [points[0], points[-1]]

            # Cache and return
            self._rdp_cache[cache_key] = result
            return result

        # For larger point sets, use numpy for faster distance calculation
        if len(points) > 20:
            # Convert to numpy arrays for vectorized operations
            points_array = np.array(points)
            start = points_array[0]
            end = points_array[-1]

            # Calculate perpendicular distances in one vectorized operation
            line_vector = end - start
            line_length = np.linalg.norm(line_vector)

            if line_length == 0:
                # If start and end are the same, use direct distance
                distances = np.linalg.norm(points_array[1:-1] - start, axis=1)
            else:
                # Normalize line vector
                line_vector = line_vector / line_length
                # Calculate perpendicular distances using vector operations
                vectors_to_points = points_array[1:-1] - start
                # Project vectors onto line vector
                projections = np.dot(vectors_to_points, line_vector)
                # Calculate projected points on line
                projected_points = start + np.outer(projections, line_vector)
                # Calculate distances from points to their projections
                distances = np.linalg.norm(points_array[1:-1] - projected_points, axis=1)

            # Find the point with maximum distance
            if len(distances) > 0:
                max_idx = np.argmax(distances)
                dmax = distances[max_idx]
                index = max_idx + 1  # +1 because we skipped the first point
            else:
                dmax = 0
                index = 0
        else:
            # For medium-sized point sets, use the original algorithm
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
            result = first_segment[:-1] + second_segment
        else:
            result = [points[0], points[-1]]

        # Limit cache size
        if len(self._rdp_cache) > 100:  # Keep only 100 most recent items
            try:
                self._rdp_cache.pop(next(iter(self._rdp_cache)))
            except (StopIteration, KeyError):
                pass

        # Cache the result
        self._rdp_cache[cache_key] = result
        return result

    # Cache for corner results
    _corners_cache = {}

    async def async_get_corners(
        self, mask: np.ndarray, epsilon_factor: float = 0.05
    ) -> List[Tuple[int, int]]:
        """
        Get the corners of a room shape as a list of (x, y) tuples.
        Uses contour detection and Douglas-Peucker algorithm to simplify the contour.
        Optimized with caching and faster calculations.

        Args:
            mask: Binary mask of the room (1 for room, 0 for background)
            epsilon_factor: Controls the level of simplification (higher = fewer points)

        Returns:
            List of (x, y) tuples representing the corners of the room
        """
        # Create a hash of the mask and epsilon factor for caching
        mask_hash = hash((mask.tobytes(), epsilon_factor))

        # Check if we have a cached result
        if mask_hash in self._corners_cache:
            return self._corners_cache[mask_hash]

        # Fast path for empty masks
        if not np.any(mask):
            return []

        # Find contours in the mask - this uses our optimized method with caching
        contour = await self.async_moore_neighbor_trace(mask)

        if not contour:
            # Fallback to bounding box if contour detection fails
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                return []

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            result = [
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_max, y_max),  # Bottom-right
                (x_min, y_max),  # Bottom-left
                (x_min, y_min),  # Back to top-left to close the polygon
            ]

            # Cache the result
            self._corners_cache[mask_hash] = result
            return result

        # For small contours (less than 10 points), skip simplification
        if len(contour) <= 10:
            # Ensure the contour is closed
            if contour[0] != contour[-1]:
                contour.append(contour[0])

            # Cache and return
            self._corners_cache[mask_hash] = contour
            return contour

        # For larger contours, calculate perimeter more efficiently using numpy
        points = np.array(contour)
        # Calculate differences between consecutive points
        diffs = np.diff(points, axis=0)
        # Calculate squared distances
        squared_dists = np.sum(diffs**2, axis=1)
        # Calculate perimeter as sum of distances
        perimeter = np.sum(np.sqrt(squared_dists))

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
            result = [
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_max, y_max),  # Bottom-right
                (x_min, y_max),  # Bottom-left
                (x_min, y_min),  # Back to top-left to close the polygon
            ]

            # Cache the result
            self._corners_cache[mask_hash] = result
            return result

        # Ensure the contour is closed
        if simplified_contour[0] != simplified_contour[-1]:
            simplified_contour.append(simplified_contour[0])

        # Limit cache size
        if len(self._corners_cache) > 50:  # Keep only 50 most recent items
            try:
                self._corners_cache.pop(next(iter(self._corners_cache)))
            except (StopIteration, KeyError):
                pass

        # Cache the result
        self._corners_cache[mask_hash] = simplified_contour
        return simplified_contour

    # Cache for labeled arrays to avoid redundant calculations
    _label_cache = {}
    _hull_cache = {}

    @staticmethod
    async def async_moore_neighbor_trace(mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Trace the contour of a binary mask using an optimized approach.
        Uses caching and simplified algorithms for better performance.

        Args:
            mask: Binary mask of the room (1 for room, 0 for background)

        Returns:
            List of (x, y) tuples representing the contour
        """
        # Create a hash of the mask for caching
        mask_hash = hash(mask.tobytes())

        # Check if we have a cached result
        if mask_hash in HypferRoomsHandler._hull_cache:
            return HypferRoomsHandler._hull_cache[mask_hash]

        # Fast path for empty masks
        if not np.any(mask):
            return []

        # Find bounding box of non-zero elements (much faster than full labeling for simple cases)
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return []

        # For very small rooms (less than 100 pixels), just use bounding box
        if len(x_indices) < 100:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Create a simple rectangle
            hull_vertices = [
                (int(x_min), int(y_min)),  # Top-left
                (int(x_max), int(y_min)),  # Top-right
                (int(x_max), int(y_max)),  # Bottom-right
                (int(x_min), int(y_max)),  # Bottom-left
                (int(x_min), int(y_min)),  # Back to top-left to close the polygon
            ]

            # Cache and return the result
            HypferRoomsHandler._hull_cache[mask_hash] = hull_vertices
            return hull_vertices

        # For larger rooms, use convex hull but with optimizations
        try:
            # Import here to avoid overhead for small rooms
            from scipy import ndimage
            from scipy.spatial import ConvexHull

            # Use cached labeled array if available
            if mask_hash in HypferRoomsHandler._label_cache:
                labeled_array = HypferRoomsHandler._label_cache[mask_hash]
            else:
                # Find connected components - this is expensive
                labeled_array, _ = ndimage.label(mask)
                # Cache the result for future use
                HypferRoomsHandler._label_cache[mask_hash] = labeled_array

                # Limit cache size to prevent memory issues
                if len(HypferRoomsHandler._label_cache) > 50:  # Keep only 50 most recent items
                    # Remove oldest item (first key)
                    try:
                        HypferRoomsHandler._label_cache.pop(next(iter(HypferRoomsHandler._label_cache)))
                    except (StopIteration, KeyError):
                        # Handle edge case of empty cache
                        pass

            # Create a mask with all components
            all_components_mask = (labeled_array > 0)

            # Sample points instead of using all points for large masks
            # This significantly reduces computation time for ConvexHull
            if len(x_indices) > 1000:
                # Sample every 10th point for very large rooms
                step = 10
            elif len(x_indices) > 500:
                # Sample every 5th point for medium-sized rooms
                step = 5
            else:
                # Use all points for smaller rooms
                step = 1

            # Sample points using the step size
            sampled_y = y_indices[::step]
            sampled_x = x_indices[::step]

            # Create a list of points
            points = np.column_stack((sampled_x, sampled_y))

            # Compute the convex hull
            hull = ConvexHull(points)

            # Extract the vertices of the convex hull
            hull_vertices = [(int(points[v, 0]), int(points[v, 1])) for v in hull.vertices]

            # Ensure the hull is closed
            if hull_vertices[0] != hull_vertices[-1]:
                hull_vertices.append(hull_vertices[0])

            # Cache and return the result
            HypferRoomsHandler._hull_cache[mask_hash] = hull_vertices

            # Limit hull cache size
            if len(HypferRoomsHandler._hull_cache) > 50:
                try:
                    HypferRoomsHandler._hull_cache.pop(next(iter(HypferRoomsHandler._hull_cache)))
                except (StopIteration, KeyError):
                    pass

            return hull_vertices

        except Exception as e:
            LOGGER.warning(f"Failed to compute convex hull: {e}. Falling back to bounding box.")

            # Fallback to bounding box if convex hull fails
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Create a simple rectangle
            hull_vertices = [
                (int(x_min), int(y_min)),  # Top-left
                (int(x_max), int(y_min)),  # Top-right
                (int(x_max), int(y_max)),  # Bottom-right
                (int(x_min), int(y_max)),  # Bottom-left
                (int(x_min), int(y_min)),  # Back to top-left to close the polygon
            ]

            # Cache and return the result
            HypferRoomsHandler._hull_cache[mask_hash] = hull_vertices
            return hull_vertices



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

        # Store the JSON data for reference in other methods
        self.current_json_data = json_data

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
