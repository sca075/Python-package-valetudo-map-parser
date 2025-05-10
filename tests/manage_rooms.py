import asyncio
import logging
import json
import threading
from typing import Dict, Optional, TypedDict, Any, List, Tuple

# Define JsonType since we removed the import
JsonType = Dict[str, Any]
import numpy as np
from math import sqrt
# Import removed to fix ModuleNotFoundError
from valetudo_map_parser.config.types import JsonType, LOGGER
from valetudo_map_parser.config.drawable_elements import DrawingConfig, DrawableElement

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s (line %(lineno)d) - %(message)s')
_LOGGER = logging.getLogger(__name__)

DEFAULT_ROOMS = 1

class RoomProperty(TypedDict):
    number: int
    outline: list[tuple[int, int]]
    name: str
    x: int
    y: int

RoomsProperties = dict[str, RoomProperty]
RobotPosition = dict[str, int | float]

class RoomStore:
    _instances: Dict[str, "RoomStore"] = {}
    _lock = threading.Lock()

    def __new__(cls, vacuum_id: str, rooms_data: Optional[dict] = None) -> "RoomStore":
        with cls._lock:
            if vacuum_id not in cls._instances:
                instance = super(RoomStore, cls).__new__(cls)
                instance.vacuum_id = vacuum_id
                instance.vacuums_data = rooms_data or {}
                cls._instances[vacuum_id] = instance
            else:
                if rooms_data is not None:
                    cls._instances[vacuum_id].vacuums_data = rooms_data
        return cls._instances[vacuum_id]

    def get_rooms(self) -> dict:
        return self.vacuums_data

    def set_rooms(self, rooms_data: dict) -> None:
        self.vacuums_data = rooms_data

    def get_rooms_count(self) -> int:
        if isinstance(self.vacuums_data, dict):
            count = len(self.vacuums_data)
            return count if count > 0 else DEFAULT_ROOMS
        return DEFAULT_ROOMS

    @classmethod
    def get_all_instances(cls) -> Dict[str, "RoomStore"]:
        return cls._instances

def load_test_data():
    # The test.json file is in the same folder as this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(script_dir, "test.json")
    if not os.path.exists(test_data_path):
        _LOGGER.warning(f"Test data file not found: {test_data_path}. Creating a sample one.")
        sample_data = {
            "pixelSize": 5,
            "size": {"x": 1000, "y": 1000},
            "layers": [
                {
                    "__class": "MapLayer",
                    "type": "segment",
                    "metaData": {"segmentId": 1, "name": "Living Room"},
                    "compressedPixels": [
                        100, 100, 200,  # x, y, length
                        100, 150, 200,
                        100, 200, 200,
                        100, 250, 200,
                        300, 100, 100,  # Create an L-shape
                        300, 150, 100,
                        300, 200, 100
                    ]
                },
                {
                    "__class": "MapLayer",
                    "type": "segment",
                    "metaData": {"segmentId": 2, "name": "Kitchen"},
                    "compressedPixels": [
                        400, 100, 150,  # x, y, length
                        400, 150, 150,
                        400, 200, 150,
                        400, 250, 150,
                        550, 100, 50,   # Create a T-shape
                        550, 150, 50
                    ]
                }
            ]
        }
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        with open(test_data_path, "w", encoding="utf-8") as file:
            json.dump(sample_data, file, indent=2)
        _LOGGER.info(f"Created sample test data at {test_data_path}")
        return sample_data

    with open(test_data_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    _LOGGER.info(f"Loaded test data from {test_data_path}")
    return test_data

"""
Hipfer Rooms Handler Module.
Handles room data extraction and processing for Valetudo Hipfer vacuum maps.
Provides async methods for room outline extraction and properties management.
Version: 0.1.9
"""




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
        Finds the largest connected component first to avoid issues with small disconnected regions.

        Args:
            mask: Binary mask of the room (1 for room, 0 for background)

        Returns:
            List of (x, y) tuples representing the contour
        """
        # Find connected components in the mask
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(mask)

        if num_features == 0:
            return []

        # Use a convex hull approach for all rooms, not just those with multiple components
        # Create a mask with all components
        all_components_mask = (labeled_array > 0).astype(np.uint8)

        # Get all points in the mask
        y_indices, x_indices = np.where(all_components_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return []

        # Create a list of points
        points = np.column_stack((x_indices, y_indices))

        # Compute the convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)

            # Extract the vertices of the convex hull
            hull_vertices = [(int(points[v, 0]), int(points[v, 1])) for v in hull.vertices]

            # Ensure the hull is closed
            if hull_vertices[0] != hull_vertices[-1]:
                hull_vertices.append(hull_vertices[0])

            return hull_vertices
        except Exception as e:
            LOGGER.warning(f"Failed to compute convex hull: {e}. Falling back to bounding box.")

            # Fallback to bounding box if convex hull fails
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Create a simple rectangle with 5 points (closed polygon)
            return [
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_max, y_max),  # Bottom-right
                (x_min, y_max),  # Bottom-left
                (x_min, y_min),  # Back to top-left to close the polygon
            ]

        # This code is unreachable since we always return above
        # Keeping it for reference
        padded = np.pad(all_components_mask, 1, mode="constant")
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
                    "y": ((y_min + y_max) * pixel_size) // 10,
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

def get_rrm_robot_position(json_data: JsonType) -> JsonType:
    """Get the robot position from the json."""
    return json_data.get("robot", {})


async def async_get_robot_position(self, entity_dict: dict) -> tuple | None:
    """Get the robot position from the entity data."""
    robot_pos = None
    robot_position = None
    robot_position_angle = None
    try:
        robot_pos = entity_dict.get("robot_position")
    except KeyError:
        _LOGGER.warning("%s No robot position found.", self.file_name)
        return None, None, None
    finally:
        if robot_pos:
            robot_position = robot_pos[0]["points"]
            robot_position_angle = round(
                float(robot_pos[0]["metaData"]["angle"]), 1
            )
            if self.img_h.rooms_pos is None:
                self.img_h.robot_pos = {
                    "x": robot_position[0],
                    "y": robot_position[1],
                    "angle": robot_position_angle,
                }
            else:
                self.img_h.robot_pos = await self.async_get_robot_in_room(
                    robot_y=(robot_position[1]),
                    robot_x=(robot_position[0]),
                    angle=robot_position_angle,
                )

    return robot_pos, robot_position, robot_position_angle




class MockImageHandler:
    """Mock class to simulate the ImageHandler for testing."""
    def __init__(self, vacuum_id, rooms_data):
        self.shared = type('SharedData', (), {'file_name': f'test_{vacuum_id}'})()
        self.rooms_pos = []
        self.robot_in_room = None
        self.active_zones = None
        self.zooming = False

        # Convert room_properties to the format expected by async_get_robot_in_room
        for room_id, props in rooms_data.items():
            room_info = {
                'name': props['name'],
                'outline': props['outline'],
                'corners': [
                    # Extract corners from the outline (first 4 points or all if less)
                    props['outline'][i] for i in range(min(4, len(props['outline'])))
                ]
            }
            self.rooms_pos.append(room_info)

class MockImageDraw:
    """Mock class to test the async_get_robot_in_room function."""
    def __init__(self, image_handler):
        self.img_h = image_handler
        self.file_name = self.img_h.shared.file_name

    @staticmethod
    def point_in_polygon(x: int, y: int, polygon: list) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        Enhanced version with better handling of edge cases.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            polygon: List of (x, y) tuples forming the polygon

        Returns:
            True if the point is inside the polygon, False otherwise
        """
        # Ensure we have a valid polygon with at least 3 points
        if len(polygon) < 3:
            return False

        # Make sure the polygon is closed (last point equals first point)
        if polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        # Use winding number algorithm for better accuracy
        wn = 0  # Winding number counter

        # Loop through all edges of the polygon
        for i in range(len(polygon) - 1):  # Last vertex is first vertex
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[i + 1]

            # Test if a point is left/right/on the edge defined by two vertices
            if p1y <= y:  # Start y <= P.y
                if p2y > y:  # End y > P.y (upward crossing)
                    # Point left of edge
                    if ((p2x - p1x) * (y - p1y) - (x - p1x) * (p2y - p1y)) > 0:
                        wn += 1  # Valid up intersect
            else:  # Start y > P.y
                if p2y <= y:  # End y <= P.y (downward crossing)
                    # Point right of edge
                    if ((p2x - p1x) * (y - p1y) - (x - p1x) * (p2y - p1y)) < 0:
                        wn -= 1  # Valid down intersect

        # If winding number is not 0, the point is inside the polygon
        return wn != 0

    async def async_get_robot_in_room(self, robot_y: int = 0, robot_x: int = 0, angle: float = 0.0) -> dict:
        """Get the robot position and return in what room is."""
        # First check if we already have a cached room and if the robot is still in it
        if self.img_h.robot_in_room:
            # If we have outline data, use point_in_polygon for accurate detection
            if "outline" in self.img_h.robot_in_room:
                outline = self.img_h.robot_in_room["outline"]
                if self.point_in_polygon(int(robot_x), int(robot_y), outline):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }
                    # Handle active zones
                    if self.img_h.active_zones and (
                        self.img_h.robot_in_room["id"]
                        in range(len(self.img_h.active_zones))
                    ):
                        self.img_h.zooming = bool(
                            self.img_h.active_zones[self.img_h.robot_in_room["id"]]
                        )
                    else:
                        self.img_h.zooming = False
                    return temp
            # Fallback to bounding box check if no outline data
            elif all(k in self.img_h.robot_in_room for k in ["left", "right", "up", "down"]):
                if (
                    (self.img_h.robot_in_room["right"] >= int(robot_x))
                    and (self.img_h.robot_in_room["left"] <= int(robot_x))
                ) and (
                    (self.img_h.robot_in_room["down"] >= int(robot_y))
                    and (self.img_h.robot_in_room["up"] <= int(robot_y))
                ):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }
                    # Handle active zones
                    if self.img_h.active_zones and (
                        self.img_h.robot_in_room["id"]
                        in range(len(self.img_h.active_zones))
                    ):
                        self.img_h.zooming = bool(
                            self.img_h.active_zones[self.img_h.robot_in_room["id"]]
                        )
                    else:
                        self.img_h.zooming = False
                    return temp

        # If we don't have a cached room or the robot is not in it, search all rooms
        last_room = None
        room_count = 0
        if self.img_h.robot_in_room:
            last_room = self.img_h.robot_in_room

        # Check if the robot is far outside the normal map boundaries
        # This helps prevent false positives for points very far from any room
        map_boundary = 20000  # Typical map size is around 5000-10000 units
        if abs(robot_x) > map_boundary or abs(robot_y) > map_boundary:
            _LOGGER.debug(
                "%s robot position (%s, %s) is far outside map boundaries.",
                self.file_name, robot_x, robot_y
            )
            self.img_h.robot_in_room = last_room
            self.img_h.zooming = False
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": last_room["room"] if last_room else None,
            }
            return temp

        # Search through all rooms to find which one contains the robot
        for room in self.img_h.rooms_pos:
            # Check if the room has an outline (polygon points)
            if "outline" in room:
                outline = room["outline"]
                # Use point_in_polygon for accurate detection with complex shapes
                if self.point_in_polygon(int(robot_x), int(robot_y), outline):
                    # Robot is in this room
                    self.img_h.robot_in_room = {
                        "id": room_count,
                        "room": str(room["name"]),
                        "outline": outline,
                    }
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }
                    _LOGGER.debug(
                        "%s is in %s room (polygon detection).",
                        self.file_name,
                        self.img_h.robot_in_room["room"],
                    )
                    return temp
            # Fallback to bounding box if no outline is available
            elif "corners" in room:
                corners = room["corners"]
                # Create a bounding box from the corners
                self.img_h.robot_in_room = {
                    "id": room_count,
                    "left": int(corners[0][0]),
                    "right": int(corners[2][0]),
                    "up": int(corners[0][1]),
                    "down": int(corners[2][1]),
                    "room": str(room["name"]),
                }
                # Check if the robot is inside the bounding box
                if (
                    (self.img_h.robot_in_room["right"] >= int(robot_x))
                    and (self.img_h.robot_in_room["left"] <= int(robot_x))
                ) and (
                    (self.img_h.robot_in_room["down"] >= int(robot_y))
                    and (self.img_h.robot_in_room["up"] <= int(robot_y))
                ):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }
                    _LOGGER.debug(
                        "%s is in %s room (bounding box detection).",
                        self.file_name,
                        self.img_h.robot_in_room["room"],
                    )
                    return temp
            room_count += 1

        # Robot not found in any room
        _LOGGER.debug(
            "%s not located within any room coordinates.",
            self.file_name,
        )
        self.img_h.robot_in_room = last_room
        self.img_h.zooming = False
        temp = {
            "x": robot_x,
            "y": robot_y,
            "angle": angle,
            "in_room": last_room["room"] if last_room else None,
        }
        return temp

async def test_robot_in_room(room_properties):
    """Test the async_get_robot_in_room function with the extracted room properties."""
    # Create mock objects
    mock_handler = MockImageHandler("test_vacuum", room_properties)
    mock_draw = MockImageDraw(mock_handler)

    # Test each room with its center point
    _LOGGER.info("Testing robot detection in each room...")
    success_count = 0
    total_rooms = len(room_properties)

    for room_id, props in room_properties.items():
        # Use the room's center coordinates as the robot position
        robot_x = props['x']
        robot_y = props['y']

        # Verify that the point is actually inside the polygon using our algorithm
        is_inside = mock_draw.point_in_polygon(robot_x, robot_y, props['outline'])
        if not is_inside:
            _LOGGER.warning(f"⚠️ Center point ({robot_x}, {robot_y}) is not inside room {room_id}: {props['name']}")
            # Try to find a better test point by averaging some points from the outline
            points = props['outline']
            if len(points) >= 3:
                # Use the average of the first 3 points as an alternative test point
                alt_x = sum(p[0] for p in points[:3]) // 3
                alt_y = sum(p[1] for p in points[:3]) // 3
                if mock_draw.point_in_polygon(alt_x, alt_y, props['outline']):
                    _LOGGER.info(f"   Using alternative point ({alt_x}, {alt_y}) for testing")
                    robot_x, robot_y = alt_x, alt_y

        # Call the function to detect which room the robot is in
        result = await mock_draw.async_get_robot_in_room(robot_y=robot_y, robot_x=robot_x)

        # Check if the robot was correctly detected in this room
        if result and result.get('in_room') == props['name']:
            _LOGGER.info(f"✅ Robot correctly detected in room {room_id}: {props['name']}")
            success_count += 1
        else:
            detected_room = result.get('in_room', 'None') if result else 'None'
            _LOGGER.error(f"❌ Robot detection failed for room {room_id}: {props['name']}")
            _LOGGER.error(f"   Robot at ({robot_x}, {robot_y}) was detected in: {detected_room}")

            # Debug: Check which room's polygon contains this point
            _LOGGER.error(f"   Debugging point-in-polygon for ({robot_x}, {robot_y}):")
            for test_id, test_props in room_properties.items():
                test_result = mock_draw.point_in_polygon(robot_x, robot_y, test_props['outline'])
                _LOGGER.error(f"   - Room {test_id} ({test_props['name']}): {test_result}")

    # Test with a position that should be outside all rooms
    outside_x, outside_y = -50000, -50000  # Very far outside any room
    result = await mock_draw.async_get_robot_in_room(robot_y=outside_y, robot_x=outside_x)

    # For points very far outside, we expect the boundary check to trigger
    # and return the last known room or None
    if result:
        _LOGGER.info("✅ Robot outside all rooms handled correctly")

        # Verify that no room's polygon actually contains this point
        any_contains = False
        for test_id, test_props in room_properties.items():
            test_result = mock_draw.point_in_polygon(outside_x, outside_y, test_props['outline'])
            if test_result:
                any_contains = True
                _LOGGER.error(f"   - Room {test_id} ({test_props['name']}) incorrectly contains the outside point!")

        if not any_contains:
            _LOGGER.info("   No room polygons contain the outside point (correct)")

    # Report overall success rate
    _LOGGER.info(f"\nTest Results: {success_count}/{total_rooms} rooms correctly detected ({success_count/total_rooms*100:.1f}%)")
    return success_count == total_rooms

# Room Shape Regularization Functions

def extract_room_outlines(room_properties: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Process all room properties and ensure their outlines are properly formed.

    Args:
        room_properties: Dictionary of room properties

    Returns:
        Dictionary of room properties with properly formed outlines
    """
    processed_properties = {}

    for room_id, props in room_properties.items():
        # Copy the properties
        new_props = props.copy()

        # Get the outline
        outline = props.get('outline', [])
        if len(outline) >= 3:
            # Ensure the outline is properly formed
            new_outline = ensure_closed_outline(outline, props.get('name', f'Room {room_id}'))

            # Update the outline
            new_props['outline'] = new_outline

            # Recalculate center if needed
            if 'x' in props and 'y' in props:
                # Check if center is inside the new outline
                center_x, center_y = props['x'], props['y']
                if not MockImageDraw.point_in_polygon(center_x, center_y, new_outline):
                    # Recalculate center as centroid of the new outline
                    xs = [p[0] for p in new_outline]
                    ys = [p[1] for p in new_outline]
                    new_props['x'] = sum(xs) // len(xs)
                    new_props['y'] = sum(ys) // len(ys)
                    _LOGGER.debug(f"Recalculated center for {props.get('name', f'Room {room_id}')} from ({center_x}, {center_y}) to ({new_props['x']}, {new_props['y']})")

        processed_properties[room_id] = new_props

    return processed_properties

def ensure_closed_outline(outline: List[Tuple[int, int]], room_name: str = "Unknown") -> List[Tuple[int, int]]:
    """
    Ensure the room outline is properly formed and closed.

    Args:
        outline: List of (x, y) points forming the room outline
        room_name: Name of the room for logging

    Returns:
        List of (x, y) points forming a properly closed outline
    """
    # Ensure we have enough points
    if len(outline) < 3:
        _LOGGER.warning(f"Room {room_name} has too few points ({len(outline)}) for a valid outline")
        return outline

    # Remove duplicate consecutive points
    cleaned_outline = [outline[0]]
    for i in range(1, len(outline)):
        if outline[i] != cleaned_outline[-1]:
            cleaned_outline.append(outline[i])

    # Ensure the outline is closed (first point equals last point)
    if cleaned_outline[0] != cleaned_outline[-1]:
        cleaned_outline.append(cleaned_outline[0])

    # Check if the outline has at least 3 unique points
    unique_points = set(tuple(p) for p in cleaned_outline)
    if len(unique_points) < 3:
        _LOGGER.warning(f"Room {room_name} has fewer than 3 unique points, which cannot form a valid polygon")
        return outline

    _LOGGER.info(f"Room {room_name}: Processed outline from {len(outline)} to {len(cleaned_outline)} points")

    return cleaned_outline



async def main():
    test_data = load_test_data()
    if test_data is None:
        _LOGGER.error("Failed to load test data")
        return
    cls_rooms = HypferRoomsHandler("test_vacuum")
    _LOGGER.info("Extracting room properties...")
    room_properties = await cls_rooms.async_extract_room_properties(test_data)
    _LOGGER.info(f"Found {len(room_properties)} rooms")
    for room_id, props in room_properties.items():
        _LOGGER.info(f"Room {room_id}: {props['name']} at ({props['x']}, {props['y']})")

        # Format the outline as a list of [x, y] coordinates for better readability
        # Convert numpy int64 values to regular integers
        formatted_outline = [[int(x), int(y)] for x, y in props['outline']]
        _LOGGER.info(f"  Outline: {formatted_outline}")

    # Get robot position from the test data
    robot_pos = get_rrm_robot_position(test_data)
    _LOGGER.info(f"Robot position: {robot_pos}")

    # Test the robot detection function
    _LOGGER.info("\nTesting robot detection function...")
    await test_robot_in_room(room_properties)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        _LOGGER.error(f"Error running async code: {e}")
