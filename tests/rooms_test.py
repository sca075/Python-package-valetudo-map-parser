import asyncio
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, TypedDict

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
)
from scipy.spatial import ConvexHull


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s (line %(lineno)d) - %(message)s",
)
_LOGGER = logging.getLogger(__name__)

DEFAULT_ROOMS = 1


class RoomProperty(TypedDict):
    number: int
    outline: list[tuple[int, int]]
    name: str
    x: int
    y: int


RoomsProperties = dict[str, RoomProperty]


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
    test_data_path = "test.json"
    if not os.path.exists(test_data_path):
        _LOGGER.warning(
            "Test data file not found: %s. Creating a sample one.", test_data_path
        )
        sample_data = {
            "pixelSize": 5,
            "size": {"x": 1000, "y": 1000},
            "layers": [
                {
                    "__class": "MapLayer",
                    "type": "segment",
                    "metaData": {"segmentId": 1, "name": "Living Room"},
                    "compressedPixels": [
                        100,
                        100,
                        200,
                        100,
                        150,
                        200,
                        100,
                        200,
                        200,
                        100,
                        250,
                        200,
                    ],
                },
                {
                    "__class": "MapLayer",
                    "type": "segment",
                    "metaData": {"segmentId": 2, "name": "Kitchen"},
                    "compressedPixels": [
                        400,
                        100,
                        150,
                        400,
                        150,
                        150,
                        400,
                        200,
                        150,
                        400,
                        250,
                        150,
                    ],
                },
            ],
        }
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        with open(test_data_path, "w", encoding="utf-8") as file:
            json.dump(sample_data, file, indent=2)
        _LOGGER.info("Created sample test data at %s", test_data_path)
        return sample_data

    with open(test_data_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    _LOGGER.info("Loaded test data from %s", test_data_path)
    return test_data


def sublist(data: list, chunk_size: int) -> list:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


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


async def async_extract_room_properties(json_data) -> RoomsProperties:
    start_total = time.time()
    room_properties = {}
    pixel_size = json_data.get("pixelSize", 5)
    height = json_data["size"]["y"]
    width = json_data["size"]["x"]
    vacuum_id = "test_instance"

    # Timing variables
    time_mask_creation = 0
    time_contour_extraction = 0
    time_scaling = 0

    for layer in json_data.get("layers", []):
        if layer.get("__class") == "MapLayer" and layer.get("type") == "segment":
            meta_data = layer.get("metaData", {})
            segment_id = meta_data.get("segmentId")
            name = meta_data.get("name", "Room {}".format(segment_id))
            compressed_pixels = layer.get("compressedPixels", [])
            pixels = sublist(compressed_pixels, 3)

            # Time mask creation
            start = time.time()

            # Optimization: Create a smaller mask for just the room area
            if not pixels:
                # Skip if no pixels
                mask = np.zeros((1, 1), dtype=np.uint8)
            else:
                # Convert to numpy arrays for vectorized operations
                pixel_data = np.array(pixels)

                if pixel_data.size > 0:
                    # Find the actual bounds of the room to create a smaller mask
                    # Add padding to ensure we don't lose edge details
                    padding = 10  # Add padding pixels around the room
                    min_x = max(0, int(np.min(pixel_data[:, 0])) - padding)
                    max_x = min(
                        width,
                        int(np.max(pixel_data[:, 0]) + np.max(pixel_data[:, 2]))
                        + padding,
                    )
                    min_y = max(0, int(np.min(pixel_data[:, 1])) - padding)
                    max_y = min(height, int(np.max(pixel_data[:, 1]) + 1) + padding)

                    # Create a smaller mask for just the room area (much faster)
                    local_width = max_x - min_x
                    local_height = max_y - min_y

                    # Skip if dimensions are invalid
                    if local_width <= 0 or local_height <= 0:
                        mask = np.zeros((1, 1), dtype=np.uint8)
                    else:
                        # Create a smaller mask
                        local_mask = np.zeros(
                            (local_height, local_width), dtype=np.uint8
                        )

                        # Fill the mask efficiently
                        for x, y, length in pixel_data:
                            x, y, length = int(x), int(y), int(length)
                            # Adjust coordinates to local mask
                            local_x = x - min_x
                            local_y = y - min_y

                            # Ensure we're within bounds
                            if (
                                0 <= local_y < local_height
                                and 0 <= local_x < local_width
                            ):
                                # Calculate the end point, clamping to mask width
                                end_x = min(local_x + length, local_width)
                                if (
                                    end_x > local_x
                                ):  # Only process if there's a valid segment
                                    local_mask[local_y, local_x:end_x] = 1

                        # Apply morphological operations
                        struct_elem = np.ones((3, 3), dtype=np.uint8)
                        eroded = binary_erosion(
                            local_mask, structure=struct_elem, iterations=1
                        )
                        mask = binary_dilation(
                            eroded, structure=struct_elem, iterations=1
                        ).astype(np.uint8)

                        # Store the offset for later use when converting coordinates back
                        mask_offset = (min_x, min_y)
                else:
                    mask = np.zeros((1, 1), dtype=np.uint8)

            time_mask_creation += time.time() - start

            # Time contour extraction
            start = time.time()

            # Extract contour from the mask
            if "mask_offset" in locals():
                # If we're using a local mask, we need to adjust the coordinates
                outline = convex_hull_outline(mask)
                if outline:
                    # Adjust coordinates back to global space
                    offset_x, offset_y = mask_offset
                    outline = [(x + offset_x, y + offset_y) for (x, y) in outline]
                    # Clear the mask_offset variable for the next iteration
                    del mask_offset
            else:
                # Regular extraction without offset
                outline = convex_hull_outline(mask)

            time_contour_extraction += time.time() - start

            if not outline:
                _LOGGER.warning(
                    "Skipping segment %s: no outline could be generated", segment_id
                )
                continue

            # Use coordinates as-is without flipping Y coordinates
            # This prevents the large Y values caused by height - 1 - y transformation
            outline = [(x, y) for (x, y) in outline]

            xs, ys = zip(*outline)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            room_id = str(segment_id)

            # Time coordinate scaling
            start = time.time()
            # Scale coordinates by pixel_size and convert to regular Python integers
            # No Y-coordinate flipping is needed
            scaled_outline = [
                (int(x * pixel_size), int(y * pixel_size)) for x, y in outline
            ]
            room_properties[room_id] = {
                "number": segment_id,
                "outline": scaled_outline,
                "name": name,
                "x": int(((x_min + x_max) * pixel_size) // 2),
                "y": int(((y_min + y_max) * pixel_size) // 2),
            }
            time_scaling += time.time() - start

    RoomStore(vacuum_id, room_properties)

    # Log timing information
    total_time = time.time() - start_total
    _LOGGER.info("Room extraction timing breakdown:")
    _LOGGER.info("  Total time: %.3fs", total_time)
    _LOGGER.info(
        "  Mask creation: %.3fs (%.1f%%)",
        time_mask_creation,
        time_mask_creation / total_time * 100,
    )
    _LOGGER.info(
        "  Contour extraction: %.3fs (%.1f%%)",
        time_contour_extraction,
        time_contour_extraction / total_time * 100,
    )
    _LOGGER.info(
        "  Coordinate scaling: %.3fs (%.1f%%)",
        time_scaling,
        time_scaling / total_time * 100,
    )
    _LOGGER.info("Room Properties: %s", room_properties)
    return room_properties


async def main():
    test_data = load_test_data()
    if test_data is None:
        _LOGGER.error("Failed to load test data")
        return

    _LOGGER.info("Extracting room properties...")
    room_properties = await async_extract_room_properties(test_data)
    _LOGGER.info("Found %d rooms", len(room_properties))
    for room_id, props in room_properties.items():
        _LOGGER.info(
            "Room %s: %s at (%d, %d)", room_id, props["name"], props["x"], props["y"]
        )
        _LOGGER.info("  Outline: %s", props["outline"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        _LOGGER.error("Error running async code: %s", e)
