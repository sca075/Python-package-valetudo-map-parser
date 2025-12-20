#!/usr/bin/env python3
"""
Analyze the connections between Room 2, Room 7, and Room 10.
"""

import json
import logging
import os

import numpy as np

# import matplotlib.pyplot as plt
from scipy import ndimage


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_LOGGER = logging.getLogger(__name__)


def main():
    # Load test data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(script_dir, "test.json")

    with open(test_data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Get map dimensions and pixel size
    pixel_size = data.get("pixelSize", 5)
    height = data["size"]["y"]
    width = data["size"]["x"]

    # Create a combined mask for all rooms
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Create individual masks for each room
    room2_mask = np.zeros((height, width), dtype=np.uint8)
    room7_mask = np.zeros((height, width), dtype=np.uint8)
    room10_mask = np.zeros((height, width), dtype=np.uint8)

    # Process each segment
    for layer in data.get("layers", []):
        if layer.get("__class") == "MapLayer" and layer.get("type") == "segment":
            segment_id = layer.get("metaData", {}).get("segmentId")
            name = layer.get("metaData", {}).get("name", f"Room {segment_id}")

            # Skip if not one of our target rooms
            if segment_id not in ["2", "7", "10"]:
                continue

            _LOGGER.info(f"Processing {name} (ID: {segment_id})")

            # Extract compressed pixels
            compressed_pixels = layer.get("compressedPixels", [])
            pixels = [
                compressed_pixels[i : i + 3]
                for i in range(0, len(compressed_pixels), 3)
            ]

            # Create a mask for this room
            room_mask = np.zeros((height, width), dtype=np.uint8)
            for pixel_run in pixels:
                x, y, length = pixel_run
                if 0 <= y < height and 0 <= x < width and x + length <= width:
                    room_mask[y, x : x + length] = 1

            # Add to the combined mask with different values for each room
            if segment_id == "2":
                room2_mask = room_mask
                combined_mask[room_mask == 1] = 1
            elif segment_id == "7":
                room7_mask = room_mask
                combined_mask[room_mask == 1] = 2
            elif segment_id == "10":
                room10_mask = room_mask
                combined_mask[room_mask == 1] = 3

    # Check if the rooms are connected
    # Find connected components in the combined mask
    labeled_array, num_features = ndimage.label(combined_mask > 0)

    _LOGGER.info(f"Number of connected components in the combined mask: {num_features}")

    # Check which rooms are in which components
    for i in range(1, num_features + 1):
        component = labeled_array == i
        room2_overlap = np.any(component & (room2_mask == 1))
        room7_overlap = np.any(component & (room7_mask == 1))
        room10_overlap = np.any(component & (room10_mask == 1))

        _LOGGER.info(
            f"Component {i} contains: Room 2: {room2_overlap}, Room 7: {room7_overlap}, Room 10: {room10_overlap}"
        )

    # Check the distance between rooms
    # Find the boundaries of each room
    room2_indices = np.where(room2_mask > 0)
    room7_indices = np.where(room7_mask > 0)
    room10_indices = np.where(room10_mask > 0)

    if len(room2_indices[0]) > 0 and len(room7_indices[0]) > 0:
        # Calculate the minimum distance between Room 2 and Room 7
        min_distance = float("inf")
        closest_point_room2 = None
        closest_point_room7 = None

        for i in range(len(room2_indices[0])):
            y2, x2 = room2_indices[0][i], room2_indices[1][i]
            for j in range(len(room7_indices[0])):
                y7, x7 = room7_indices[0][j], room7_indices[1][j]
                distance = np.sqrt((x2 - x7) ** 2 + (y2 - y7) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point_room2 = (x2, y2)
                    closest_point_room7 = (x7, y7)

        _LOGGER.info(f"Minimum distance between Room 2 and Room 7: {min_distance}")
        _LOGGER.info(
            f"Closest point in Room 2: {closest_point_room2}, scaled: {(closest_point_room2[0] * pixel_size, closest_point_room2[1] * pixel_size)}"
        )
        _LOGGER.info(
            f"Closest point in Room 7: {closest_point_room7}, scaled: {(closest_point_room7[0] * pixel_size, closest_point_room7[1] * pixel_size)}"
        )

    if len(room2_indices[0]) > 0 and len(room10_indices[0]) > 0:
        # Calculate the minimum distance between Room 2 and Room 10
        min_distance = float("inf")
        closest_point_room2 = None
        closest_point_room10 = None

        for i in range(len(room2_indices[0])):
            y2, x2 = room2_indices[0][i], room2_indices[1][i]
            for j in range(len(room10_indices[0])):
                y10, x10 = room10_indices[0][j], room10_indices[1][j]
                distance = np.sqrt((x2 - x10) ** 2 + (y2 - y10) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point_room2 = (x2, y2)
                    closest_point_room10 = (x10, y10)

        _LOGGER.info(f"Minimum distance between Room 2 and Room 10: {min_distance}")
        _LOGGER.info(
            f"Closest point in Room 2: {closest_point_room2}, scaled: {(closest_point_room2[0] * pixel_size, closest_point_room2[1] * pixel_size)}"
        )
        _LOGGER.info(
            f"Closest point in Room 10: {closest_point_room10}, scaled: {(closest_point_room10[0] * pixel_size, closest_point_room10[1] * pixel_size)}"
        )

    # Create a text-based visualization of the rooms
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Now analyze all rooms
    _LOGGER.info("\nAnalyzing all rooms...")

    # Process each segment
    for layer in data.get("layers", []):
        if layer.get("__class") == "MapLayer" and layer.get("type") == "segment":
            segment_id = layer.get("metaData", {}).get("segmentId")
            name = layer.get("metaData", {}).get("name", f"Room {segment_id}")

            # Extract compressed pixels
            compressed_pixels = layer.get("compressedPixels", [])
            pixels = [
                compressed_pixels[i : i + 3]
                for i in range(0, len(compressed_pixels), 3)
            ]

            # Create a mask for this room
            room_mask = np.zeros((height, width), dtype=np.uint8)
            for pixel_run in pixels:
                x, y, length = pixel_run
                if 0 <= y < height and 0 <= x < width and x + length <= width:
                    room_mask[y, x : x + length] = 1

            # Count the number of pixels in this room
            num_pixels = np.sum(room_mask)

            # Find connected components in this room
            labeled_array, num_features = ndimage.label(room_mask)
            _LOGGER.info(
                f"Room {segment_id} ({name}) has {num_features} connected components"
            )

            # Calculate the bounding box
            y_indices, x_indices = np.where(room_mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                _LOGGER.info(f"  Bounding box: X: {x_min}-{x_max}, Y: {y_min}-{y_max}")
                _LOGGER.info(
                    f"  Scaled: X: {x_min * pixel_size}-{x_max * pixel_size}, Y: {y_min * pixel_size}-{y_max * pixel_size}"
                )

    _LOGGER.info("Analysis complete")


if __name__ == "__main__":
    main()
