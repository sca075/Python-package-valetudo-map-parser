#!/usr/bin/env python3
"""
Analyze Room 12 (Living Room) data to understand why it has such a small outline.
"""

import json
import logging
import os

import numpy as np


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

    # Find Room 12
    room12 = None
    for layer in data.get("layers", []):
        if (
            layer.get("__class") == "MapLayer"
            and layer.get("type") == "segment"
            and layer.get("metaData", {}).get("segmentId") == "12"
        ):
            room12 = layer
            break

    if not room12:
        _LOGGER.error("Room 12 not found in test data")
        return

    # Get map dimensions and pixel size
    pixel_size = data.get("pixelSize", 5)
    height = data["size"]["y"]
    width = data["size"]["x"]

    # Extract compressed pixels
    compressed_pixels = room12.get("compressedPixels", [])
    pixels = [compressed_pixels[i : i + 3] for i in range(0, len(compressed_pixels), 3)]

    _LOGGER.info(f"Room 12 (Living Room) has {len(pixels)} pixel runs")
    _LOGGER.info(f"Map dimensions: {width}x{height}, Pixel size: {pixel_size}")

    # Create a binary mask for the room
    mask = np.zeros((height, width), dtype=np.uint8)
    for pixel_run in pixels:
        x, y, length = pixel_run
        if 0 <= y < height and 0 <= x < width and x + length <= width:
            mask[y, x : x + length] = 1

    # Analyze the mask
    total_pixels = np.sum(mask)
    _LOGGER.info(f"Total pixels in mask: {total_pixels}")

    if total_pixels > 0:
        # Get the bounding box
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        _LOGGER.info(f"Bounding box: X: {x_min}-{x_max}, Y: {y_min}-{y_max}")
        _LOGGER.info(
            f"Scaled bounding box: X: {x_min * pixel_size}-{x_max * pixel_size}, Y: {y_min * pixel_size}-{y_max * pixel_size}"
        )

        # Check if there's a small isolated region
        # Count connected components
        from scipy import ndimage

        labeled_array, num_features = ndimage.label(mask)
        _LOGGER.info(f"Number of connected components: {num_features}")

        # Analyze each component
        for i in range(1, num_features + 1):
            component = labeled_array == i
            component_size = np.sum(component)
            comp_y_indices, comp_x_indices = np.where(component)
            comp_x_min, comp_x_max = np.min(comp_x_indices), np.max(comp_x_indices)
            comp_y_min, comp_y_max = np.min(comp_y_indices), np.max(comp_y_indices)

            _LOGGER.info(f"Component {i}: Size: {component_size} pixels")
            _LOGGER.info(
                f"Component {i} bounding box: X: {comp_x_min}-{comp_x_max}, Y: {comp_y_min}-{comp_y_max}"
            )
            _LOGGER.info(
                f"Component {i} scaled: X: {comp_x_min * pixel_size}-{comp_x_max * pixel_size}, Y: {comp_y_min * pixel_size}-{comp_y_max * pixel_size}"
            )

            # Check if this component matches the tiny outline we're seeing
            if (
                comp_x_min * pixel_size <= 3350
                and comp_x_max * pixel_size >= 3345
                and comp_y_min * pixel_size <= 2540
                and comp_y_max * pixel_size >= 2535
            ):
                _LOGGER.info(f"Found the problematic component: Component {i}")

                # Check the pixel runs that contribute to this component
                for j, (x, y, length) in enumerate(pixels):
                    if comp_x_min <= x <= comp_x_max and comp_y_min <= y <= comp_y_max:
                        _LOGGER.info(f"Pixel run {j}: x={x}, y={y}, length={length}")
    else:
        _LOGGER.warning("Room 12 mask is empty")


if __name__ == "__main__":
    main()
