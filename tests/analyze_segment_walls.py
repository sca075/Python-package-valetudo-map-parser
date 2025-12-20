#!/usr/bin/env python3
"""
Analyze the relationship between segment data and wall data.
This script extracts segment and wall data from test.json and analyzes their relationship.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s (line %(lineno)d) - %(message)s",
)
_LOGGER = logging.getLogger(__name__)


def load_test_data():
    """Load the test.json file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(script_dir, "test.json")

    if not os.path.exists(test_data_path):
        _LOGGER.error(f"Test data file not found: {test_data_path}")
        return None

    with open(test_data_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    _LOGGER.info(f"Loaded test data from {test_data_path}")
    return test_data


def extract_segment_data(json_data: Dict[str, Any], segment_id: int) -> List[List[int]]:
    """
    Extract segment data for a specific segment ID.

    Args:
        json_data: The JSON data from test.json
        segment_id: The segment ID to extract

    Returns:
        List of [x, y, length] triplets for the segment
    """
    segment_pixels = []

    for layer in json_data.get("layers", []):
        if (
            layer.get("__class") == "MapLayer"
            and layer.get("type") == "segment"
            and layer.get("metaData", {}).get("segmentId") == segment_id
        ):
            compressed_pixels = layer.get("compressedPixels", [])
            if not compressed_pixels:
                continue

            # Process pixels in triplets (x, y, length)
            for i in range(0, len(compressed_pixels), 3):
                if i + 2 < len(compressed_pixels):
                    x = compressed_pixels[i]
                    y = compressed_pixels[i + 1]
                    length = compressed_pixels[i + 2]
                    segment_pixels.append([x, y, length])

    return segment_pixels


def extract_wall_data(json_data: Dict[str, Any]) -> List[List[int]]:
    """
    Extract wall data from the JSON.

    Args:
        json_data: The JSON data from test.json

    Returns:
        List of [x, y, length] triplets for the walls
    """
    wall_pixels = []

    for layer in json_data.get("layers", []):
        if layer.get("__class") == "MapLayer" and layer.get("type") == "wall":
            compressed_pixels = layer.get("compressedPixels", [])
            if not compressed_pixels:
                continue

            # Process pixels in triplets (x, y, length)
            for i in range(0, len(compressed_pixels), 3):
                if i + 2 < len(compressed_pixels):
                    x = compressed_pixels[i]
                    y = compressed_pixels[i + 1]
                    length = compressed_pixels[i + 2]
                    wall_pixels.append([x, y, length])

    return wall_pixels


def find_adjacent_pixels(
    segment_pixels: List[List[int]], wall_pixels: List[List[int]]
) -> List[Tuple[List[int], List[int]]]:
    """
    Find segment pixels that are adjacent to wall pixels.

    Args:
        segment_pixels: List of [x, y, length] triplets for the segment
        wall_pixels: List of [x, y, length] triplets for the walls

    Returns:
        List of tuples (segment_pixel, wall_pixel) where segment_pixel is adjacent to wall_pixel
    """
    adjacent_pairs = []

    # Expand segment pixels into individual coordinates
    segment_coords = []
    for x, y, length in segment_pixels:
        for i in range(length):
            segment_coords.append((x + i, y))

    # Expand wall pixels into individual coordinates
    wall_coords = []
    for x, y, length in wall_pixels:
        for i in range(length):
            wall_coords.append((x + i, y))

    # Find segment pixels that are adjacent to wall pixels
    for sx, sy in segment_coords:
        for wx, wy in wall_coords:
            # Check if the segment pixel is adjacent to the wall pixel
            if abs(sx - wx) <= 1 and abs(sy - wy) <= 1:
                adjacent_pairs.append(((sx, sy), (wx, wy)))
                break

    return adjacent_pairs


def analyze_segment_wall_relationship(segment_id: int):
    """
    Analyze the relationship between a segment and walls.

    Args:
        segment_id: The segment ID to analyze
    """
    # Load test data
    json_data = load_test_data()
    if not json_data:
        return

    # Extract segment and wall data
    segment_pixels = extract_segment_data(json_data, segment_id)
    wall_pixels = extract_wall_data(json_data)

    # Get pixel size
    pixel_size = json_data.get("pixelSize", 5)

    # Get segment name
    segment_name = "Unknown"
    for layer in json_data.get("layers", []):
        if (
            layer.get("__class") == "MapLayer"
            and layer.get("type") == "segment"
            and layer.get("metaData", {}).get("segmentId") == segment_id
        ):
            segment_name = layer.get("metaData", {}).get("name", f"Room {segment_id}")
            break

    _LOGGER.info(f"Analyzing segment {segment_id} ({segment_name})")
    _LOGGER.info(f"Pixel size: {pixel_size}")
    _LOGGER.info(f"Found {len(segment_pixels)} segment pixel runs")
    _LOGGER.info(f"Found {len(wall_pixels)} wall pixel runs")

    # Calculate total pixels
    total_segment_pixels = sum(length for _, _, length in segment_pixels)
    total_wall_pixels = sum(length for _, _, length in wall_pixels)
    _LOGGER.info(f"Total segment pixels: {total_segment_pixels}")
    _LOGGER.info(f"Total wall pixels: {total_wall_pixels}")

    # Find segment pixels that are adjacent to wall pixels
    adjacent_pairs = find_adjacent_pixels(segment_pixels, wall_pixels)
    _LOGGER.info(f"Found {len(adjacent_pairs)} segment pixels adjacent to wall pixels")

    # Save results to output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save segment data
    segment_data_path = os.path.join(output_dir, f"segment_{segment_id}_data.json")
    with open(segment_data_path, "w", encoding="utf-8") as f:
        json.dump(segment_pixels, f, indent=2)
    _LOGGER.info(f"Segment data saved to {segment_data_path}")

    # Save wall data
    wall_data_path = os.path.join(output_dir, "wall_data.json")
    with open(wall_data_path, "w", encoding="utf-8") as f:
        json.dump(wall_pixels, f, indent=2)
    _LOGGER.info(f"Wall data saved to {wall_data_path}")

    # Save adjacent pairs
    adjacent_pairs_path = os.path.join(
        output_dir, f"segment_{segment_id}_adjacent_walls.json"
    )
    with open(adjacent_pairs_path, "w", encoding="utf-8") as f:
        # Convert tuples to lists for JSON serialization
        serializable_pairs = [
            {"segment": list(segment), "wall": list(wall)}
            for segment, wall in adjacent_pairs[
                :100
            ]  # Limit to 100 pairs to avoid huge files
        ]
        json.dump(serializable_pairs, f, indent=2)
    _LOGGER.info(f"Adjacent pairs data saved to {adjacent_pairs_path}")

    # Create a simple visualization of the segment and walls
    _LOGGER.info("\nTo visualize the data, run: python3 visualize_room_outlines.py")


if __name__ == "__main__":
    try:
        # Analyze segment 1
        analyze_segment_wall_relationship(1)
    except Exception as e:
        _LOGGER.error(f"Error analyzing segment-wall relationship: {e}", exc_info=True)
