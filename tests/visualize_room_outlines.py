#!/usr/bin/env python3
"""
Visualize Room Outlines

This script extracts room outlines from map data and visualizes them.
It focuses on preserving the actual shape of rooms without regularization.

Usage:
    python3 visualize_room_outlines.py [path_to_map_data.json]

If no path is provided, it will use the test.json file in the tests directory.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Import the room extraction code from manage_rooms.py
from manage_rooms import load_test_data, HypferRoomsHandler
from rooms_test import async_extract_room_properties
from SCR.valetudo_map_parser.rooms_handler import RoomsHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s (line %(lineno)d) - %(message)s",
)
_LOGGER = logging.getLogger(__name__)

# Define colors for visualization (RGB)
ROOM_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Blue
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
]


def create_map_visualization(
    room_properties: Dict[str, Dict[str, Any]], output_path: str
) -> None:
    """
    Create a visualization of the map with room outlines.

    Args:
        room_properties: Dictionary of room properties
        output_path: Path to save the visualization
    """
    # Find the overall bounds
    all_x = []
    all_y = []

    for props in room_properties.values():
        outline = props.get("outline", [])
        if outline:
            all_x.extend([p[0] for p in outline])
            all_y.extend([p[1] for p in outline])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    padding = int(max(max_x - min_x, max_y - min_y) * 0.05)
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Calculate image dimensions
    width = max_x - min_x
    height = max_y - min_y

    # Create the image
    img = Image.new("RGBA", (width, height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw grid lines for better orientation
    grid_spacing = 500  # Grid lines every 500 pixels
    grid_color = (200, 200, 200)  # Light gray

    # Draw vertical grid lines
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)

    # Draw horizontal grid lines
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Draw room outlines
    for i, (room_id, props) in enumerate(room_properties.items()):
        outline = props.get("outline", [])
        if not outline:
            continue

        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        room_name = props.get("name", f"Room {room_id}")

        # Convert outline to screen coordinates
        points = [(int(p[0]) - min_x, int(p[1]) - min_y) for p in outline]

        # Draw outline with thicker border and semi-transparent fill
        # Add alpha channel for transparency
        fill_color = color + (64,)
        draw.polygon(points, outline=color, fill=fill_color, width=2)

        # Draw points with numbers to show the order
        for j, point in enumerate(points):
            # Draw point
            point_color = (
                (0, 0, 0) if j == 0 else (100, 100, 100)
            )  # First point is black, others gray
            draw.ellipse(
                (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                fill=point_color,
            )

            # Draw point number
            if (
                j < len(points) - 1 or points[0] != points[-1]
            ):  # Skip last point if it's the same as first
                draw.text((point[0] + 5, point[1] - 5), str(j), fill=(50, 50, 50))

        # Draw center point
        center_x, center_y = props.get("x", 0), props.get("y", 0)
        center_x -= min_x
        center_y -= min_y
        draw.ellipse(
            (center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill=(255, 0, 0)
        )

        # Draw room ID and name
        draw.text(
            (center_x - 10, center_y + 10), f"{room_id}: {room_name}", fill=(0, 0, 0)
        )

        # Calculate and show area
        area = calculate_polygon_area(outline)
        draw.text((center_x - 10, center_y + 25), f"Area: {area:.0f}", fill=(0, 0, 0))

        # Show point count
        draw.text(
            (center_x - 10, center_y + 40), f"Points: {len(outline)}", fill=(0, 0, 0)
        )

    # Draw legend
    legend_y = height - 100
    draw.text((10, legend_y), "Room Legend:", fill=(0, 0, 0))
    legend_y += 20

    # Create two columns for the legend
    col_width = width // 2
    row_height = 20
    max_rooms_per_col = 6

    for i, (room_id, props) in enumerate(room_properties.items()):
        col = i // max_rooms_per_col
        row = i % max_rooms_per_col

        x = 10 + col * col_width
        y = legend_y + row * row_height

        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        room_name = props.get("name", f"Room {room_id}")

        # Draw color box
        draw.rectangle((x, y, x + 15, y + 15), fill=color)

        # Draw room name and ID
        draw.text((x + 20, y), f"{room_id}: {room_name}", fill=(0, 0, 0))

    # Save image
    img.save(output_path)
    img.show()
    _LOGGER.info(f"Map visualization saved to {output_path}")


def calculate_polygon_area(polygon: List[Tuple[int, int]]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.

    Args:
        polygon: List of (x, y) points forming the polygon

    Returns:
        The area of the polygon
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2.0


async def main():
    # Get the map data file path from command line or use default
    if len(sys.argv) > 1:
        map_data_path = sys.argv[1]
        with open(map_data_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)
    else:
        map_data = load_test_data()

    if not map_data:
        _LOGGER.error("Failed to load map data")
        return

    # Extract room properties with timing
    _LOGGER.info("Extracting room properties...")
    start_time = time.time()
    rooms_handler = RoomsHandler("test")
    room_properties = await rooms_handler.async_extract_room_properties(map_data)
    end_time = time.time()
    extraction_time = end_time - start_time

    _LOGGER.info(f"Found {len(room_properties)} rooms in {extraction_time:.3f} seconds")

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create map visualization with timing
    _LOGGER.info("Creating map visualization...")
    start_time = time.time()
    output_path = os.path.join(output_dir, "map_comparison.png")
    create_map_visualization(room_properties, output_path)
    _LOGGER.debug(f"Output path: {room_properties}")
    end_time = time.time()
    visualization_time = end_time - start_time
    _LOGGER.info(f"Map visualization created in {visualization_time:.3f} seconds")

    # Print a summary of the rooms
    _LOGGER.info("\nRoom Properties Summary:")
    for room_id, props in room_properties.items():
        outline = props.get("outline", [])
        area = calculate_polygon_area(outline)

        _LOGGER.info(f"Room {room_id} ({props.get('name', 'Unknown')}):")
        _LOGGER.info(f"  Points: {len(outline)}")
        _LOGGER.info(f"  Area: {area:.2f}")
        _LOGGER.info(
            f"  Center: ({props.get('x', 'Unknown')}, {props.get('y', 'Unknown')})"
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        _LOGGER.error(f"Error running visualization: {e}", exc_info=True)
