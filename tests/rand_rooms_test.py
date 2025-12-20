"""
Test file for developing the RandRoomsHandler class.
This class will enhance room boundary detection for Rand25 vacuums.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull


# Add the parent directory to the path so we can import the SCR module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.drawable_elements import (
    DrawableElement,
    DrawingConfig,
)
from SCR.valetudo_map_parser.config.types import RoomsProperties
from SCR.valetudo_map_parser.map_data import RandImageData


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


class RandRoomsHandler:
    """
    Handler for extracting and managing room data from Rand25 vacuum maps.

    This class provides methods to:
    - Extract room outlines using the Convex Hull algorithm
    - Process room properties from JSON data and destinations JSON
    - Generate room masks and extract contours

    All methods are async for better integration with the rest of the codebase.
    """

    def __init__(self, vacuum_id: str, drawing_config: Optional[DrawingConfig] = None):
        """
        Initialize the RandRoomsHandler.

        Args:
            vacuum_id: Identifier for the vacuum
            drawing_config: Configuration for which elements to draw (optional)
        """
        self.vacuum_id = vacuum_id
        self.drawing_config = drawing_config
        self.current_json_data = (
            None  # Will store the current JSON data being processed
        )
        self.segment_data = None  # Segment data
        self.outlines = None  # Outlines data

    @staticmethod
    def sublist(data: list, chunk_size: int) -> list:
        """Split a list into chunks of specified size."""
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    @staticmethod
    def convex_hull_outline(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Generate a convex hull outline from a set of points.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            List of (x, y) tuples forming the convex hull outline
        """
        if len(points) == 0:
            return []

        # Convert to numpy array for processing
        points_array = np.array(points)

        if len(points) < 3:
            # Not enough points for a convex hull, return the points as is
            return [(int(x), int(y)) for x, y in points_array]

        try:
            # Calculate the convex hull
            hull = ConvexHull(points_array)

            # Extract the vertices in order
            hull_points = [
                (int(points_array[vertex][0]), int(points_array[vertex][1]))
                for vertex in hull.vertices
            ]

            # Close the polygon by adding the first point at the end
            if hull_points[0] != hull_points[-1]:
                hull_points.append(hull_points[0])

            return hull_points

        except Exception as e:
            _LOGGER.warning(f"Error calculating convex hull: {e}")

            # Fallback to bounding box if convex hull fails
            x_min, y_min = np.min(points_array, axis=0)
            x_max, y_max = np.max(points_array, axis=0)

            return [
                (int(x_min), int(y_min)),  # Top-left
                (int(x_max), int(y_min)),  # Top-right
                (int(x_max), int(y_max)),  # Bottom-right
                (int(x_min), int(y_max)),  # Bottom-left
                (int(x_min), int(y_min)),  # Back to top-left to close the polygon
            ]

    async def _process_segment_data(
        self, segment_data: List, segment_id: int, pixel_size: int
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process a single segment and extract its outline.

        Args:
            segment_data: The segment pixel data
            segment_id: The ID of the segment
            pixel_size: The size of each pixel

        Returns:
            Tuple of (room_id, room_data) or (None, None) if processing failed
        """
        # Check if this room is enabled in the drawing configuration
        if self.drawing_config is not None:
            try:
                # Convert segment_id to room element (ROOM_1 to ROOM_15)
                room_element_id = int(segment_id)
                if 1 <= room_element_id <= 15:
                    room_element = getattr(
                        DrawableElement, f"ROOM_{room_element_id}", None
                    )
                    if room_element:
                        is_enabled = self.drawing_config.is_enabled(room_element)
                        if not is_enabled:
                            # Skip this room if it's disabled
                            _LOGGER.debug("Skipping disabled room %s", segment_id)
                            return None, None
            except (ValueError, TypeError):
                # If segment_id is not a valid integer, we can't map it to a room element
                # In this case, we'll include the room (fail open)
                _LOGGER.debug(
                    "Could not convert segment_id %s to room element", segment_id
                )

        # Skip if no pixels
        if not segment_data:
            return None, None

        # Extract points from segment data
        points = []
        for x, y, _ in segment_data:
            points.append((int(x), int(y)))

        if not points:
            return None, None

        # Use convex hull to get the outline
        outline = self.convex_hull_outline(points)
        if not outline:
            return None, None

        # Calculate bounding box for the room
        xs, ys = zip(*outline)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Scale coordinates by pixel_size
        scaled_outline = [
            (int(x * pixel_size), int(y * pixel_size)) for x, y in outline
        ]

        room_id = str(segment_id)
        room_data = {
            "number": segment_id,
            "outline": scaled_outline,
            "name": f"Room {segment_id}",  # Default name, will be updated from destinations
            "x": int(((x_min + x_max) * pixel_size) // 2),
            "y": int(((y_min + y_max) * pixel_size) // 2),
        }

        return room_id, room_data

    async def async_extract_room_properties(
        self, json_data: Dict[str, Any], destinations: Dict[str, Any]
    ) -> RoomsProperties:
        """
        Extract room properties from the JSON data and destinations.

        Args:
            json_data: The JSON data from the vacuum
            destinations: The destinations JSON containing room names and IDs

        Returns:
            Dictionary of room properties
        """
        start_total = time.time()
        room_properties = {}

        # Get basic map information
        unsorted_id = RandImageData.get_rrm_segments_ids(json_data)
        size_x, size_y = RandImageData.get_rrm_image_size(json_data)
        top, left = RandImageData.get_rrm_image_position(json_data)
        pixel_size = 50  # Rand25 vacuums use a larger pixel size to match the original implementation

        # Get segment data and outlines if not already available
        if not self.segment_data or not self.outlines:
            (
                self.segment_data,
                self.outlines,
            ) = await RandImageData.async_get_rrm_segments(
                json_data, size_x, size_y, top, left, True
            )

        # Process destinations JSON to get room names
        dest_json = destinations
        room_data = dest_json.get("rooms", [])
        room_id_to_data = {room["id"]: room for room in room_data}

        # Process each segment
        if unsorted_id and self.segment_data and self.outlines:
            for idx, segment_id in enumerate(unsorted_id):
                # Extract points from segment data
                points = []
                for x, y, _ in self.segment_data[idx]:
                    points.append((int(x), int(y)))

                if not points:
                    continue

                # Use convex hull to get the outline
                outline = self.convex_hull_outline(points)
                if not outline:
                    continue

                # Scale coordinates by pixel_size
                scaled_outline = [
                    (int(x * pixel_size), int(y * pixel_size)) for x, y in outline
                ]

                # Calculate center point
                xs, ys = zip(*outline)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                center_x = int(((x_min + x_max) * pixel_size) // 2)
                center_y = int(((y_min + y_max) * pixel_size) // 2)

                # Create room data
                room_id = str(segment_id)
                room_data = {
                    "number": segment_id,
                    "outline": scaled_outline,
                    "name": f"Room {segment_id}",  # Default name, will be updated from destinations
                    "x": center_x,
                    "y": center_y,
                }

                # Update room name from destinations if available
                if segment_id in room_id_to_data:
                    room_info = room_id_to_data[segment_id]
                    room_data["name"] = room_info.get("name", room_data["name"])

                room_properties[room_id] = room_data

        # Log timing information
        total_time = time.time() - start_total
        _LOGGER.debug("Room extraction Total time: %.3fs", total_time)

        return room_properties


def load_test_data():
    """Load test data from the rand.json file."""
    test_file_path = os.path.join(os.path.dirname(__file__), "rand.json")
    if not os.path.exists(test_file_path):
        _LOGGER.warning(f"Test data file not found: {test_file_path}")
        return None

    with open(test_file_path, "r") as file:
        test_data = json.load(file)

    _LOGGER.info(f"Loaded test data from {test_file_path}")
    return test_data


def load_destinations_data():
    """Load sample destinations data."""
    return {
        "spots": [{"name": "test_point", "coordinates": [25566, 27289]}],
        "zones": [
            {"name": "test_zone", "coordinates": [[20809, 25919, 22557, 26582, 1]]}
        ],
        "rooms": [
            {"name": "Bathroom", "id": 19},
            {"name": "Bedroom", "id": 20},
            {"name": "Entrance", "id": 18},
            {"name": "Kitchen", "id": 17},
            {"name": "Living Room", "id": 16},
        ],
        "updated": 1746298038728,
    }


async def test_rand_rooms_handler():
    """Test the RandRoomsHandler class."""
    _LOGGER.info("Starting test_rand_rooms_handler...")

    # Load test data
    test_data = load_test_data()
    if not test_data:
        _LOGGER.error("Failed to load test data")
        return

    # Load destinations data
    destinations = load_destinations_data()

    # Create a drawing config
    drawing_config = DrawingConfig()

    # Create a handler instance
    handler = RandRoomsHandler("test_vacuum", drawing_config)

    # Extract room properties
    try:
        _LOGGER.info("Extracting room properties...")
        room_properties = await handler.async_extract_room_properties(
            test_data, destinations
        )

        if room_properties:
            _LOGGER.info(
                f"Successfully extracted {len(room_properties)} rooms: {room_properties}"
            )
            for room_id, props in room_properties.items():
                _LOGGER.info(f"Room {room_id}: {props['name']}")
                _LOGGER.info(f"  Outline points: {len(props['outline'])}")
                _LOGGER.info(f"  Center: ({props['x']}, {props['y']})")
        else:
            _LOGGER.warning("No room properties extracted")

    except Exception as e:
        _LOGGER.error(f"Error extracting room properties: {e}", exc_info=True)


def __main__():
    """Main function."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(test_rand_rooms_handler())
    finally:
        loop.close()


if __name__ == "__main__":
    __main__()
