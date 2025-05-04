"""
Room Outline Extraction Utilities.
Uses scipy for efficient room outline extraction.
Version: 0.1.9
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .types import LOGGER


async def extract_room_outline_with_scipy(
    room_mask, min_x, min_y, max_x, max_y, file_name=None, room_id=None
):
    """Extract a room outline using scipy for contour finding.

    Args:
        room_mask: Binary mask of the room (1 for room, 0 for non-room)
        min_x, min_y, max_x, max_y: Bounding box coordinates
        file_name: Optional file name for logging
        room_id: Optional room ID for logging

    Returns:
        List of points forming the outline of the room
    """
    # If the mask is empty, return a rectangular outline
    if np.sum(room_mask) == 0:
        LOGGER.warning(
            "%s: Empty room mask for room %s, using rectangular outline",
            file_name or "RoomOutline",
            str(room_id) if room_id is not None else "unknown",
        )
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    # Use scipy to clean up the mask (remove noise, fill holes)
    # Fill small holes
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)

    # Remove small objects
    labeled_array, num_features = ndimage.label(room_mask)
    if num_features > 1:
        # Find the largest connected component
        component_sizes = np.bincount(labeled_array.ravel())[1:]
        largest_component = np.argmax(component_sizes) + 1
        room_mask = (labeled_array == largest_component).astype(np.uint8)

    # Find the boundary points by tracing the perimeter
    boundary_points = []
    height, width = room_mask.shape

    # Scan horizontally (top and bottom edges)
    for x in range(width):
        # Top edge
        for y in range(height):
            if room_mask[y, x] == 1:
                boundary_points.append((x + min_x, y + min_y))
                break

        # Bottom edge
        for y in range(height - 1, -1, -1):
            if room_mask[y, x] == 1:
                boundary_points.append((x + min_x, y + min_y))
                break

    # Scan vertically (left and right edges)
    for y in range(height):
        # Left edge
        for x in range(width):
            if room_mask[y, x] == 1:
                boundary_points.append((x + min_x, y + min_y))
                break

        # Right edge
        for x in range(width - 1, -1, -1):
            if room_mask[y, x] == 1:
                boundary_points.append((x + min_x, y + min_y))
                break

    # Remove duplicates while preserving order
    unique_points = []
    for point in boundary_points:
        if point not in unique_points:
            unique_points.append(point)

    # If we have too few points, return a simple rectangle
    if len(unique_points) < 4:
        LOGGER.warning(
            "%s: Too few boundary points for room %s, using rectangular outline",
            file_name or "RoomOutline",
            str(room_id) if room_id is not None else "unknown",
        )
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    # Simplify the outline by keeping only significant points
    simplified = simplify_outline(unique_points, tolerance=5)

    LOGGER.debug(
        "%s: Extracted outline for room %s with %d points",
        file_name or "RoomOutline",
        str(room_id) if room_id is not None else "unknown",
        len(simplified),
    )

    return simplified


def simplify_outline(points, tolerance=5):
    """Simplify an outline by removing points that don't contribute much to the shape."""
    if len(points) <= 4:
        return points

    # Start with the first point
    simplified = [points[0]]

    # Process remaining points
    for i in range(1, len(points) - 1):
        # Get previous and next points
        prev = simplified[-1]
        current = points[i]
        next_point = points[i + 1]

        # Calculate vectors
        v1 = (current[0] - prev[0], current[1] - prev[1])
        v2 = (next_point[0] - current[0], next_point[1] - current[1])

        # Calculate change in direction
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        len_v1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        len_v2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

        # Avoid division by zero
        if len_v1 == 0 or len_v2 == 0:
            continue

        # Calculate cosine of angle between vectors
        cos_angle = dot_product / (len_v1 * len_v2)

        # If angle is significant or distance is large, keep the point
        if abs(cos_angle) < 0.95 or len_v1 > tolerance or len_v2 > tolerance:
            simplified.append(current)

    # Add the last point
    simplified.append(points[-1])

    return simplified
