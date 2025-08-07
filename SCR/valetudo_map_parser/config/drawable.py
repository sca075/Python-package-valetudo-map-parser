"""
Collections of Drawing Utility
Drawable is part of the Image_Handler
used functions to draw the elements on the Numpy Array
that is actually our camera frame.
Version: v0.1.10
Refactored for clarity, consistency, and optimized parameter usage.
Optimized with NumPy and SciPy for better performance.
"""

from __future__ import annotations

import logging
import math
import asyncio
import inspect

import numpy as np
from PIL import ImageDraw, ImageFont

from .color_utils import get_blended_color
from .colors import ColorsManagement
from .types import Color, NumpyArray, PilPNG, Point, Tuple, Union


_LOGGER = logging.getLogger(__name__)


class Drawable:
    """
    Collection of drawing utility functions for the image handlers.
    This class contains static methods to draw various elements on NumPy arrays (images).
    We can't use OpenCV because it is not supported by the Home Assistant OS.
    """

    ERROR_OUTLINE: Color = (0, 0, 0, 255)  # Red color for error messages
    ERROR_COLOR: Color = (
        255,
        0,
        0,
        191,
    )  # Red color with lower opacity for error outlines

    @staticmethod
    async def create_empty_image(
        width: int, height: int, background_color: Color
    ) -> NumpyArray:
        """Create the empty background image NumPy array.
        Background color is specified as an RGBA tuple.
        Optimized: Uses np.empty + broadcast instead of np.full for better performance."""
        # Use np.empty + broadcast instead of np.full (avoids double initialization)
        img_array = np.empty((height, width, 4), dtype=np.uint8)
        img_array[:] = background_color  # Broadcast color to all pixels efficiently
        return img_array

    @staticmethod
    async def from_json_to_image(
        layer: NumpyArray, pixels: Union[dict, list], pixel_size: int, color: Color
    ) -> NumpyArray:
        """Draw the layers (rooms) from the vacuum JSON data onto the image array."""
        image_array = layer
        # Extract alpha from color
        alpha = color[3] if len(color) == 4 else 255

        # Create the full color with alpha
        full_color = color if len(color) == 4 else (*color, 255)

        # Check if we need to blend colors (alpha < 255)
        need_blending = alpha < 255

        # Loop through pixels to find min and max coordinates
        for x, y, z in pixels:
            col = x * pixel_size
            row = y * pixel_size
            # Draw pixels as blocks
            for i in range(z):
                # Get the region to update
                region_slice = (
                    slice(row, row + pixel_size),
                    slice(col + i * pixel_size, col + (i + 1) * pixel_size),
                )

                if need_blending:
                    # Sample the center of the region for blending
                    center_y = row + pixel_size // 2
                    center_x = col + i * pixel_size + pixel_size // 2

                    # Only blend if coordinates are valid
                    if (
                        0 <= center_y < image_array.shape[0]
                        and 0 <= center_x < image_array.shape[1]
                    ):
                        # Get blended color
                        blended_color = ColorsManagement.sample_and_blend_color(
                            image_array, center_x, center_y, full_color
                        )
                        # Apply blended color to the region
                        image_array[region_slice] = blended_color
                    else:
                        # Use original color if out of bounds
                        image_array[region_slice] = full_color
                else:
                    # No blending needed, use direct assignment
                    image_array[region_slice] = full_color

        return image_array

    @staticmethod
    async def battery_charger(
        layers: NumpyArray, x: int, y: int, color: Color
    ) -> NumpyArray:
        """Draw the battery charger on the input layer with color blending."""
        # Check if coordinates are within bounds
        height, width = layers.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return layers

        # Calculate charger dimensions
        charger_width = 10
        charger_height = 20
        start_row = max(0, y - charger_height // 2)
        end_row = min(height, start_row + charger_height)
        start_col = max(0, x - charger_width // 2)
        end_col = min(width, start_col + charger_width)

        # Skip if charger is completely outside the image
        if start_row >= end_row or start_col >= end_col:
            return layers

        # Extract alpha from color
        alpha = color[3] if len(color) == 4 else 255

        # Check if we need to blend colors (alpha < 255)
        if alpha < 255:
            # Sample the center of the charger for blending
            center_y = (start_row + end_row) // 2
            center_x = (start_col + end_col) // 2

            # Get blended color
            blended_color = ColorsManagement.sample_and_blend_color(
                layers, center_x, center_y, color
            )

            # Apply blended color
            layers[start_row:end_row, start_col:end_col] = blended_color
        else:
            # No blending needed, use direct assignment
            layers[start_row:end_row, start_col:end_col] = color

        return layers

    @staticmethod
    async def go_to_flag(
        layer: NumpyArray, center: Point, rotation_angle: int, flag_color: Color
    ) -> NumpyArray:
        """
        Draw a flag centered at specified coordinates on the input layer.
        It uses the rotation angle of the image to orient the flag.
        Includes color blending for better visual integration.
        """
        await asyncio.sleep(0)  # Yield control

        # Check if coordinates are within bounds
        height, width = layer.shape[:2]
        x, y = center
        if not (0 <= x < width and 0 <= y < height):
            return layer

        # Get blended colors for flag and pole
        flag_alpha = flag_color[3] if len(flag_color) == 4 else 255
        pole_color_base = (0, 0, 255)  # Blue for the pole
        pole_alpha = 255

        # Blend flag color if needed
        if flag_alpha < 255:
            flag_color = ColorsManagement.sample_and_blend_color(
                layer, x, y, flag_color
            )

        # Create pole color with alpha
        pole_color: Color = (*pole_color_base, pole_alpha)

        # Blend pole color if needed
        if pole_alpha < 255:
            pole_color = ColorsManagement.sample_and_blend_color(
                layer, x, y, pole_color
            )

        flag_size = 50
        pole_width = 6
        # Adjust flag coordinates based on rotation angle
        if rotation_angle == 90:
            x1 = center[0] + flag_size
            y1 = center[1] - (pole_width // 2)
            x2 = x1 - (flag_size // 4)
            y2 = y1 + (flag_size // 2)
            x3 = center[0] + (flag_size // 2)
            y3 = center[1] - (pole_width // 2)
            xp1, yp1 = center[0], center[1] - (pole_width // 2)
            xp2, yp2 = center[0] + flag_size, center[1] - (pole_width // 2)
        elif rotation_angle == 180:
            x1 = center[0]
            y1 = center[1] - (flag_size // 2)
            x2 = center[0] - (flag_size // 2)
            y2 = y1 + (flag_size // 4)
            x3, y3 = center[0], center[1]
            xp1, yp1 = center[0] + (pole_width // 2), center[1] - flag_size
            xp2, yp2 = center[0] + (pole_width // 2), y3
        elif rotation_angle == 270:
            x1 = center[0] - flag_size
            y1 = center[1] + (pole_width // 2)
            x2 = x1 + (flag_size // 4)
            y2 = y1 - (flag_size // 2)
            x3 = center[0] - (flag_size // 2)
            y3 = center[1] + (pole_width // 2)
            xp1, yp1 = center[0] - flag_size, center[1] + (pole_width // 2)
            xp2, yp2 = center[0], center[1] + (pole_width // 2)
        else:  # rotation_angle == 0 (no rotation)
            x1, y1 = center[0], center[1]
            x2, y2 = center[0] + (flag_size // 2), center[1] + (flag_size // 4)
            x3, y3 = center[0], center[1] + flag_size // 2
            xp1, yp1 = center[0] - (pole_width // 2), y1
            xp2, yp2 = center[0] - (pole_width // 2), center[1] + flag_size

        # Draw flag outline using _polygon_outline
        points = [(x1, y1), (x2, y2), (x3, y3)]
        layer = Drawable._polygon_outline(layer, points, 1, flag_color, flag_color)
        # Draw pole using _line
        layer = Drawable._line(layer, xp1, yp1, xp2, yp2, pole_color, pole_width)
        return layer

    @staticmethod
    def point_inside(x: int, y: int, points: list[Tuple[int, int]]) -> bool:
        """
        Check if a point (x, y) is inside a polygon defined by a list of points.
        """
        n = len(points)
        inside = False
        xinters = 0.0
        p1x, p1y = points[0]
        for i in range(1, n + 1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def _line(
        layer: NumpyArray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Color,
        width: int = 3,
    ) -> NumpyArray:
        """
        Draw a line on a NumPy array (layer) from point A to B using vectorized operations.

        Args:
            layer: The numpy array to draw on
            x1, y1: Start point coordinates
            x2, y2: End point coordinates
            color: Color to draw with
            width: Width of the line
        """
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get blended color for the line
        blended_color = get_blended_color(x1, y1, x2, y2, layer, color)

        # Calculate line length
        length = max(abs(x2 - x1), abs(y2 - y1))
        if length == 0:  # Handle case of a single point
            # Draw a dot with the specified width
            for i in range(-width // 2, (width + 1) // 2):
                for j in range(-width // 2, (width + 1) // 2):
                    if 0 <= x1 + i < layer.shape[1] and 0 <= y1 + j < layer.shape[0]:
                        layer[y1 + j, x1 + i] = blended_color
            return layer

        # Create parametric points along the line
        t = np.linspace(0, 1, length * 2)  # Double the points for smoother lines
        x_coords = np.round(x1 * (1 - t) + x2 * t).astype(int)
        y_coords = np.round(y1 * (1 - t) + y2 * t).astype(int)

        # Draw the line with the specified width
        if width == 1:
            # Fast path for width=1
            for x, y in zip(x_coords, y_coords):
                if 0 <= x < layer.shape[1] and 0 <= y < layer.shape[0]:
                    layer[y, x] = blended_color
        else:
            # For thicker lines, draw a rectangle at each point
            half_width = width // 2
            for x, y in zip(x_coords, y_coords):
                for i in range(-half_width, half_width + 1):
                    for j in range(-half_width, half_width + 1):
                        if (
                            i * i + j * j <= half_width * half_width  # Make it round
                            and 0 <= x + i < layer.shape[1]
                            and 0 <= y + j < layer.shape[0]
                        ):
                            layer[y + j, x + i] = blended_color

        return layer

    @staticmethod
    async def draw_virtual_walls(
        layer: NumpyArray, virtual_walls, color: Color
    ) -> NumpyArray:
        """
        Draw virtual walls on the input layer.
        """
        for wall in virtual_walls:
            for i in range(0, len(wall), 4):
                x1, y1, x2, y2 = wall[i : i + 4]
                # Draw the virtual wall as a line with a fixed width of 6 pixels
                layer = Drawable._line(layer, x1, y1, x2, y2, color, width=6)
        return layer

    @staticmethod
    async def lines(arr: NumpyArray, coords, width: int, color: Color) -> NumpyArray:
        """
        Join the coordinates creating a continuous line (path).
        Optimized with vectorized operations for better performance.
        """

        # Handle case where arr might be a coroutine (shouldn't happen but let's be safe)
        if inspect.iscoroutine(arr):
            arr = await arr

        for i, coord in enumerate(coords):
            x0, y0 = coord[0]
            try:
                x1, y1 = coord[1]
            except IndexError:
                x1, y1 = x0, y0

            # Skip if coordinates are the same
            if x0 == x1 and y0 == y1:
                continue

            # Get blended color for this line segment
            blended_color = get_blended_color(x0, y0, x1, y1, arr, color)

            # Use the optimized line drawing method
            arr = Drawable._line(arr, x0, y0, x1, y1, blended_color, width)

            # Yield control every 100 operations to prevent blocking
            if i % 100 == 0:
                await asyncio.sleep(0)

        return arr

    @staticmethod
    def _filled_circle(
        image: NumpyArray,
        center: Point,
        radius: int,
        color: Color,
        outline_color: Color = None,
        outline_width: int = 0,
    ) -> NumpyArray:
        """
        Draw a filled circle on the image using NumPy.
        Optimized to only process the bounding box of the circle.
        """
        y, x = center
        height, width = image.shape[:2]

        # Calculate the bounding box of the circle
        min_y = max(0, y - radius - outline_width)
        max_y = min(height, y + radius + outline_width + 1)
        min_x = max(0, x - radius - outline_width)
        max_x = min(width, x + radius + outline_width + 1)

        # Create coordinate arrays for the bounding box
        y_indices, x_indices = np.ogrid[min_y:max_y, min_x:max_x]

        # Calculate distances from center
        dist_sq = (y_indices - y) ** 2 + (x_indices - x) ** 2

        # Create masks for the circle and outline
        circle_mask = dist_sq <= radius**2

        # Apply the fill color
        image[min_y:max_y, min_x:max_x][circle_mask] = color

        # Draw the outline if needed
        if outline_width > 0 and outline_color is not None:
            outer_mask = dist_sq <= (radius + outline_width) ** 2
            outline_mask = outer_mask & ~circle_mask
            image[min_y:max_y, min_x:max_x][outline_mask] = outline_color

        return image

    @staticmethod
    def _filled_circle_optimized(
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Color,
        outline_color: Color = None,
        outline_width: int = 0,
    ) -> np.ndarray:
        """
        Optimized _filled_circle ensuring dtype compatibility with uint8.
        """
        x, y = center
        h, w = image.shape[:2]
        color_np = np.array(color, dtype=image.dtype)
        outline_color_np = (
            np.array(outline_color, dtype=image.dtype)
            if outline_color is not None
            else None
        )
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dist_sq = (y_indices - y) ** 2 + (x_indices - x) ** 2
        circle_mask = dist_sq <= radius**2
        image[circle_mask] = color_np
        if outline_width > 0 and outline_color_np is not None:
            outer_mask = dist_sq <= (radius + outline_width) ** 2
            outline_mask = outer_mask & ~circle_mask
            image[outline_mask] = outline_color_np
        return image

    @staticmethod
    def _ellipse(
        image: NumpyArray, center: Point, radius: int, color: Color
    ) -> NumpyArray:
        """
        Draw an ellipse on the image using NumPy.
        """
        x, y = center
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y + radius
        image[y1:y2, x1:x2] = color
        return image

    @staticmethod
    def _polygon_outline(
        arr: NumpyArray,
        points: list[Tuple[int, int]],
        width: int,
        outline_color: Color,
        fill_color: Color = None,
    ) -> NumpyArray:
        """
        Draw the outline of a polygon on the array using _line, and optionally fill it.
        Uses NumPy vectorized operations for improved performance.
        """
        # Draw the outline
        for i, _ in enumerate(points):
            current_point = points[i]
            next_point = points[(i + 1) % len(points)]
            arr = Drawable._line(
                arr,
                current_point[0],
                current_point[1],
                next_point[0],
                next_point[1],
                outline_color,
                width,
            )

        # Fill the polygon if a fill color is provided
        if fill_color is not None:
            # Get the bounding box of the polygon
            min_x = max(0, min(p[0] for p in points))
            max_x = min(arr.shape[1] - 1, max(p[0] for p in points))
            min_y = max(0, min(p[1] for p in points))
            max_y = min(arr.shape[0] - 1, max(p[1] for p in points))

            # Create a mask for the polygon region
            mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)

            # Adjust points to the mask's coordinate system
            adjusted_points = [(p[0] - min_x, p[1] - min_y) for p in points]

            # Create a grid of coordinates and use it to test all points at once
            y_indices, x_indices = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]

            # Test each point in the grid
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    mask[i, j] = Drawable.point_inside(j, i, adjusted_points)

            # Apply the fill color to the masked region
            arr[min_y : max_y + 1, min_x : max_x + 1][mask] = fill_color

        return arr

    @staticmethod
    async def zones(layers: NumpyArray, coordinates, color: Color) -> NumpyArray:
        """
        Draw the zones on the input layer with color blending.
        Optimized with parallel processing for better performance.
        """
        await asyncio.sleep(0)  # Yield control

        dot_radius = 1  # Number of pixels for the dot
        dot_spacing = 4  # Space between dots

        # Process zones in parallel if there are multiple zones
        if len(coordinates) > 1:
            # Create tasks for parallel zone processing
            zone_tasks = []
            for zone in coordinates:
                zone_tasks.append(Drawable._process_single_zone(layers.copy(), zone, color, dot_radius, dot_spacing))

            # Execute all zone processing tasks in parallel
            zone_results = await asyncio.gather(*zone_tasks, return_exceptions=True)

            # Merge results back into the main layer
            for result in zone_results:
                if not isinstance(result, Exception):
                    # Simple overlay - pixels that are different from original get updated
                    mask = result != layers
                    layers[mask] = result[mask]
        else:
            # Single zone - process directly
            for zone in coordinates:
                points = zone["points"]
                min_x = max(0, min(points[::2]))
                max_x = min(layers.shape[1] - 1, max(points[::2]))
                min_y = max(0, min(points[1::2]))
                max_y = min(layers.shape[0] - 1, max(points[1::2]))

                # Skip if zone is outside the image
                if min_x >= max_x or min_y >= max_y:
                    continue

                # Sample a point from the zone to get the background color
                # Use the center of the zone for sampling
                sample_x = (min_x + max_x) // 2
                sample_y = (min_y + max_y) // 2

                # Blend the color with the background color at the sample point
                if 0 <= sample_y < layers.shape[0] and 0 <= sample_x < layers.shape[1]:
                    blended_color = ColorsManagement.sample_and_blend_color(
                        layers, sample_x, sample_y, color
                    )
                else:
                    blended_color = color

                # Create a grid of dot centers
                x_centers = np.arange(min_x, max_x, dot_spacing)
                y_centers = np.arange(min_y, max_y, dot_spacing)

                # Draw dots at each grid point
                for y in y_centers:
                    for x in x_centers:
                        # Create a small mask for the dot
                        y_min = max(0, y - dot_radius)
                        y_max = min(layers.shape[0], y + dot_radius + 1)
                        x_min = max(0, x - dot_radius)
                        x_max = min(layers.shape[1], x + dot_radius + 1)

                        # Create coordinate arrays for the dot
                        y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]

                        # Create a circular mask
                        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= dot_radius**2

                        # Apply the color to the masked region
                        layers[y_min:y_max, x_min:x_max][mask] = blended_color

        return layers

    @staticmethod
    async def _process_single_zone(layers: NumpyArray, zone, color: Color, dot_radius: int, dot_spacing: int) -> NumpyArray:
        """Process a single zone for parallel execution."""
        await asyncio.sleep(0)  # Yield control

        points = zone["points"]
        min_x = max(0, min(points[::2]))
        max_x = min(layers.shape[1] - 1, max(points[::2]))
        min_y = max(0, min(points[1::2]))
        max_y = min(layers.shape[0] - 1, max(points[1::2]))

        # Skip if zone is outside the image
        if min_x >= max_x or min_y >= max_y:
            return layers

        # Sample a point from the zone to get the background color
        sample_x = (min_x + max_x) // 2
        sample_y = (min_y + max_y) // 2

        # Blend the color with the background color at the sample point
        if 0 <= sample_y < layers.shape[0] and 0 <= sample_x < layers.shape[1]:
            blended_color = ColorsManagement.sample_and_blend_color(
                layers, sample_x, sample_y, color
            )
        else:
            blended_color = color

        # Create a dotted pattern within the zone
        for y in range(min_y, max_y + 1, dot_spacing):
            for x in range(min_x, max_x + 1, dot_spacing):
                if Drawable.point_inside(x, y, points):
                    # Draw a small filled circle (dot) using vectorized operations
                    y_min = max(0, y - dot_radius)
                    y_max = min(layers.shape[0], y + dot_radius + 1)
                    x_min = max(0, x - dot_radius)
                    x_max = min(layers.shape[1], x + dot_radius + 1)

                    if y_min < y_max and x_min < x_max:
                        y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
                        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= dot_radius**2
                        layers[y_min:y_max, x_min:x_max][mask] = blended_color

        return layers

    @staticmethod
    async def robot(
        layers: NumpyArray,
        x: int,
        y: int,
        angle: float,
        fill: Color,
        robot_state: str | None = None,
    ) -> NumpyArray:
        """
        Draw the robot on a smaller array to reduce memory cost.
        Optimized with NumPy vectorized operations for better performance.
        """
        # Ensure coordinates are within bounds
        height, width = layers.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return layers

        # Calculate the bounding box for the robot
        radius = 25
        box_size = radius * 2 + 2  # Add a small margin

        # Calculate the region to draw on
        top_left_x = max(0, x - radius - 1)
        top_left_y = max(0, y - radius - 1)
        bottom_right_x = min(width, x + radius + 1)
        bottom_right_y = min(height, y + radius + 1)

        # Skip if the robot is completely outside the image
        if top_left_x >= bottom_right_x or top_left_y >= bottom_right_y:
            return layers

        # Create a temporary layer for the robot
        tmp_width = bottom_right_x - top_left_x
        tmp_height = bottom_right_y - top_left_y
        tmp_layer = layers[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

        # Calculate the robot center in the temporary layer
        tmp_x = x - top_left_x
        tmp_y = y - top_left_y

        # Calculate robot parameters
        r_scaled = radius // 11
        r_cover = r_scaled * 12
        lidar_angle = np.deg2rad(angle + 90)
        r_lidar = r_scaled * 3
        r_button = r_scaled * 1

        # Set colors based on robot state
        if robot_state == "error":
            outline = Drawable.ERROR_OUTLINE
            fill = Drawable.ERROR_COLOR
        else:
            outline = (fill[0] // 2, fill[1] // 2, fill[2] // 2, fill[3])

        # Draw the main robot body
        tmp_layer = Drawable._filled_circle(
            tmp_layer, (tmp_y, tmp_x), radius, fill, outline, 1
        )

        # Draw the robot direction indicator
        angle -= 90
        a1 = ((angle + 90) - 80) / 180 * math.pi
        a2 = ((angle + 90) + 80) / 180 * math.pi
        x1 = int(tmp_x - r_cover * math.sin(a1))
        y1 = int(tmp_y + r_cover * math.cos(a1))
        x2 = int(tmp_x - r_cover * math.sin(a2))
        y2 = int(tmp_y + r_cover * math.cos(a2))

        # Draw the direction line
        if (
            0 <= x1 < tmp_width
            and 0 <= y1 < tmp_height
            and 0 <= x2 < tmp_width
            and 0 <= y2 < tmp_height
        ):
            tmp_layer = Drawable._line(tmp_layer, x1, y1, x2, y2, outline, width=1)

        # Draw the lidar indicator
        lidar_x = int(tmp_x + 15 * np.cos(lidar_angle))
        lidar_y = int(tmp_y + 15 * np.sin(lidar_angle))
        if 0 <= lidar_x < tmp_width and 0 <= lidar_y < tmp_height:
            tmp_layer = Drawable._filled_circle(
                tmp_layer, (lidar_y, lidar_x), r_lidar, outline
            )

        # Draw the button indicator
        butt_x = int(tmp_x - 20 * np.cos(lidar_angle))
        butt_y = int(tmp_y - 20 * np.sin(lidar_angle))
        if 0 <= butt_x < tmp_width and 0 <= butt_y < tmp_height:
            tmp_layer = Drawable._filled_circle(
                tmp_layer, (butt_y, butt_x), r_button, outline
            )

        # Copy the robot layer back to the main layer
        layers[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = tmp_layer

        return layers

    @staticmethod
    def overlay_robot(
        background_image: NumpyArray, robot_image: NumpyArray, x: int, y: int
    ) -> NumpyArray:
        """
        Overlay the robot image on the background image at the specified coordinates.
        """
        robot_height, robot_width, _ = robot_image.shape
        robot_center_x = robot_width // 2
        robot_center_y = robot_height // 2
        top_left_x = x - robot_center_x
        top_left_y = y - robot_center_y
        bottom_right_x = top_left_x + robot_width
        bottom_right_y = top_left_y + robot_height
        background_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = (
            robot_image
        )
        return background_image

    @staticmethod
    def draw_filled_circle(
        image: np.ndarray,
        centers: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Draw multiple filled circles at once using a single NumPy mask.
        """
        h, w = image.shape[:2]
        y_indices, x_indices = np.ogrid[:h, :w]  # Precompute coordinate grids
        mask = np.zeros((h, w), dtype=bool)
        for cx, cy in centers:
            mask |= (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius**2
        image[mask] = color
        return image

    @staticmethod
    def batch_draw_elements(
        image: np.ndarray,
        elements: list,
        element_type: str,
        color: Color,
    ) -> np.ndarray:
        """
        Efficiently draw multiple elements of the same type at once.

        Args:
            image: The image array to draw on
            elements: List of element data (coordinates, etc.)
            element_type: Type of element to draw ('circle', 'line', etc.)
            color: Color to use for drawing

        Returns:
            Modified image array
        """
        if not elements or len(elements) == 0:
            return image

        # Get image dimensions
        height, width = image.shape[:2]

        if element_type == "circle":
            # Extract circle centers and radii
            centers = []
            radii = []
            for elem in elements:
                if isinstance(elem, dict) and "center" in elem and "radius" in elem:
                    centers.append(elem["center"])
                    radii.append(elem["radius"])
                elif isinstance(elem, (list, tuple)) and len(elem) >= 3:
                    # Format: (x, y, radius)
                    centers.append((elem[0], elem[1]))
                    radii.append(elem[2])

            # Process circles with the same radius together
            for radius in set(radii):
                same_radius_centers = [
                    centers[i] for i in range(len(centers)) if radii[i] == radius
                ]
                if same_radius_centers:
                    # Create a combined mask for all circles with this radius
                    mask = np.zeros((height, width), dtype=bool)
                    for cx, cy in same_radius_centers:
                        if 0 <= cx < width and 0 <= cy < height:
                            # Calculate circle bounds
                            min_y = max(0, cy - radius)
                            max_y = min(height, cy + radius + 1)
                            min_x = max(0, cx - radius)
                            max_x = min(width, cx + radius + 1)

                            # Create coordinate arrays for the circle
                            y_indices, x_indices = np.ogrid[min_y:max_y, min_x:max_x]

                            # Add this circle to the mask
                            circle_mask = (y_indices - cy) ** 2 + (
                                x_indices - cx
                            ) ** 2 <= radius**2
                            mask[min_y:max_y, min_x:max_x] |= circle_mask

                    # Apply color to all circles at once
                    image[mask] = color

        elif element_type == "line":
            # Extract line endpoints
            lines = []
            widths = []
            for elem in elements:
                if isinstance(elem, dict) and "start" in elem and "end" in elem:
                    lines.append((elem["start"], elem["end"]))
                    widths.append(elem.get("width", 1))
                elif isinstance(elem, (list, tuple)) and len(elem) >= 4:
                    # Format: (x1, y1, x2, y2, [width])
                    lines.append(((elem[0], elem[1]), (elem[2], elem[3])))
                    widths.append(elem[4] if len(elem) > 4 else 1)

            # Process lines with the same width together
            for width in set(widths):
                same_width_lines = [
                    lines[i] for i in range(len(lines)) if widths[i] == width
                ]
                if same_width_lines:
                    # Create a combined mask for all lines with this width
                    mask = np.zeros((height, width), dtype=bool)

                    # Draw all lines into the mask
                    for start, end in same_width_lines:
                        x1, y1 = start
                        x2, y2 = end

                        # Skip invalid lines
                        if not (
                            0 <= x1 < width
                            and 0 <= y1 < height
                            and 0 <= x2 < width
                            and 0 <= y2 < height
                        ):
                            continue

                        # Use Bresenham's algorithm to get line points
                        length = max(abs(x2 - x1), abs(y2 - y1))
                        if length == 0:
                            continue

                        t = np.linspace(0, 1, length * 2)
                        x_coords = np.round(x1 * (1 - t) + x2 * t).astype(int)
                        y_coords = np.round(y1 * (1 - t) + y2 * t).astype(int)

                        # Add line points to mask
                        for x, y in zip(x_coords, y_coords):
                            if width == 1:
                                mask[y, x] = True
                            else:
                                # For thicker lines
                                half_width = width // 2
                                min_y = max(0, y - half_width)
                                max_y = min(height, y + half_width + 1)
                                min_x = max(0, x - half_width)
                                max_x = min(width, x + half_width + 1)

                                # Create a circular brush
                                y_indices, x_indices = np.ogrid[
                                    min_y:max_y, min_x:max_x
                                ]
                                brush = (y_indices - y) ** 2 + (
                                    x_indices - x
                                ) ** 2 <= half_width**2
                                mask[min_y:max_y, min_x:max_x] |= brush

                    # Apply color to all lines at once
                    image[mask] = color

        return image

    @staticmethod
    async def async_draw_obstacles(
        image: np.ndarray, obstacle_info_list, color: Color
    ) -> np.ndarray:
        """
        Optimized async version of draw_obstacles using batch processing.
        Includes color blending for better visual integration.
        """
        if not obstacle_info_list:
            return image

        # Extract alpha from color
        alpha = color[3] if len(color) == 4 else 255
        need_blending = alpha < 255

        # Extract obstacle centers and prepare for batch processing
        centers = []
        for obs in obstacle_info_list:
            try:
                x = obs["points"]["x"]
                y = obs["points"]["y"]

                # Skip if coordinates are out of bounds
                if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                    continue

                # Apply color blending if needed
                obstacle_color = color
                if need_blending:
                    obstacle_color = ColorsManagement.sample_and_blend_color(
                        image, x, y, color
                    )

                # Add to centers list with radius
                centers.append({"center": (x, y), "radius": 6, "color": obstacle_color})
            except (KeyError, TypeError):
                continue

        # Draw each obstacle with its blended color
        if centers:
            for obstacle in centers:
                cx, cy = obstacle["center"]
                radius = obstacle["radius"]
                obs_color = obstacle["color"]

                # Create a small mask for the obstacle
                min_y = max(0, cy - radius)
                max_y = min(image.shape[0], cy + radius + 1)
                min_x = max(0, cx - radius)
                max_x = min(image.shape[1], cx + radius + 1)

                # Create coordinate arrays for the circle
                y_indices, x_indices = np.ogrid[min_y:max_y, min_x:max_x]

                # Create a circular mask
                mask = (y_indices - cy) ** 2 + (x_indices - cx) ** 2 <= radius**2

                # Apply the color to the masked region
                image[min_y:max_y, min_x:max_x][mask] = obs_color

        return image

    @staticmethod
    def status_text(
        image: PilPNG,
        size: int,
        color: Color,
        status: list[str],
        path_font: str,
        position: bool,
    ) -> None:
        """Draw the status text on the image."""
        path_default_font = (
            "custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf"
        )
        default_font = ImageFont.truetype(path_default_font, size)
        user_font = ImageFont.truetype(path_font, size)
        if position:
            x, y = 10, 10
        else:
            x, y = 10, image.height - 20 - size
        draw = ImageDraw.Draw(image)
        for text in status:
            if "\u2211" in text or "\u03de" in text:
                font = default_font
                width = None
            else:
                font = user_font
                width = 2 if path_font.endswith("VT.ttf") else None
            if width:
                draw.text((x, y), text, font=font, fill=color, stroke_width=width)
            else:
                draw.text((x, y), text, font=font, fill=color)
            x += draw.textlength(text, font=default_font)
