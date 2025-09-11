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
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
        Background color is specified as an RGBA tuple."""
        return np.full((height, width, 4), background_color, dtype=np.uint8)

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
        # Check if coordinates are within bounds
        height, width = layer.shape[:2]
        x, y = center
        if not (0 <= x < width and 0 <= y < height):
            return layer

        # Get blended colors for flag and pole
        flag_alpha = flag_color[3] if len(flag_color) == 4 else 255
        pole_color_base = [0, 0, 255]  # Blue for the pole
        pole_alpha = 255

        # Blend flag color if needed
        if flag_alpha < 255:
            flag_color = ColorsManagement.sample_and_blend_color(
                layer, x, y, flag_color
            )

        # Create pole color with alpha
        pole_color: Color = (
            pole_color_base[0],
            pole_color_base[1],
            pole_color_base[2],
            pole_alpha,
        )

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
        """Check if a point (x, y) is inside a polygon defined by a list of points."""
        n = len(points)
        inside = False
        inters_x = 0.0
        p1x, p1y = points[0]
        for i in range(1, n + 1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        inters_x = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= inters_x:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def _line(
        layer: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Color,
        width: int = 3,
    ) -> np.ndarray:
        """Draw a line on a NumPy array (layer) from point A to B using Bresenham's algorithm.

        Args:
            layer: The numpy array to draw on (H, W, C)
            x1, y1: Start point coordinates
            x2, y2: End point coordinates
            color: Color to draw with (tuple or array)
            width: Width of the line in pixels
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        blended_color = get_blended_color(x1, y1, x2, y2, layer, color)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        half_w = width // 2
        h, w = layer.shape[:2]

        while True:
            # Draw a filled circle for thickness
            yy, xx = np.ogrid[-half_w : half_w + 1, -half_w : half_w + 1]
            mask = xx**2 + yy**2 <= half_w**2
            y_min = max(0, y1 - half_w)
            y_max = min(h, y1 + half_w + 1)
            x_min = max(0, x1 - half_w)
            x_max = min(w, x1 + half_w + 1)

            sub_mask = mask[
                (y_min - (y1 - half_w)) : (y_max - (y1 - half_w)),
                (x_min - (x1 - half_w)) : (x_max - (x1 - half_w)),
            ]
            layer[y_min:y_max, x_min:x_max][sub_mask] = blended_color

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

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
    async def lines(
        arr: NumpyArray, coordinates, width: int, color: Color
    ) -> NumpyArray:
        """
        Join the coordinates creating a continuous line (path).
        Optimized with vectorized operations for better performance.
        """
        for coord in coordinates:
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
        Draw zones as solid filled polygons with alpha blending using a per-zone mask.
        Keeps API the same; no dotted rendering.
        """
        if not coordinates:
            return layers

        height, width = layers.shape[:2]
        # Precompute color and alpha
        r, g, b, a = color
        alpha = a / 255.0
        inv_alpha = 1.0 - alpha
        color_rgb = np.array([r, g, b], dtype=np.float32)

        for zone in coordinates:
            try:
                pts = zone["points"]
            except (KeyError, TypeError):
                continue
            if not pts or len(pts) < 6:
                continue

            # Compute bounding box and clamp
            min_x = max(0, int(min(pts[::2])))
            max_x = min(width - 1, int(max(pts[::2])))
            min_y = max(0, int(min(pts[1::2])))
            max_y = min(height - 1, int(max(pts[1::2])))
            if min_x >= max_x or min_y >= max_y:
                continue

            # Adjust polygon points to local bbox coordinates
            poly_xy = [
                (int(pts[i] - min_x), int(pts[i + 1] - min_y))
                for i in range(0, len(pts), 2)
            ]
            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1

            # Build mask via PIL polygon fill (fast, C-impl)
            mask_img = Image.new("L", (box_w, box_h), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(poly_xy, fill=255)
            zone_mask = np.array(mask_img, dtype=bool)
            if not np.any(zone_mask):
                continue

            # Vectorized alpha blend on RGB channels only
            region = layers[min_y : max_y + 1, min_x : max_x + 1]
            rgb = region[..., :3].astype(np.float32)
            mask3 = zone_mask[:, :, None]
            blended_rgb = np.where(mask3, rgb * inv_alpha + color_rgb * alpha, rgb)
            region[..., :3] = blended_rgb.astype(np.uint8)
            # Leave alpha channel unchanged to avoid stacking transparency

        return layers

    @staticmethod
    async def robot(
        layers: NumpyArray,
        x: int,
        y: int,
        angle: float,
        fill: Color,
        robot_state: str | None = None,
        radius: int = 25,  # user-configurable
    ) -> NumpyArray:
        """
        Draw the robot with configurable size. All elements scale with radius.
        """
        # Minimum radius to keep things visible
        radius = max(8, min(radius, 25))

        height, width = layers.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return layers

        # Bounding box
        top_left_x = max(0, x - radius - 1)
        top_left_y = max(0, y - radius - 1)
        bottom_right_x = min(width, x + radius + 1)
        bottom_right_y = min(height, y + radius + 1)

        if top_left_x >= bottom_right_x or top_left_y >= bottom_right_y:
            return layers

        tmp_width = bottom_right_x - top_left_x
        tmp_height = bottom_right_y - top_left_y
        tmp_layer = layers[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

        tmp_x = x - top_left_x
        tmp_y = y - top_left_y

        # All geometry proportional to radius
        r_scaled: float = max(1.0, radius / 11.0)
        r_cover = int(r_scaled * 10)
        r_lidar = max(1, int(r_scaled * 3))
        r_button = max(1, int(r_scaled * 1))
        lidar_offset = int(radius * 0.6)  # was fixed 15
        button_offset = int(radius * 0.8)  # was fixed 20

        lidar_angle = np.deg2rad(angle + 90)

        if robot_state == "error":
            outline = Drawable.ERROR_OUTLINE
            fill = Drawable.ERROR_COLOR
        else:
            outline = (fill[0] // 2, fill[1] // 2, fill[2] // 2, fill[3])

        # Body
        tmp_layer = Drawable._filled_circle(
            tmp_layer, (tmp_y, tmp_x), radius, fill, outline, 1
        )

        # Direction wedge
        angle -= 90
        a1 = np.deg2rad((angle + 90) - 80)
        a2 = np.deg2rad((angle + 90) + 80)
        x1 = int(tmp_x - r_cover * np.sin(a1))
        y1 = int(tmp_y + r_cover * np.cos(a1))
        x2 = int(tmp_x - r_cover * np.sin(a2))
        y2 = int(tmp_y + r_cover * np.cos(a2))
        if (
            0 <= x1 < tmp_width
            and 0 <= y1 < tmp_height
            and 0 <= x2 < tmp_width
            and 0 <= y2 < tmp_height
        ):
            tmp_layer = Drawable._line(tmp_layer, x1, y1, x2, y2, outline, width=1)

        # Lidar
        lidar_x = int(tmp_x + lidar_offset * np.cos(lidar_angle))
        lidar_y = int(tmp_y + lidar_offset * np.sin(lidar_angle))
        if 0 <= lidar_x < tmp_width and 0 <= lidar_y < tmp_height:
            tmp_layer = Drawable._filled_circle(
                tmp_layer, (lidar_y, lidar_x), r_lidar, outline
            )

        # Button
        butt_x = int(tmp_x - button_offset * np.cos(lidar_angle))
        butt_y = int(tmp_y - button_offset * np.sin(lidar_angle))
        if 0 <= butt_x < tmp_width and 0 <= butt_y < tmp_height:
            tmp_layer = Drawable._filled_circle(
                tmp_layer, (butt_y, butt_x), r_button, outline
            )

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
                        x_coordinates = np.round(x1 * (1 - t) + x2 * t).astype(int)
                        y_coordinates = np.round(y1 * (1 - t) + y2 * t).astype(int)

                        # Add line points to mask
                        for x, y in zip(x_coordinates, y_coordinates):
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
        Optimized async version of draw_obstacles using a precomputed mask
        and minimal Python overhead. Handles hundreds of obstacles efficiently.
        """
        if not obstacle_info_list:
            return image

        h, w = image.shape[:2]
        alpha = color[3] if len(color) == 4 else 255
        need_blending = alpha < 255

        # Precompute circular mask for radius
        radius = 6
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        circle_mask = (xx**2 + yy**2) <= radius**2

        # Collect valid obstacles
        centers = []
        for obs in obstacle_info_list:
            try:
                x = obs["points"]["x"]
                y = obs["points"]["y"]

                if not (0 <= x < w and 0 <= y < h):
                    continue

                if need_blending:
                    obs_color = ColorsManagement.sample_and_blend_color(
                        image, x, y, color
                    )
                else:
                    obs_color = color

                centers.append((x, y, obs_color))
            except (KeyError, TypeError):
                continue

        # Draw all obstacles
        for cx, cy, obs_color in centers:
            min_y = max(0, cy - radius)
            max_y = min(h, cy + radius + 1)
            min_x = max(0, cx - radius)
            max_x = min(w, cx + radius + 1)

            # Slice mask to fit image edges
            mask_y_start = min_y - (cy - radius)
            mask_y_end = mask_y_start + (max_y - min_y)
            mask_x_start = min_x - (cx - radius)
            mask_x_end = mask_x_start + (max_x - min_x)

            mask = circle_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

            # Apply color in one vectorized step
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
        module_dir = Path(__file__).resolve().parent
        default_font_path = module_dir / "fonts" / "FiraSans.ttf"
        default_font = ImageFont.truetype(str(default_font_path), size)

        user_font_path = Path(path_font)
        if not user_font_path.is_absolute():
            repo_root = module_dir.parents[2]
            user_font_path = (repo_root / user_font_path).resolve()
        user_font = ImageFont.truetype(str(user_font_path), size)
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
