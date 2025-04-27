"""
Collections of Drawing Utility
Drawable is part of the Image_Handler
used functions to draw the elements on the Numpy Array
that is actually our camera frame.
Version: v2024.12.0
Refactored for clarity, consistency, and optimized parameter usage.
"""

from __future__ import annotations

import asyncio
import logging
import math

# cv2 is imported but not used directly in this file
# It's needed for other modules that import from here
import numpy as np
from PIL import ImageDraw, ImageFont

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

        # For debugging
        _LOGGER.debug("Drawing with color %s and alpha %s", color, alpha)

        # Create the full color with alpha
        full_color = color if len(color) == 4 else (*color, 255)

        # Loop through pixels to find min and max coordinates
        for x, y, z in pixels:
            col = x * pixel_size
            row = y * pixel_size
            # Draw pixels as blocks
            for i in range(z):
                # Get the region to update
                region = image_array[
                    row : row + pixel_size,
                    col + i * pixel_size : col + (i + 1) * pixel_size,
                ]

                # Simple direct assignment - ignore alpha for now to ensure visibility
                region[:] = full_color

        return image_array

    @staticmethod
    async def battery_charger(
        layers: NumpyArray, x: int, y: int, color: Color
    ) -> NumpyArray:
        """Draw the battery charger on the input layer."""
        charger_width = 10
        charger_height = 20
        start_row = y - charger_height // 2
        end_row = start_row + charger_height
        start_col = x - charger_width // 2
        end_col = start_col + charger_width
        layers[start_row:end_row, start_col:end_col] = color
        return layers

    @staticmethod
    async def go_to_flag(
        layer: NumpyArray, center: Point, rotation_angle: int, flag_color: Color
    ) -> NumpyArray:
        """
        Draw a flag centered at specified coordinates on the input layer.
        It uses the rotation angle of the image to orient the flag.
        """
        pole_color: Color = (0, 0, 255, 255)  # Blue for the pole
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
        Draw a line on a NumPy array (layer) from point A to B using Bresenham's algorithm.
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
            # Draw a rectangle at the current coordinates with the specified width
            for i in range(-width // 2, (width + 1) // 2):
                for j in range(-width // 2, (width + 1) // 2):
                    if 0 <= x1 + i < layer.shape[1] and 0 <= y1 + j < layer.shape[0]:
                        layer[y1 + j, x1 + i] = color
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
    async def lines(arr: NumpyArray, coords, width: int, color: Color) -> NumpyArray:
        """
        Join the coordinates creating a continuous line (path).
        """
        for coord in coords:
            x0, y0 = coord[0]
            try:
                x1, y1 = coord[1]
            except IndexError:
                x1, y1 = x0, y0
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            line_pixels: list[Tuple[int, int]] = []
            while True:
                line_pixels.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            # Draw filled rectangles for each pixel in the line
            for pixel in line_pixels:
                x, y = pixel
                for i in range(width):
                    for j in range(width):
                        if 0 <= x + i < arr.shape[1] and 0 <= y + j < arr.shape[0]:
                            arr[y + j, x + i] = color
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
        """
        y, x = center
        rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
        circle = (rr - x) ** 2 + (cc - y) ** 2 <= radius**2
        image[circle] = color
        if outline_width > 0:
            outer_circle = (rr - x) ** 2 + (cc - y) ** 2 <= (
                radius + outline_width
            ) ** 2
            outline_mask = outer_circle & ~circle
            image[outline_mask] = outline_color
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
        """
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
            if fill_color is not None:
                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_y = max(p[1] for p in points)
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if Drawable.point_inside(x, y, points):
                            arr[y, x] = fill_color
        return arr

    @staticmethod
    async def zones(layers: NumpyArray, coordinates, color: Color) -> NumpyArray:
        """
        Draw the zones on the input layer.
        """
        dot_radius = 1  # Number of pixels for the dot
        dot_spacing = 4  # Space between dots
        for zone in coordinates:
            points = zone["points"]
            min_x = min(points[::2])
            max_x = max(points[::2])
            min_y = min(points[1::2])
            max_y = max(points[1::2])
            for y in range(min_y, max_y, dot_spacing):
                for x in range(min_x, max_x, dot_spacing):
                    for _ in range(dot_radius):
                        layers = Drawable._ellipse(layers, (x, y), dot_radius, color)
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
        """
        top_left_x = x - 26
        top_left_y = y - 26
        bottom_right_x = top_left_x + 52
        bottom_right_y = top_left_y + 52
        tmp_layer = layers[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()
        tmp_x, tmp_y = 26, 26
        radius = 25
        r_scaled = radius // 11
        r_cover = r_scaled * 12
        lidar_angle = np.deg2rad(angle + 90)
        r_lidar = r_scaled * 3
        r_button = r_scaled * 1
        if robot_state == "error":
            outline = Drawable.ERROR_OUTLINE
            fill = Drawable.ERROR_COLOR
        else:
            outline = (fill[0] // 2, fill[1] // 2, fill[2] // 2, fill[3])
        tmp_layer = Drawable._filled_circle(
            tmp_layer, (tmp_x, tmp_y), radius, fill, outline, 1
        )
        angle -= 90
        a1 = ((angle + 90) - 80) / 180 * math.pi
        a2 = ((angle + 90) + 80) / 180 * math.pi
        x1 = int(tmp_x - r_cover * math.sin(a1))
        y1 = int(tmp_y + r_cover * math.cos(a1))
        x2 = int(tmp_x - r_cover * math.sin(a2))
        y2 = int(tmp_y + r_cover * math.cos(a2))
        tmp_layer = Drawable._line(tmp_layer, x1, y1, x2, y2, outline, width=1)
        lidar_x = int(tmp_x + 15 * np.cos(lidar_angle))
        lidar_y = int(tmp_y + 15 * np.sin(lidar_angle))
        tmp_layer = Drawable._filled_circle(
            tmp_layer, (lidar_x, lidar_y), r_lidar, outline
        )
        butt_x = int(tmp_x - 20 * np.cos(lidar_angle))
        butt_y = int(tmp_y - 20 * np.sin(lidar_angle))
        tmp_layer = Drawable._filled_circle(
            tmp_layer, (butt_x, butt_y), r_button, outline
        )
        layers = Drawable.overlay_robot(layers, tmp_layer, x, y)
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
    async def async_draw_obstacles(
        image: np.ndarray, obstacle_info_list, color: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Optimized async version of draw_obstacles using asyncio.gather().
        """

        def extract_centers(obs_list) -> np.ndarray:
            return np.array(
                [[obs["points"]["x"], obs["points"]["y"]] for obs in obs_list],
                dtype=np.int32,
            )

        centers = await asyncio.get_running_loop().run_in_executor(
            None, extract_centers, obstacle_info_list
        )
        Drawable.draw_filled_circle(image, centers, 6, color)
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
