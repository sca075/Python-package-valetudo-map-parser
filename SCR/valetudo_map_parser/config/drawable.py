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
from mvcrender.blend import get_blended_color, sample_and_blend_color
from mvcrender.draw import circle_u8, line_u8, polygon_u8
from PIL import Image, ImageDraw, ImageFont

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
        need_blending = color[3] < 255

        for x, y, z in pixels:
            col = x * pixel_size
            row = y * pixel_size
            for i in range(z):
                region_slice = (
                    slice(row, row + pixel_size),
                    slice(col + i * pixel_size, col + (i + 1) * pixel_size),
                )

                if need_blending:
                    cy = row + pixel_size // 2
                    cx = col + i * pixel_size + pixel_size // 2
                    if (
                        0 <= cy < image_array.shape[0]
                        and 0 <= cx < image_array.shape[1]
                    ):
                        px = sample_and_blend_color(image_array, cx, cy, color)
                        image_array[region_slice] = px
                    else:
                        image_array[region_slice] = color
                else:
                    image_array[region_slice] = color

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
            blended_color = sample_and_blend_color(layers, center_x, center_y, color)

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
        Uses mvcrender's polygon_u8 for efficient triangle drawing.
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
            flag_color = sample_and_blend_color(layer, x, y, flag_color)

        # Create pole color with alpha
        pole_color: Color = (
            pole_color_base[0],
            pole_color_base[1],
            pole_color_base[2],
            pole_alpha,
        )

        # Blend pole color if needed
        if pole_alpha < 255:
            pole_color = sample_and_blend_color(layer, x, y, pole_color)

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

        # Draw flag triangle using mvcrender's polygon_u8 (much faster than _polygon_outline)
        xs = np.array([x1, x2, x3], dtype=np.int32)
        ys = np.array([y1, y2, y3], dtype=np.int32)
        # Draw filled triangle with thin outline
        polygon_u8(layer, xs, ys, flag_color, 1, flag_color)

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
        layer: NumpyArray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Color,
        width: int = 3,
    ) -> NumpyArray:
        """Segment-aware preblend, then stamp a solid line."""
        width = int(max(1, width))
        # Preblend once for this segment
        seg = get_blended_color(int(x1), int(y1), int(x2), int(y2), layer, color)
        line_u8(layer, int(x1), int(y1), int(x2), int(y2), seg, width)
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

            # Use the optimized line drawing method
            arr = Drawable._line(arr, x0, y0, x1, y1, color, width)

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
        Draw a filled circle and optional outline using mvcrender.draw.circle_u8.
        If alpha<255, preblend once at the center and stamp solid.
        """
        cy, cx = (
            int(center[0]),
            int(center[1]),
        )  # incoming Point is (y,x) in your codebase
        h, w = image.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return image

        fill_rgba = color
        if fill_rgba[3] < 255:
            fill_rgba = sample_and_blend_color(image, cx, cy, fill_rgba)

        circle_u8(image, int(cx), int(cy), int(radius), fill_rgba, -1)

        if outline_color is not None and outline_width > 0:
            out_rgba = outline_color
            if out_rgba[3] < 255:
                out_rgba = sample_and_blend_color(image, cx, cy, out_rgba)
            # outlined stroke thickness = outline_width
            circle_u8(
                image, int(cx), int(cy), int(radius), out_rgba, int(outline_width)
            )

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
        Draw zones as filled polygons with alpha blending using mvcrender.
        Creates a mask with polygon_u8 and blends it onto the image with proper alpha.
        This eliminates PIL dependency for zone drawing.
        """
        if not coordinates:
            return layers

        height, width = layers.shape[:2]
        r, g, b, a = color
        alpha = a / 255.0
        inv_alpha = 1.0 - alpha
        # Pre-allocate color array once (avoid creating it in every iteration)
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

            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1

            # Create mask using mvcrender's polygon_u8
            mask_rgba = np.zeros((box_h, box_w, 4), dtype=np.uint8)

            # Convert points to xs, ys arrays (adjusted to local bbox coordinates)
            xs = np.array([int(pts[i] - min_x) for i in range(0, len(pts), 2)], dtype=np.int32)
            ys = np.array([int(pts[i] - min_y) for i in range(1, len(pts), 2)], dtype=np.int32)

            # Draw filled polygon on mask
            polygon_u8(mask_rgba, xs, ys, (0, 0, 0, 0), 0, (255, 255, 255, 255))

            # Extract boolean mask from first channel
            zone_mask = (mask_rgba[:, :, 0] > 0)
            del mask_rgba
            del xs
            del ys

            if not np.any(zone_mask):
                del zone_mask
                continue

            # Optimized alpha blend - minimize temporary allocations
            region = layers[min_y : max_y + 1, min_x : max_x + 1]

            # Work directly on the region's RGB channels
            rgb_region = region[..., :3]

            # Apply blending only where mask is True
            # Use boolean indexing to avoid creating full-size temporary arrays
            rgb_masked = rgb_region[zone_mask].astype(np.float32)

            # Blend: new_color = old_color * (1 - alpha) + zone_color * alpha
            rgb_masked *= inv_alpha
            rgb_masked += color_rgb * alpha

            # Write back (convert to uint8)
            rgb_region[zone_mask] = rgb_masked.astype(np.uint8)

            del zone_mask
            del rgb_masked

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
                    obs_color = sample_and_blend_color(image, x, y, color)
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
        # Load default font with safety fallback to PIL's built-in if missing
        try:
            default_font = ImageFont.truetype(str(default_font_path), size)
        except OSError:
            _LOGGER.warning(
                "Default font not found at %s; using PIL default font",
                default_font_path,
            )
            default_font = ImageFont.load_default()

        # Use provided font directly if available; else fall back to default
        user_font = default_font
        if path_font:
            try:
                user_font = ImageFont.truetype(str(path_font), size)
            except OSError:
                user_font = default_font
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
