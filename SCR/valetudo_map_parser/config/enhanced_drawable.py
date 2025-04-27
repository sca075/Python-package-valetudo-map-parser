"""
Enhanced Drawable Class.
Provides drawing utilities with element selection support.
Version: 0.1.9
"""

from __future__ import annotations

import logging

# math is not used in this file
from typing import Optional, Tuple

import numpy as np

from .drawable import Drawable
from .drawable_elements import DrawableElement, DrawingConfig


# Type aliases
NumpyArray = np.ndarray
Color = Tuple[int, int, int, int]

_LOGGER = logging.getLogger(__name__)


class EnhancedDrawable(Drawable):
    """Enhanced drawing utilities with element selection support."""

    def __init__(self, drawing_config: Optional[DrawingConfig] = None):
        """Initialize with optional drawing configuration."""
        super().__init__()
        self.drawing_config = drawing_config or DrawingConfig()

    @staticmethod
    def blend_colors(base_color: Color, overlay_color: Color) -> Color:
        """
        Blend two RGBA colors, considering alpha channels.

        Args:
            base_color: The base RGBA color
            overlay_color: The overlay RGBA color to blend on top

        Returns:
            The blended RGBA color
        """
        # Extract components
        r1, g1, b1, a1 = base_color
        r2, g2, b2, a2 = overlay_color

        # Convert alpha to 0-1 range
        a1 = a1 / 255.0
        a2 = a2 / 255.0

        # Calculate resulting alpha
        a_out = a1 + a2 * (1 - a1)

        # Avoid division by zero
        if a_out < 0.0001:
            return (0, 0, 0, 0)

        # Calculate blended RGB components
        r_out = (r1 * a1 + r2 * a2 * (1 - a1)) / a_out
        g_out = (g1 * a1 + g2 * a2 * (1 - a1)) / a_out
        b_out = (b1 * a1 + b2 * a2 * (1 - a1)) / a_out

        # Convert back to 0-255 range and return as tuple
        return (
            int(max(0, min(255, r_out))),
            int(max(0, min(255, g_out))),
            int(max(0, min(255, b_out))),
            int(max(0, min(255, a_out * 255))),
        )

    def blend_pixel(
        self,
        array: NumpyArray,
        x: int,
        y: int,
        color: Color,
        element: DrawableElement,
        element_map: NumpyArray,
    ) -> None:
        """
        Blend a pixel color with the existing color at the specified position.
        Also updates the element map if the new element has higher z-index.

        Args:
            array: The image array to modify
            x, y: Pixel coordinates
            color: RGBA color to blend
            element: The element being drawn
            element_map: The element map to update
        """
        # Check bounds
        if not (0 <= y < array.shape[0] and 0 <= x < array.shape[1]):
            return

        # Get current element at this position
        current_element = element_map[y, x]

        # Get z-indices for comparison
        current_z = (
            self.drawing_config.get_property(current_element, "z_index", 0)
            if current_element
            else 0
        )
        new_z = self.drawing_config.get_property(element, "z_index", 0)

        # Get current color at this position
        current_color = tuple(array[y, x])

        # Blend colors
        blended_color = self.blend_colors(current_color, color)

        # Update pixel color
        array[y, x] = blended_color

        # Update element map if new element has higher z-index
        if new_z >= current_z:
            element_map[y, x] = element

    async def draw_map(
        self, map_data: dict, base_array: Optional[NumpyArray] = None
    ) -> Tuple[NumpyArray, NumpyArray]:
        """
        Draw the map with selected elements.

        Args:
            map_data: The map data dictionary
            base_array: Optional base array to draw on

        Returns:
            Tuple of (image_array, element_map)
        """
        # Get map dimensions
        size_x = map_data.get("size", {}).get("x", 1024)
        size_y = map_data.get("size", {}).get("y", 1024)

        # Create empty image if none provided
        if base_array is None:
            background_color = self.drawing_config.get_property(
                DrawableElement.FLOOR, "color", (200, 200, 200, 255)
            )
            base_array = await self.create_empty_image(size_x, size_y, background_color)

        # Create a 2D map for element identification
        element_map = np.zeros(
            (base_array.shape[0], base_array.shape[1]), dtype=np.int32
        )

        # Draw elements in order of z-index
        for element in self.drawing_config.get_drawing_order():
            if element == DrawableElement.FLOOR:
                base_array, element_map = await self._draw_floor(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.WALL:
                base_array, element_map = await self._draw_walls(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.ROBOT:
                base_array, element_map = await self._draw_robot(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.CHARGER:
                base_array, element_map = await self._draw_charger(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.VIRTUAL_WALL:
                base_array, element_map = await self._draw_virtual_walls(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.RESTRICTED_AREA:
                base_array, element_map = await self._draw_restricted_areas(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.NO_MOP_AREA:
                base_array, element_map = await self._draw_no_mop_areas(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.PATH:
                base_array, element_map = await self._draw_path(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.PREDICTED_PATH:
                base_array, element_map = await self._draw_predicted_path(
                    map_data, base_array, element_map
                )
            elif element == DrawableElement.GO_TO_TARGET:
                base_array, element_map = await self._draw_go_to_target(
                    map_data, base_array, element_map
                )
            elif DrawableElement.ROOM_1 <= element <= DrawableElement.ROOM_15:
                room_id = element - DrawableElement.ROOM_1 + 1
                base_array, element_map = await self._draw_room(
                    map_data, room_id, base_array, element_map
                )

        return base_array, element_map

    async def _draw_floor(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the floor layer."""
        if not self.drawing_config.is_enabled(DrawableElement.FLOOR):
            return array, element_map

        # Implementation depends on the map data format
        # This is a placeholder - actual implementation would use map_data to draw floor
        # For now, we'll just mark the entire map as floor in the element map
        element_map[:] = DrawableElement.FLOOR

        return array, element_map

    async def _draw_walls(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the walls."""
        if not self.drawing_config.is_enabled(DrawableElement.WALL):
            return array, element_map

        # Get wall color from drawing config
        wall_color = self.drawing_config.get_property(
            DrawableElement.WALL, "color", (255, 255, 0, 255)
        )

        # Implementation depends on the map data format
        # For Valetudo maps, we would look at the layers with type "wall"
        # This is a simplified example - in a real implementation, we would extract the actual wall pixels

        # Find wall data in map_data
        wall_pixels = []
        for layer in map_data.get("layers", []):
            if layer.get("type") == "wall":
                # Extract wall pixels from the layer
                # This is a placeholder - actual implementation would depend on the map data format
                wall_pixels = layer.get("pixels", [])
                break

        # Draw wall pixels with color blending
        for x, y in wall_pixels:
            # Use blend_pixel to properly blend colors and update element map
            self.blend_pixel(array, x, y, wall_color, DrawableElement.WALL, element_map)

        return array, element_map

    async def _draw_robot(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the robot."""
        if not self.drawing_config.is_enabled(DrawableElement.ROBOT):
            return array, element_map

        # Get robot color from drawing config
        robot_color = self.drawing_config.get_property(
            DrawableElement.ROBOT, "color", (255, 255, 204, 255)
        )

        # Extract robot position and angle from map_data
        robot_position = map_data.get("robot", {}).get("position", None)
        robot_angle = map_data.get("robot", {}).get("angle", 0)

        if robot_position:
            x, y = robot_position.get("x", 0), robot_position.get("y", 0)

            # Draw robot with color blending
            # Create a circle around the robot position
            radius = 25  # Same as in the robot drawing method
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        map_x, map_y = int(x + dx), int(y + dy)
                        # Use blend_pixel to properly blend colors and update element map
                        self.blend_pixel(
                            array,
                            map_x,
                            map_y,
                            robot_color,
                            DrawableElement.ROBOT,
                            element_map,
                        )

            # TODO: Draw robot orientation indicator
            # This would be a line or triangle showing the direction
            # For now, we'll skip this part as it requires more complex drawing

        return array, element_map

    async def _draw_charger(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the charger."""
        if not self.drawing_config.is_enabled(DrawableElement.CHARGER):
            return array, element_map

        # Get charger color from drawing config
        charger_color = self.drawing_config.get_property(
            DrawableElement.CHARGER, "color", (255, 128, 0, 255)
        )

        # Implementation depends on the map data format
        # This would extract charger data from map_data and draw it

        return array, element_map

    async def _draw_virtual_walls(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw virtual walls."""
        if not self.drawing_config.is_enabled(DrawableElement.VIRTUAL_WALL):
            return array, element_map

        # Get virtual wall color from drawing config
        wall_color = self.drawing_config.get_property(
            DrawableElement.VIRTUAL_WALL, "color", (255, 0, 0, 255)
        )

        # Implementation depends on the map data format
        # This would extract virtual wall data from map_data and draw it

        return array, element_map

    async def _draw_restricted_areas(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw restricted areas."""
        if not self.drawing_config.is_enabled(DrawableElement.RESTRICTED_AREA):
            return array, element_map

        # Get restricted area color from drawing config
        area_color = self.drawing_config.get_property(
            DrawableElement.RESTRICTED_AREA, "color", (255, 0, 0, 125)
        )

        # Implementation depends on the map data format
        # This would extract restricted area data from map_data and draw it

        return array, element_map

    async def _draw_no_mop_areas(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw no-mop areas."""
        if not self.drawing_config.is_enabled(DrawableElement.NO_MOP_AREA):
            return array, element_map

        # Get no-mop area color from drawing config
        area_color = self.drawing_config.get_property(
            DrawableElement.NO_MOP_AREA, "color", (0, 0, 255, 125)
        )

        # Implementation depends on the map data format
        # This would extract no-mop area data from map_data and draw it

        return array, element_map

    async def _draw_path(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the robot's path."""
        if not self.drawing_config.is_enabled(DrawableElement.PATH):
            return array, element_map

        # Get path color from drawing config
        path_color = self.drawing_config.get_property(
            DrawableElement.PATH, "color", (238, 247, 255, 255)
        )

        # Implementation depends on the map data format
        # This would extract path data from map_data and draw it

        return array, element_map

    async def _draw_predicted_path(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the predicted path."""
        if not self.drawing_config.is_enabled(DrawableElement.PREDICTED_PATH):
            return array, element_map

        # Get predicted path color from drawing config
        path_color = self.drawing_config.get_property(
            DrawableElement.PREDICTED_PATH, "color", (238, 247, 255, 125)
        )

        # Implementation depends on the map data format
        # This would extract predicted path data from map_data and draw it

        return array, element_map

    async def _draw_go_to_target(
        self, map_data: dict, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw the go-to target."""
        if not self.drawing_config.is_enabled(DrawableElement.GO_TO_TARGET):
            return array, element_map

        # Get go-to target color from drawing config
        target_color = self.drawing_config.get_property(
            DrawableElement.GO_TO_TARGET, "color", (0, 255, 0, 255)
        )

        # Implementation depends on the map data format
        # This would extract go-to target data from map_data and draw it

        return array, element_map

    async def _draw_room(
        self, map_data: dict, room_id: int, array: NumpyArray, element_map: NumpyArray
    ) -> Tuple[NumpyArray, NumpyArray]:
        """Draw a specific room."""
        element = getattr(DrawableElement, f"ROOM_{room_id}")
        if not self.drawing_config.is_enabled(element):
            return array, element_map

        # Get room color from drawing config
        room_color = self.drawing_config.get_property(
            element,
            "color",
            (135, 206, 250, 255),  # Default light blue
        )

        # Implementation depends on the map data format
        # For Valetudo maps, we would look at the layers with type "segment"
        # This is a simplified example - in a real implementation, we would extract the actual room pixels

        # Find room data in map_data
        room_pixels = []
        for layer in map_data.get("layers", []):
            if layer.get("type") == "segment" and str(
                layer.get("metaData", {}).get("segmentId")
            ) == str(room_id):
                # Extract room pixels from the layer
                # This is a placeholder - actual implementation would depend on the map data format
                # For example, it might use compressed pixels or other data structures

                # For demonstration, let's assume we have a list of (x, y) coordinates
                room_pixels = layer.get("pixels", [])
                break

        # Draw room pixels with color blending
        for x, y in room_pixels:
            # Use blend_pixel to properly blend colors and update element map
            self.blend_pixel(array, x, y, room_color, element, element_map)

        return array, element_map
