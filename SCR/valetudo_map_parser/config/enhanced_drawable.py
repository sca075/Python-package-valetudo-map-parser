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

from .colors import ColorsManagement
from .drawable import Drawable
from .drawable_elements import (
    DrawableElement,
    DrawingConfig,
)


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

    # Color blending methods have been moved to ColorsManagement class in colors.py

    # Pixel blending methods have been moved to ColorsManagement class in colors.py

    async def draw_map(
        self, map_data: dict, base_array: Optional[NumpyArray] = None
    ) -> NumpyArray:
        """
        Draw the map with selected elements.

        Args:
            map_data: The map data dictionary
            base_array: Optional base array to draw on

        Returns:
            The image array with all elements drawn
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

        # Draw elements in order of z-index
        for element in self.drawing_config.get_drawing_order():
            if element == DrawableElement.FLOOR:
                base_array = await self._draw_floor(map_data, base_array)
            elif element == DrawableElement.WALL:
                base_array = await self._draw_walls(map_data, base_array)
            elif element == DrawableElement.ROBOT:
                base_array = await self._draw_robot(map_data, base_array)
            elif element == DrawableElement.CHARGER:
                base_array = await self._draw_charger(map_data, base_array)
            elif element == DrawableElement.VIRTUAL_WALL:
                base_array = await self._draw_virtual_walls(map_data, base_array)
            elif element == DrawableElement.RESTRICTED_AREA:
                base_array = await self._draw_restricted_areas(map_data, base_array)
            elif element == DrawableElement.NO_MOP_AREA:
                base_array = await self._draw_no_mop_areas(map_data, base_array)
            elif element == DrawableElement.PATH:
                base_array = await self._draw_path(map_data, base_array)
            elif element == DrawableElement.PREDICTED_PATH:
                base_array = await self._draw_predicted_path(map_data, base_array)
            elif element == DrawableElement.GO_TO_TARGET:
                base_array = await self._draw_go_to_target(map_data, base_array)
            elif DrawableElement.ROOM_1 <= element <= DrawableElement.ROOM_15:
                room_id = element - DrawableElement.ROOM_1 + 1
                base_array = await self._draw_room(map_data, room_id, base_array)

        return base_array

    async def _draw_floor(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw the floor layer."""
        if not self.drawing_config.is_enabled(DrawableElement.FLOOR):
            return array

        # Implementation depends on the map data format
        # This is a placeholder - actual implementation would use map_data to draw floor

        return array

    async def _draw_walls(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw the walls."""
        if not self.drawing_config.is_enabled(DrawableElement.WALL):
            return array

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
            # Use sample_and_blend_color from ColorsManagement
            blended_color = ColorsManagement.sample_and_blend_color(
                array, x, y, wall_color
            )
            if 0 <= y < array.shape[0] and 0 <= x < array.shape[1]:
                array[y, x] = blended_color

        return array

    async def _draw_robot(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw the robot."""
        if not self.drawing_config.is_enabled(DrawableElement.ROBOT):
            return array

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
                        # Use sample_and_blend_color from ColorsManagement
                        blended_color = ColorsManagement.sample_and_blend_color(
                            array, map_x, map_y, robot_color
                        )
                        if 0 <= map_y < array.shape[0] and 0 <= map_x < array.shape[1]:
                            array[map_y, map_x] = blended_color
        return array

    async def _draw_charger(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw the charger."""
        if not self.drawing_config.is_enabled(DrawableElement.CHARGER):
            return array

        # Get charger color from drawing config
        charger_color = self.drawing_config.get_property(
            DrawableElement.CHARGER, "color", (255, 128, 0, 255)
        )

        # Implementation depends on the map data format
        # This would extract charger data from map_data and draw it

        return array

    async def _draw_virtual_walls(
        self, map_data: dict, array: NumpyArray
    ) -> NumpyArray:
        """Draw virtual walls."""
        if not self.drawing_config.is_enabled(DrawableElement.VIRTUAL_WALL):
            return array

        # Get virtual wall color from drawing config
        wall_color = self.drawing_config.get_property(
            DrawableElement.VIRTUAL_WALL, "color", (255, 0, 0, 255)
        )

        # Implementation depends on the map data format
        # This would extract virtual wall data from map_data and draw it

        return array

    async def _draw_restricted_areas(
        self, map_data: dict, array: NumpyArray
    ) -> NumpyArray:
        """Draw restricted areas."""
        if not self.drawing_config.is_enabled(DrawableElement.RESTRICTED_AREA):
            return array

        # Get restricted area color from drawing config
        area_color = self.drawing_config.get_property(
            DrawableElement.RESTRICTED_AREA, "color", (255, 0, 0, 125)
        )

        # Implementation depends on the map data format
        # This would extract restricted area data from map_data and draw it

        return array

    async def _draw_no_mop_areas(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw no-mop areas."""
        if not self.drawing_config.is_enabled(DrawableElement.NO_MOP_AREA):
            return array

        # Get no-mop area color from drawing config
        area_color = self.drawing_config.get_property(
            DrawableElement.NO_MOP_AREA, "color", (0, 0, 255, 125)
        )

        # Implementation depends on the map data format
        # This would extract no-mop area data from map_data and draw it

        return array

    async def _draw_path(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw the robot's path."""
        if not self.drawing_config.is_enabled(DrawableElement.PATH):
            return array

        # Get path color from drawing config
        path_color = self.drawing_config.get_property(
            DrawableElement.PATH, "color", (238, 247, 255, 255)
        )

        # Implementation depends on the map data format
        # This would extract path data from map_data and draw it

        return array

    async def _draw_predicted_path(
        self, map_data: dict, array: NumpyArray
    ) -> NumpyArray:
        """Draw the predicted path."""
        if not self.drawing_config.is_enabled(DrawableElement.PREDICTED_PATH):
            return array

        # Get predicted path color from drawing config
        path_color = self.drawing_config.get_property(
            DrawableElement.PREDICTED_PATH, "color", (238, 247, 255, 125)
        )

        # Implementation depends on the map data format
        # This would extract predicted path data from map_data and draw it

        return array

    async def _draw_go_to_target(self, map_data: dict, array: NumpyArray) -> NumpyArray:
        """Draw the go-to target."""
        if not self.drawing_config.is_enabled(DrawableElement.GO_TO_TARGET):
            return array

        # Get go-to target color from drawing config
        target_color = self.drawing_config.get_property(
            DrawableElement.GO_TO_TARGET, "color", (0, 255, 0, 255)
        )

        # Implementation depends on the map data format
        # This would extract go-to target data from map_data and draw it

        return array

    async def _draw_room(
        self, map_data: dict, room_id: int, array: NumpyArray
    ) -> NumpyArray:
        """Draw a specific room."""
        element = getattr(DrawableElement, f"ROOM_{room_id}")
        if not self.drawing_config.is_enabled(element):
            return array

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
            # Use sample_and_blend_color from ColorsManagement
            blended_color = ColorsManagement.sample_and_blend_color(
                array, x, y, room_color
            )
            if 0 <= y < array.shape[0] and 0 <= x < array.shape[1]:
                array[y, x] = blended_color

        return array
