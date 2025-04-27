"""
Image Handler Module for Valetudo Re Vacuums.
It returns the PIL PNG image frame relative to the Map Data extrapolated from the vacuum json.
It also returns calibration, rooms data to the card and other images information to the camera.
Version: 0.1.9
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from PIL import Image

from .config.auto_crop import AutoCrop
from .config.drawable import Drawable
from .config.drawable_elements import DrawableElement, DrawingConfig
from .config.enhanced_drawable import EnhancedDrawable
from .config.types import (
    COLORS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PIXEL_SIZE,
    Colors,
    JsonType,
    PilPNG,
    RobotPosition,
    RoomsProperties,
    RoomStore,
)
from .config.utils import BaseHandler, prepare_resize_params
from .map_data import RandImageData
from .reimg_draw import ImageDraw


_LOGGER = logging.getLogger(__name__)


# noinspection PyTypeChecker
class ReImageHandler(BaseHandler, AutoCrop):
    """
    Image Handler for Valetudo Re Vacuums.
    """

    def __init__(self, shared_data):
        BaseHandler.__init__(self)
        self.shared = shared_data  # Shared data
        AutoCrop.__init__(self, self)
        self.auto_crop = None  # Auto crop flag
        self.segment_data = None  # Segment data
        self.outlines = None  # Outlines data
        self.calibration_data = None  # Calibration data
        self.data = RandImageData  # Image Data

        # Initialize drawing configuration using the shared utility function
        from .config.utils import initialize_drawing_config
        self.drawing_config, self.draw, self.enhanced_draw = initialize_drawing_config(self)
        self.element_map = None  # Map of element codes
        self.go_to = None  # Go to position data
        self.img_base_layer = None  # Base image layer
        self.img_rotate = shared_data.image_rotate  # Image rotation
        self.room_propriety = None  # Room propriety data
        self.active_zones = None  # Active zones
        self.file_name = self.shared.file_name  # File name
        self.imd = ImageDraw(self)  # Image Draw

    async def extract_room_outline_from_map(self, room_id_int, pixels):
        """Extract the outline of a room using the pixel data and element map.

        Args:
            room_id_int: The room ID as an integer
            pixels: List of pixel coordinates in the format [[x, y, z], ...]

        Returns:
            List of points forming the outline of the room
        """
        # Calculate x and y min/max from compressed pixels for rectangular fallback
        x_values = []
        y_values = []
        for x, y, _ in pixels:
            x_values.append(x)
            y_values.append(y)

        if not x_values or not y_values:
            return []

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        # If we don't have an element map, return a rectangular outline
        if not hasattr(self, "element_map") or self.element_map is None:
            # Return rectangular outline
            return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        # Create a binary mask for this room using the pixel data
        # This is more reliable than using the element_map since we're directly using the pixel data
        height, width = self.element_map.shape
        room_mask = np.zeros((height, width), dtype=np.uint8)

        # Fill the mask with room pixels using the pixel data
        for x, y, _ in pixels:  # Using _ instead of z since z is unused
            # Make sure we're within bounds
            if 0 <= y < height and 0 <= x < width:
                # Mark a pixel at this position
                room_mask[y, x] = 1

        # Debug log to check if we have any room pixels
        num_room_pixels = np.sum(room_mask)
        _LOGGER.debug(
            "%s: Room %s mask has %d pixels",
            self.file_name, str(room_id_int), int(num_room_pixels)
        )

        # Use the shared utility function to extract the room outline
        from .config.utils import async_extract_room_outline
        return await async_extract_room_outline(
            room_mask, min_x, min_y, max_x, max_y, self.file_name, room_id_int, _LOGGER
        )

    async def extract_room_properties(
        self, json_data: JsonType, destinations: JsonType
    ) -> RoomsProperties:
        """Extract the room properties."""
        unsorted_id = RandImageData.get_rrm_segments_ids(json_data)
        size_x, size_y = RandImageData.get_rrm_image_size(json_data)
        top, left = RandImageData.get_rrm_image_position(json_data)
        try:
            if not self.segment_data or not self.outlines:
                (
                    self.segment_data,
                    self.outlines,
                ) = await RandImageData.async_get_rrm_segments(
                    json_data, size_x, size_y, top, left, True
                )
            dest_json = destinations
            room_data = dict(dest_json).get("rooms", [])
            zones_data = dict(dest_json).get("zones", [])
            points_data = dict(dest_json).get("spots", [])
            room_id_to_data = {room["id"]: room for room in room_data}
            self.rooms_pos = []
            room_properties = {}
            if self.outlines:
                for id_x, room_id in enumerate(unsorted_id):
                    if room_id in room_id_to_data:
                        room_info = room_id_to_data[room_id]
                        name = room_info.get("name")
                        # Calculate x and y min/max from outlines
                        x_min = self.outlines[id_x][0][0]
                        x_max = self.outlines[id_x][1][0]
                        y_min = self.outlines[id_x][0][1]
                        y_max = self.outlines[id_x][1][1]

                        # Get rectangular corners as a fallback
                        corners = self.get_corners(x_max, x_min, y_max, y_min)

                        # Try to extract a more accurate room outline from the element map
                        try:
                            # Extract the room outline using the element map
                            outline = await self.extract_room_outline_from_map(
                                room_id, self.segment_data[id_x]
                            )
                            _LOGGER.debug(
                                "%s: Traced outline for room %s with %d points",
                                self.file_name,
                                room_id,
                                len(outline),
                            )
                        except (ValueError, IndexError, TypeError, ArithmeticError) as e:
                            from .config.utils import handle_room_outline_error
                            handle_room_outline_error(self.file_name, room_id, e, _LOGGER)
                            outline = corners

                        # rand256 vacuums accept int(room_id) or str(name)
                        # the card will soon support int(room_id) but the camera will send name
                        # this avoids the manual change of the values in the card.
                        self.rooms_pos.append(
                            {
                                "name": name,
                                "corners": corners,
                            }
                        )
                        room_properties[int(room_id)] = {
                            "number": int(room_id),
                            "outline": outline,
                            "name": name,
                            "x": (x_min + x_max) // 2,
                            "y": (y_min + y_max) // 2,
                        }
                # get the zones and points data
                zone_properties = await self.async_zone_propriety(zones_data)
                # get the points data
                point_properties = await self.async_points_propriety(points_data)
                if room_properties or zone_properties:
                    extracted_data = [
                        f"{len(room_properties)} Rooms" if room_properties else None,
                        f"{len(zone_properties)} Zones" if zone_properties else None,
                    ]
                    extracted_data = ", ".join(filter(None, extracted_data))
                    _LOGGER.debug("Extracted data: %s", extracted_data)
                else:
                    self.rooms_pos = None
                    _LOGGER.debug(
                        "%s: Rooms and Zones data not available!", self.file_name
                    )
                rooms = RoomStore(self.file_name, room_properties)
                _LOGGER.debug("Rooms Data: %s", rooms.get_rooms())
                return room_properties, zone_properties, point_properties
        except (RuntimeError, ValueError) as e:
            _LOGGER.debug(
                "No rooms Data or Error in extract_room_properties: %s",
                e,
                exc_info=True,
            )
            return None, None, None

    async def get_image_from_rrm(
        self,
        m_json: JsonType,  # json data
        destinations: None = None,  # MQTT destinations for labels
    ) -> PilPNG or None:
        """Generate Images from the json data."""
        colors: Colors = {
            name: self.shared.user_colors[idx] for idx, name in enumerate(COLORS)
        }
        self.active_zones = self.shared.rand256_active_zone

        try:
            if (m_json is not None) and (not isinstance(m_json, tuple)):
                _LOGGER.info("%s: Composing the image for the camera.", self.file_name)
                self.json_data = m_json
                size_x, size_y = self.data.get_rrm_image_size(m_json)
                self.img_size = DEFAULT_IMAGE_SIZE
                self.json_id = str(uuid.uuid4())  # image id
                _LOGGER.info("Vacuum Data ID: %s", self.json_id)

                (
                    img_np_array,
                    robot_position,
                    robot_position_angle,
                ) = await self._setup_robot_and_image(
                    m_json, size_x, size_y, colors, destinations
                )

                # Increment frame number
                self.frame_number += 1
                img_np_array = await self.async_copy_array(self.img_base_layer)
                _LOGGER.debug(
                    "%s: Frame number %s", self.file_name, str(self.frame_number)
                )
                if self.frame_number > 5:
                    self.frame_number = 0

                # Draw map elements
                img_np_array = await self._draw_map_elements(
                    img_np_array, m_json, colors, robot_position, robot_position_angle
                )

                # Final adjustments
                pil_img = Image.fromarray(img_np_array, mode="RGBA")
                del img_np_array  # free memory

                return await self._finalize_image(pil_img)

        except (RuntimeError, RuntimeWarning) as e:
            _LOGGER.warning(
                "%s: Runtime Error %s during image creation.",
                self.file_name,
                str(e),
                exc_info=True,
            )
            return None

    async def _setup_robot_and_image(
        self, m_json, size_x, size_y, colors, destinations
    ):
        (
            _,
            robot_position,
            robot_position_angle,
        ) = await self.imd.async_get_robot_position(m_json)

        if self.frame_number == 0:
            # Create element map for tracking what's drawn where
            self.element_map = np.zeros((size_y, size_x), dtype=np.int32)
            self.element_map[:] = DrawableElement.FLOOR

            # Draw base layer if floor is enabled
            if self.drawing_config.is_enabled(DrawableElement.FLOOR):
                room_id, img_np_array = await self.imd.async_draw_base_layer(
                    m_json,
                    size_x,
                    size_y,
                    colors["wall"],
                    colors["zone_clean"],
                    colors["background"],
                    DEFAULT_PIXEL_SIZE,
                )
                _LOGGER.info("%s: Completed base Layers", self.file_name)

                # Update element map for rooms
                if 0 < room_id <= 15:
                    # This is a simplification - in a real implementation we would
                    # need to identify the exact pixels that belong to each room
                    pass

                if room_id > 0 and not self.room_propriety:
                    self.room_propriety = await self.get_rooms_attributes(destinations)
                    if self.rooms_pos:
                        self.robot_pos = await self.async_get_robot_in_room(
                            (robot_position[0] * 10),
                            (robot_position[1] * 10),
                            robot_position_angle,
                        )
                self.img_base_layer = await self.async_copy_array(img_np_array)
            else:
                # If floor is disabled, create an empty image
                background_color = self.drawing_config.get_property(
                    DrawableElement.FLOOR, "color", colors["background"]
                )
                img_np_array = await self.draw.create_empty_image(
                    size_x, size_y, background_color
                )
                self.img_base_layer = await self.async_copy_array(img_np_array)
        return self.img_base_layer, robot_position, robot_position_angle

    async def _draw_map_elements(
        self, img_np_array, m_json, colors, robot_position, robot_position_angle
    ):
        # Create element map for tracking what's drawn where if it doesn't exist
        if self.element_map is None:
            self.element_map = np.zeros(
                (img_np_array.shape[0], img_np_array.shape[1]), dtype=np.int32
            )
            self.element_map[:] = DrawableElement.FLOOR

        # Draw charger if enabled
        if self.drawing_config.is_enabled(DrawableElement.CHARGER):
            img_np_array, self.charger_pos = await self.imd.async_draw_charger(
                img_np_array, m_json, colors["charger"]
            )
            # Update element map for charger position
            if self.charger_pos:
                charger_radius = 15
                for dy in range(-charger_radius, charger_radius + 1):
                    for dx in range(-charger_radius, charger_radius + 1):
                        if dx * dx + dy * dy <= charger_radius * charger_radius:
                            cx, cy = (
                                int(self.charger_pos[0] + dx),
                                int(self.charger_pos[1] + dy),
                            )
                            if (
                                0 <= cy < self.element_map.shape[0]
                                and 0 <= cx < self.element_map.shape[1]
                            ):
                                self.element_map[cy, cx] = DrawableElement.CHARGER

        # Draw zones if enabled
        if self.drawing_config.is_enabled(DrawableElement.RESTRICTED_AREA):
            img_np_array = await self.imd.async_draw_zones(
                m_json, img_np_array, colors["zone_clean"]
            )

        # Draw virtual restrictions if enabled
        if self.drawing_config.is_enabled(DrawableElement.VIRTUAL_WALL):
            img_np_array = await self.imd.async_draw_virtual_restrictions(
                m_json, img_np_array, colors["no_go"]
            )

        # Draw path if enabled
        if self.drawing_config.is_enabled(DrawableElement.PATH):
            img_np_array = await self.imd.async_draw_path(
                img_np_array, m_json, colors["move"]
            )

        # Draw go-to flag if enabled
        if self.drawing_config.is_enabled(DrawableElement.GO_TO_TARGET):
            img_np_array = await self.imd.async_draw_go_to_flag(
                img_np_array, m_json, colors["go_to"]
            )

        # Draw robot if enabled
        if robot_position and self.drawing_config.is_enabled(DrawableElement.ROBOT):
            # Get robot color (allows for customization)
            robot_color = self.drawing_config.get_property(
                DrawableElement.ROBOT, "color", colors["robot"]
            )

            img_np_array = await self.imd.async_draw_robot_on_map(
                img_np_array, robot_position, robot_position_angle, robot_color
            )

            # Update element map for robot position
            from .config.utils import update_element_map_with_robot
            update_element_map_with_robot(self.element_map, robot_position, DrawableElement.ROBOT)
        img_np_array = await self.async_auto_trim_and_zoom_image(
            img_np_array,
            detect_colour=colors["background"],
            margin_size=int(self.shared.margins),
            rotate=int(self.shared.image_rotate),
            zoom=self.zooming,
            rand256=True,
        )
        return img_np_array

    async def _finalize_image(self, pil_img):
        if not self.shared.image_ref_width or not self.shared.image_ref_height:
            _LOGGER.warning(
                "Image finalization failed: Invalid image dimensions. Returning original image."
            )
            return pil_img
        if self.check_zoom_and_aspect_ratio():
            resize_params = prepare_resize_params(self, pil_img, True)
            pil_img = await self.async_resize_images(resize_params)
        _LOGGER.debug("%s: Frame Completed.", self.file_name)
        return pil_img

    async def get_rooms_attributes(
        self, destinations: JsonType = None
    ) -> RoomsProperties:
        """Return the rooms attributes."""
        if self.room_propriety:
            return self.room_propriety
        if self.json_data and destinations:
            _LOGGER.debug("Checking for rooms data..")
            self.room_propriety = await self.extract_room_properties(
                self.json_data, destinations
            )
            if self.room_propriety:
                _LOGGER.debug("Got Rooms Attributes.")
        return self.room_propriety

    async def async_get_robot_in_room(
        self, robot_x: int, robot_y: int, angle: float
    ) -> RobotPosition:
        """Get the robot position and return in what room is."""

        def _check_robot_position(x: int, y: int) -> bool:
            # Check if the robot coordinates are inside the room's corners
            return (
                self.robot_in_room["left"] >= x >= self.robot_in_room["right"]
                and self.robot_in_room["up"] >= y >= self.robot_in_room["down"]
            )

        # If the robot coordinates are inside the room's
        if self.robot_in_room and _check_robot_position(robot_x, robot_y):
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": self.robot_in_room["room"],
            }
            self.active_zones = self.shared.rand256_active_zone
            self.zooming = False
            if self.active_zones and (
                (self.robot_in_room["id"]) in range(len(self.active_zones))
            ):  # issue #100 Index out of range
                self.zooming = bool(self.active_zones[self.robot_in_room["id"]])
            return temp
        # else we need to search and use the async method
        _LOGGER.debug("%s Changed room.. searching..", self.file_name)
        room_count = -1
        last_room = None
        if self.rooms_pos:
            if self.robot_in_room:
                last_room = self.robot_in_room
            for room in self.rooms_pos:
                corners = room["corners"]
                room_count += 1
                self.robot_in_room = {
                    "id": room_count,
                    "left": corners[0][0],
                    "right": corners[2][0],
                    "up": corners[0][1],
                    "down": corners[2][1],
                    "room": room["name"],
                }
                # Check if the robot coordinates are inside the room's corners
                if _check_robot_position(robot_x, robot_y):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.robot_in_room["room"],
                    }
                    _LOGGER.debug(
                        "%s is in %s", self.file_name, self.robot_in_room["room"]
                    )
                    del room, corners, robot_x, robot_y  # free memory.
                    return temp
            del room, corners  # free memory.
            _LOGGER.debug(
                "%s: Not located within Camera Rooms coordinates.", self.file_name
            )
            self.zooming = False
            self.robot_in_room = last_room
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": self.robot_in_room["room"],
            }
            return temp

    def get_calibration_data(self, rotation_angle: int = 0) -> Any:
        """Return the map calibration data."""
        if not self.calibration_data and self.crop_img_size:
            self.calibration_data = []
            _LOGGER.info(
                "%s: Getting Calibrations points %s",
                self.file_name,
                str(self.crop_area),
            )

            # Define the map points (fixed)
            map_points = self.get_map_points()

            # Valetudo Re version need corrections of the coordinates and are implemented with *10
            vacuum_points = self.re_get_vacuum_points(rotation_angle)

            # Create the calibration data for each point
            for vacuum_point, map_point in zip(vacuum_points, map_points):
                calibration_point = {"vacuum": vacuum_point, "map": map_point}
                self.calibration_data.append(calibration_point)

        return self.calibration_data

    # Element selection methods
    def enable_element(self, element_code: DrawableElement) -> None:
        """Enable drawing of a specific element."""
        self.drawing_config.enable_element(element_code)

    def disable_element(self, element_code: DrawableElement) -> None:
        """Disable drawing of a specific element."""
        from .config.utils import manage_drawable_elements
        manage_drawable_elements(self, "disable", element_code=element_code)

    def set_elements(self, element_codes: list[DrawableElement]) -> None:
        """Enable only the specified elements, disable all others."""
        from .config.utils import manage_drawable_elements
        manage_drawable_elements(self, "set_elements", element_codes=element_codes)

    def set_element_property(
        self, element_code: DrawableElement, property_name: str, value
    ) -> None:
        """Set a drawing property for an element."""
        from .config.utils import manage_drawable_elements
        manage_drawable_elements(self, "set_property", element_code=element_code, property_name=property_name, value=value)

    def get_element_at_position(self, x: int, y: int) -> DrawableElement:
        """Get the element code at a specific position."""
        from .config.utils import get_element_at_position
        return get_element_at_position(self.element_map, x, y)

    def get_room_at_position(self, x: int, y: int) -> int:
        """Get the room ID at a specific position, or None if not a room."""
        from .config.utils import get_room_at_position
        return get_room_at_position(self.element_map, x, y, DrawableElement.ROOM_1)
