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

from PIL import Image

from .config.auto_crop import AutoCrop
from .config.types import (
    COLORS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PIXEL_SIZE,
    Colors,
    JsonType,
    PilPNG,
    RobotPosition,
    RoomsProperties,
)
from .config.utils import BaseHandler
from .map_data import RandImageData
from .reimg_draw import ImageDraw

_LOGGER = logging.getLogger(__name__)


# noinspection PyTypeChecker
class ReImageHandler(BaseHandler):
    """
    Image Handler for Valetudo Re Vacuums.
    """

    def __init__(self, camera_shared):
        super().__init__()
        self.auto_crop = None  # Auto crop flag
        self.segment_data = None  # Segment data
        self.outlines = None  # Outlines data
        self.calibration_data = None  # Calibration data
        self.crop_area = None  # Crop area
        self.data = RandImageData  # Image Data
        self.go_to = None  # Go to position data
        self.img_base_layer = None  # Base image layer
        self.img_rotate = camera_shared.image_rotate  # Image rotation
        self.room_propriety = None  # Room propriety data
        self.shared = camera_shared  # Shared data
        self.active_zones = None  # Active zones
        self.trim_down = None  # Trim down
        self.trim_left = None  # Trim left
        self.trim_right = None  # Trim right
        self.trim_up = None  # Trim up
        self.file_name = self.shared.file_name  # File name
        self.offset_top = self.shared.offset_top  # offset top
        self.offset_bottom = self.shared.offset_down  # offset bottom
        self.offset_left = self.shared.offset_left  # offset left
        self.offset_right = self.shared.offset_right  # offset right
        self.imd = ImageDraw(self)  # Image Draw
        self.crop = AutoCrop(self)

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
                        corners = self.get_corners(x_max, x_min, y_max, y_min)
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
                            "outline": corners,
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
                return room_properties, zone_properties, point_properties
        except RuntimeError as e:
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
            if (room_id > 0) and not self.room_propriety:
                self.room_propriety = await self.get_rooms_attributes(destinations)
                if self.rooms_pos:
                    self.robot_pos = await self.async_get_robot_in_room(
                        (robot_position[0] * 10),
                        (robot_position[1] * 10),
                        robot_position_angle,
                    )
            self.img_base_layer = await self.async_copy_array(img_np_array)
        return self.img_base_layer, robot_position, robot_position_angle

    async def _draw_map_elements(
        self, img_np_array, m_json, colors, robot_position, robot_position_angle
    ):
        img_np_array, self.charger_pos = await self.imd.async_draw_charger(
            img_np_array, m_json, colors["charger"]
        )
        img_np_array = await self.imd.async_draw_zones(
            m_json, img_np_array, colors["zone_clean"]
        )
        img_np_array = await self.imd.async_draw_virtual_restrictions(
            m_json, img_np_array, colors["no_go"]
        )
        img_np_array = await self.imd.async_draw_path(
            img_np_array, m_json, colors["move"]
        )
        img_np_array = await self.imd.async_draw_go_to_flag(
            img_np_array, m_json, colors["go_to"]
        )
        img_np_array = await self.imd.async_draw_robot_on_map(
            img_np_array, robot_position, robot_position_angle, colors["robot"]
        )
        img_np_array = await self.crop.async_auto_trim_and_zoom_image(
            img_np_array,
            detect_colour=colors["background"],
            margin_size=int(self.shared.margins),
            rotate=int(self.shared.image_rotate),
            zoom=self.zooming,
            rand256=True,
        )
        return img_np_array

    async def _finalize_image(self, pil_img):
        if self.check_zoom_and_aspect_ratio():
            pil_img = await self.async_resize_image(
                pil_img, self.shared.image_aspect_ratio, True
            )
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
