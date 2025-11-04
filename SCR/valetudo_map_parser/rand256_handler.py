"""
Image Handler Module for Valetudo Re Vacuums.
It returns the PIL PNG image frame relative to the Map Data extrapolated from the vacuum json.
It also returns calibration, rooms data to the card and other images information to the camera.
Version: 0.1.10
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from mvcrender.autocrop import AutoCrop

from .config.async_utils import AsyncPIL
from .config.drawable_elements import DrawableElement
from .config.types import (
    LOGGER,
    Colors,
    Destinations,
    JsonType,
    PilPNG,
    RobotPosition,
    RoomsProperties,
    RoomStore,
)
from .config.utils import (
    BaseHandler,
    initialize_drawing_config,
    point_in_polygon,
)
from .const import COLORS, DEFAULT_IMAGE_SIZE, DEFAULT_PIXEL_SIZE
from .map_data import RandImageData
from .reimg_draw import ImageDraw
from .rooms_handler import RandRoomsHandler


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
        self.element_map = None  # Element map for tracking drawable elements
        self.robot_position = None  # Robot position for zoom functionality
        self.outlines = None  # Outlines data
        self.calibration_data = None  # Calibration data
        self.data = RandImageData  # Image Data

        # Initialize drawing configuration using the shared utility function
        self.drawing_config, self.draw = initialize_drawing_config(self)
        self.go_to = None  # Go to position data
        self.img_base_layer = None  # Base image layer
        self.img_work_layer = None  # Persistent working buffer (reused across frames)
        self.img_rotate = shared_data.image_rotate  # Image rotation
        self.room_propriety = None  # Room propriety data
        self.active_zones = None  # Active zones
        self.file_name = self.shared.file_name  # File name
        self.imd = ImageDraw(self)  # Image Draw
        self.rooms_handler = RandRoomsHandler(
            self.file_name, self.drawing_config
        )  # Room data handler

    async def extract_room_properties(
        self,
        json_data: JsonType,
        destinations: Destinations | None = None,
    ) -> RoomsProperties:
        """Extract the room properties."""
        # unsorted_id = RandImageData.get_rrm_segments_ids(json_data)
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

            dest_json = destinations if destinations else {}
            zones_data = dest_json.get("zones", [])
            points_data = dest_json.get("spots", [])

            # Use the RandRoomsHandler to extract room properties
            room_properties = await self.rooms_handler.async_extract_room_properties(
                json_data, dest_json
            )

            # Update self.rooms_pos from room_properties for compatibility with other methods
            self.rooms_pos = []
            for _, room_data in room_properties.items():
                self.rooms_pos.append(
                    {"name": room_data["name"], "outline": room_data["outline"]}
                )

            # Update shared.map_rooms with the full room properties (consistent with Hypfer)
            self.shared.map_rooms = room_properties
            # get the zones and points data
            self.shared.map_pred_zones = await self.async_zone_propriety(zones_data)
            # get the points data
            self.shared.map_pred_points = await self.async_points_propriety(points_data)

            if not (room_properties or self.shared.map_pred_zones):
                self.rooms_pos = None

            _ = RoomStore(self.file_name, room_properties)
            return room_properties
        except (RuntimeError, ValueError) as e:
            LOGGER.warning(
                "No rooms Data or Error in extract_room_properties: %s",
                e,
                exc_info=True,
            )
            return None

    async def get_image_from_rrm(
        self,
        m_json: JsonType,  # json data
        destinations: Destinations | None = None,  # MQTT destinations for labels
    ) -> PilPNG | None:
        """Generate Images from the json data.
        @param m_json: The JSON data to use to draw the image.
        @param destinations: MQTT destinations for labels (unused).
        @return Image.Image: PIL Image.
        """
        colors: Colors = {
            name: self.shared.user_colors[idx] for idx, name in enumerate(COLORS)
        }
        self.active_zones = self.shared.rand256_active_zone

        try:
            if (m_json is not None) and (not isinstance(m_json, tuple)):
                LOGGER.info("%s: Composing the image for the camera.", self.file_name)
                self.json_data = m_json
                size_x, size_y = self.data.get_rrm_image_size(m_json)
                self.img_size = DEFAULT_IMAGE_SIZE
                self.json_id = str(uuid.uuid4())  # image id
                LOGGER.info("Vacuum Data ID: %s", self.json_id)

                (
                    img_np_array,
                    robot_position,
                    robot_position_angle,
                ) = await self._setup_robot_and_image(
                    m_json, size_x, size_y, colors, destinations
                )

                # Increment frame number
                self.frame_number += 1
                if self.frame_number > 5:
                    self.frame_number = 0

                # Ensure persistent working buffer exists and matches base (allocate only when needed)
                if (
                    self.img_work_layer is None
                    or self.img_work_layer.shape != self.img_base_layer.shape
                    or self.img_work_layer.dtype != self.img_base_layer.dtype
                ):
                    # Delete old buffer before creating new one to free memory
                    if self.img_work_layer is not None:
                        del self.img_work_layer
                    self.img_work_layer = np.empty_like(self.img_base_layer)

                # Copy the base layer into the persistent working buffer (no new allocation per frame)
                np.copyto(self.img_work_layer, self.img_base_layer)
                img_np_array = self.img_work_layer

                # Draw map elements
                img_np_array = await self._draw_map_elements(
                    img_np_array, m_json, colors, robot_position, robot_position_angle
                )

                # Return PIL Image using async utilities
                pil_img = await AsyncPIL.async_fromarray(img_np_array, mode="RGBA")
                # Note: Don't delete img_np_array here as it's the persistent work buffer
                return await self._finalize_image(pil_img)

        except (RuntimeError, RuntimeWarning) as e:
            LOGGER.warning(
                "%s: Runtime Error %s during image creation.",
                self.file_name,
                str(e),
                exc_info=True,
            )
            return None

        # If we reach here without returning, return None
        return None

    async def _initialize_base_layer(
        self,
        m_json,
        size_x,
        size_y,
        colors,
        destinations,
        robot_position,
        robot_position_angle,
    ):
        """Initialize the base layer on first frame."""
        self.element_map = np.zeros((size_y, size_x), dtype=np.int32)
        self.element_map[:] = DrawableElement.FLOOR

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
            LOGGER.info("%s: Completed base Layers", self.file_name)

            if room_id > 0 and not self.room_propriety:
                self.room_propriety = await self.get_rooms_attributes(destinations)

            if not self.rooms_pos and not self.room_propriety:
                self.room_propriety = await self.get_rooms_attributes(destinations)

            if (
                self.rooms_pos
                and robot_position
                and (self.robot_pos is None or "in_room" not in self.robot_pos)
            ):
                self.robot_pos = await self.async_get_robot_in_room(
                    (robot_position[0] * 10),
                    (robot_position[1] * 10),
                    robot_position_angle,
                )
        else:
            background_color = self.drawing_config.get_property(
                DrawableElement.FLOOR, "color", colors["background"]
            )
            img_np_array = await self.draw.create_empty_image(
                size_x, size_y, background_color
            )

        if self.img_base_layer is not None:
            del self.img_base_layer
        self.img_base_layer = await self.async_copy_array(img_np_array)
        del img_np_array

    async def _check_zoom_conditions(self, m_json, robot_position, destinations):
        """Check and set zoom conditions based on active zones."""
        if not (
            self.shared.image_auto_zoom
            and self.shared.vacuum_state == "cleaning"
            and robot_position
            and destinations
        ):
            return

        try:
            temp_room_properties = (
                await self.rooms_handler.async_extract_room_properties(
                    m_json, destinations
                )
            )
            if temp_room_properties:
                temp_rooms_pos = []
                for _, room_data in temp_room_properties.items():
                    temp_rooms_pos.append(
                        {"name": room_data["name"], "outline": room_data["outline"]}
                    )
                original_rooms_pos = self.rooms_pos
                self.rooms_pos = temp_rooms_pos
                self.rooms_pos = original_rooms_pos
        except (ValueError, KeyError, TypeError):
            if (
                self.shared.image_auto_zoom
                and self.shared.vacuum_state == "cleaning"
                and robot_position
            ):
                self.zooming = True

    async def _setup_robot_and_image(
        self, m_json, size_x, size_y, colors, destinations
    ):
        """Set up the elements of the map and the image."""
        (
            _,
            robot_position,
            robot_position_angle,
        ) = await self.imd.async_get_robot_position(m_json)

        if self.frame_number == 0:
            await self._initialize_base_layer(
                m_json,
                size_x,
                size_y,
                colors,
                destinations,
                robot_position,
                robot_position_angle,
            )

        await self._check_zoom_conditions(m_json, robot_position, destinations)

        return self.img_base_layer, robot_position, robot_position_angle

    async def _draw_map_elements(
        self, img_np_array, m_json, colors, robot_position, robot_position_angle
    ):
        """Draw map elements on the image."""
        # Draw charger if enabled
        if self.drawing_config.is_enabled(DrawableElement.CHARGER):
            img_np_array, self.charger_pos = await self.imd.async_draw_charger(
                img_np_array, m_json, colors["charger"]
            )

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

        # Store robot position for potential zoom function use
        if robot_position:
            self.robot_position = robot_position

        # Check if Zoom should be enabled based on active zones
        if (
            self.shared.image_auto_zoom
            and self.shared.vacuum_state == "cleaning"
            and robot_position
        ):
            # For Rand256, we need to check active zones differently since room data is not available yet
            # Use a simplified approach: enable zoom if any active zones are set
            active_zones = self.shared.rand256_active_zone
            if active_zones and any(zone for zone in active_zones):
                self.zooming = True
            else:
                self.zooming = False

        img_np_array = self.auto_trim_and_zoom_image(
            img_np_array,
            detect_colour=colors["background"],
            margin_size=int(self.shared.margins),
            rotate=int(self.shared.image_rotate),
            zoom=self.zooming,
            rand256=True,
        )
        return img_np_array

    async def _finalize_image(self, pil_img):
        """Finalize the image by resizing if needed."""
        if pil_img is None:
            LOGGER.warning("%s: Image is None. Returning None.", self.file_name)
            return None
        if self.check_zoom_and_aspect_ratio():
            resize_params = self.prepare_resize_params(pil_img, True)
            pil_img = await self.async_resize_images(resize_params)
        return pil_img

    async def get_rooms_attributes(
        self, destinations: JsonType = None
    ) -> tuple[RoomsProperties, Any, Any]:
        """Return the rooms attributes."""
        if self.json_data and destinations:
            self.room_propriety = await self.extract_room_properties(
                self.json_data, destinations
            )
        return self.room_propriety

    def _create_robot_position_dict(
        self, robot_x: int, robot_y: int, angle: float, room_name: str
    ) -> RobotPosition:
        """Create a robot position dictionary."""
        return {
            "x": robot_x,
            "y": robot_y,
            "angle": angle,
            "in_room": room_name,
        }

    def _set_zooming_from_active_zones(self) -> None:
        """Set zooming based on active zones."""
        self.active_zones = self.shared.rand256_active_zone
        self.zooming = False
        if self.active_zones and (
            self.robot_in_room["id"] in range(len(self.active_zones))
        ):
            self.zooming = bool(self.active_zones[self.robot_in_room["id"]])

    def _check_cached_room_outline_rand(
        self, robot_x: int, robot_y: int, angle: float
    ) -> RobotPosition | None:
        """Check if robot is still in cached room using outline."""
        if "outline" in self.robot_in_room:
            outline = self.robot_in_room["outline"]
            if point_in_polygon(int(robot_x), int(robot_y), outline):
                self._set_zooming_from_active_zones()
                LOGGER.debug(
                    "%s: Robot is in %s room (polygon detection). %s",
                    self.file_name,
                    self.robot_in_room["room"],
                    self.active_zones,
                )
                return self._create_robot_position_dict(
                    robot_x, robot_y, angle, self.robot_in_room["room"]
                )
        return None

    def _check_cached_room_bbox_rand(
        self, robot_x: int, robot_y: int, angle: float
    ) -> RobotPosition | None:
        """Check if robot is still in cached room using bounding box."""
        if all(k in self.robot_in_room for k in ["left", "right", "up", "down"]):
            if (
                self.robot_in_room["right"]
                <= int(robot_x)
                <= self.robot_in_room["left"]
            ) and (
                self.robot_in_room["up"] <= int(robot_y) <= self.robot_in_room["down"]
            ):
                self._set_zooming_from_active_zones()
                return self._create_robot_position_dict(
                    robot_x, robot_y, angle, self.robot_in_room["room"]
                )
        return None

    def _check_room_with_outline_rand(
        self, room: dict, room_count: int, robot_x: int, robot_y: int, angle: float
    ) -> RobotPosition | None:
        """Check if robot is in room using outline polygon."""
        outline = room["outline"]
        if point_in_polygon(int(robot_x), int(robot_y), outline):
            self.robot_in_room = {
                "id": room_count,
                "room": str(room["name"]),
                "outline": outline,
            }
            self._set_zooming_from_active_zones()
            return self._create_robot_position_dict(
                robot_x, robot_y, angle, self.robot_in_room["room"]
            )
        return None

    async def async_get_robot_in_room(
        self, robot_x: int, robot_y: int, angle: float
    ) -> RobotPosition:
        """Get the robot position and return in what room is."""
        # Check cached room first
        if self.robot_in_room:
            result = self._check_cached_room_outline_rand(robot_x, robot_y, angle)
            if result:
                return result
            result = self._check_cached_room_bbox_rand(robot_x, robot_y, angle)
            if result:
                return result

        # Prepare for room search
        last_room = self.robot_in_room
        map_boundary = 50000

        # Check boundary conditions or missing room data
        if (
            abs(robot_x) > map_boundary
            or abs(robot_y) > map_boundary
            or not self.rooms_pos
        ):
            self.robot_in_room = last_room
            self.zooming = False
            return self._create_robot_position_dict(
                robot_x, robot_y, angle, last_room["room"] if last_room else "unknown"
            )

        # Search through all rooms
        for room_count, room in enumerate(self.rooms_pos):
            if "outline" in room:
                result = self._check_room_with_outline_rand(
                    room, room_count, robot_x, robot_y, angle
                )
                if result:
                    return result

        # Robot not found in any room
        self.robot_in_room = last_room
        self.zooming = False
        return self._create_robot_position_dict(
            robot_x, robot_y, angle, last_room["room"] if last_room else "unknown"
        )

    def get_calibration_data(self, rotation_angle: int = 0) -> Any:
        """Return the map calibration data."""
        if not self.calibration_data and self.crop_img_size:
            self.calibration_data = []
            LOGGER.info(
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
