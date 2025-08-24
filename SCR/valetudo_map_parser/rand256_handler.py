"""
Image Handler Module for Valetudo Re Vacuums.
It returns the PIL PNG image frame relative to the Map Data extrapolated from the vacuum json.
It also returns calibration, rooms data to the card and other images information to the camera.
Version: 0.1.9.a6
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from .config.async_utils import AsyncNumPy, AsyncPIL
from .config.auto_crop import AutoCrop
from .config.drawable_elements import DrawableElement
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
    WebPBytes,
)
from .config.utils import (
    BaseHandler,
    initialize_drawing_config,
    manage_drawable_elements,
    numpy_to_webp_bytes,
    prepare_resize_params,
)
from .map_data import RandImageData
from .reimg_draw import ImageDraw
from .rooms_handler import RandRoomsHandler


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
        self.drawing_config, self.draw, self.enhanced_draw = initialize_drawing_config(
            self
        )
        self.go_to = None  # Go to position data
        self.img_base_layer = None  # Base image layer
        self.img_rotate = shared_data.image_rotate  # Image rotation
        self.room_propriety = None  # Room propriety data
        self.active_zones = None  # Active zones
        self.file_name = self.shared.file_name  # File name
        self.imd = ImageDraw(self)  # Image Draw
        self.rooms_handler = RandRoomsHandler(
            self.file_name, self.drawing_config
        )  # Room data handler

    async def extract_room_properties(
        self, json_data: JsonType, destinations: JsonType
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

            dest_json = destinations
            zones_data = dict(dest_json).get("zones", [])
            points_data = dict(dest_json).get("spots", [])

            # Use the RandRoomsHandler to extract room properties
            room_properties = await self.rooms_handler.async_extract_room_properties(
                json_data, dest_json
            )

            # Update self.rooms_pos from room_properties for compatibility with other methods
            self.rooms_pos = []
            room_ids = []  # Collect room IDs for shared.map_rooms
            for room_id, room_data in room_properties.items():
                self.rooms_pos.append(
                    {"name": room_data["name"], "outline": room_data["outline"]}
                )
                # Store the room number (segment ID) for MQTT active zone mapping
                room_ids.append(room_data["number"])

            # Update shared.map_rooms with the room IDs for MQTT active zone mapping
            self.shared.map_rooms = room_ids
            _LOGGER.debug("Updated shared.map_rooms with room IDs: %s", room_ids)

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
                _LOGGER.debug("%s: Rooms and Zones data not available!", self.file_name)

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
        return_webp: bool = False,
    ) -> WebPBytes | PilPNG | None:
        """Generate Images from the json data.
        @param m_json: The JSON data to use to draw the image.
        @param destinations: MQTT destinations for labels (unused).
        @param return_webp: If True, return WebP bytes; if False, return PIL Image (default).
        @return WebPBytes | Image.Image: WebP bytes or PIL Image depending on return_webp parameter.
        """
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

                # Return WebP bytes or PIL Image based on parameter
                if return_webp:
                    # Convert directly to WebP bytes for better performance
                    webp_bytes = await numpy_to_webp_bytes(img_np_array)
                    del img_np_array  # free memory
                    return webp_bytes
                else:
                    # Convert to PIL Image using async utilities
                    pil_img = await AsyncPIL.async_fromarray(img_np_array, mode="RGBA")
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

        # If we reach here without returning, return None
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

                # Ensure room data is available for robot room detection (even if not extracted above)
                if not self.rooms_pos and not self.room_propriety:
                    self.room_propriety = await self.get_rooms_attributes(destinations)

                # Always check robot position for zooming (fallback)
                if self.rooms_pos and robot_position and not hasattr(self, "robot_pos"):
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

        # Check active zones BEFORE auto-crop to enable proper zoom functionality
        # This needs to run on every frame, not just frame 0
        if (
            self.shared.image_auto_zoom
            and self.shared.vacuum_state == "cleaning"
            and robot_position
            and destinations  # Check if we have destinations data for room extraction
        ):
            # Extract room data early if we have destinations
            try:
                temp_room_properties = (
                    await self.rooms_handler.async_extract_room_properties(
                        m_json, destinations
                    )
                )
                if temp_room_properties:
                    # Create temporary rooms_pos for robot room detection
                    temp_rooms_pos = []
                    for room_id, room_data in temp_room_properties.items():
                        temp_rooms_pos.append(
                            {"name": room_data["name"], "outline": room_data["outline"]}
                        )

                    # Store original rooms_pos and temporarily use the new one
                    original_rooms_pos = self.rooms_pos
                    self.rooms_pos = temp_rooms_pos

                    # Restore original rooms_pos
                    self.rooms_pos = original_rooms_pos

            except Exception as e:
                _LOGGER.debug(
                    "%s: Early room extraction failed: %s, falling back to robot-position zoom",
                    self.file_name,
                    e,
                )
                # Fallback to robot-position-based zoom if room extraction fails
                if (
                    self.shared.image_auto_zoom
                    and self.shared.vacuum_state == "cleaning"
                    and robot_position
                ):
                    self.zooming = True
                    _LOGGER.debug(
                        "%s: Enabling fallback robot-position-based zoom",
                        self.file_name,
                    )

        return self.img_base_layer, robot_position, robot_position_angle

    async def _draw_map_elements(
        self, img_np_array, m_json, colors, robot_position, robot_position_angle
    ):
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
                _LOGGER.debug(
                    "%s: Enabling zoom for Rand256 - active zones detected: %s",
                    self.file_name,
                    active_zones,
                )
            else:
                self.zooming = False
                _LOGGER.debug(
                    "%s: Zoom disabled for Rand256 - no active zones set",
                    self.file_name,
                )

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
    ) -> tuple[RoomsProperties, Any, Any]:
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

    @staticmethod
    def point_in_polygon(x: int, y: int, polygon: list) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        Enhanced version with better handling of edge cases.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            polygon: List of (x, y) tuples forming the polygon

        Returns:
            True if the point is inside the polygon, False otherwise
        """
        # Ensure we have a valid polygon with at least 3 points
        if len(polygon) < 3:
            return False

        # Make sure the polygon is closed (last point equals first point)
        if polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        # Use winding number algorithm for better accuracy
        wn = 0  # Winding number counter

        # Loop through all edges of the polygon
        for i in range(len(polygon) - 1):  # Last vertex is first vertex
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[i + 1]

            # Test if a point is left/right/on the edge defined by two vertices
            if p1y <= y:  # Start y <= P.y
                if p2y > y:  # End y > P.y (upward crossing)
                    # Point left of edge
                    if ((p2x - p1x) * (y - p1y) - (x - p1x) * (p2y - p1y)) > 0:
                        wn += 1  # Valid up intersect
            else:  # Start y > P.y
                if p2y <= y:  # End y <= P.y (downward crossing)
                    # Point right of edge
                    if ((p2x - p1x) * (y - p1y) - (x - p1x) * (p2y - p1y)) < 0:
                        wn -= 1  # Valid down intersect

        # If winding number is not 0, the point is inside the polygon
        return wn != 0

    async def async_get_robot_in_room(
        self, robot_x: int, robot_y: int, angle: float
    ) -> RobotPosition:
        """Get the robot position and return in what room is."""
        # First check if we already have a cached room and if the robot is still in it
        if self.robot_in_room:
            # If we have outline data, use point_in_polygon for accurate detection
            if "outline" in self.robot_in_room:
                outline = self.robot_in_room["outline"]
                if self.point_in_polygon(int(robot_x), int(robot_y), outline):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.robot_in_room["room"],
                    }
                    # Handle active zones
                    self.active_zones = self.shared.rand256_active_zone
                    self.zooming = False
                    if self.active_zones and (
                        self.robot_in_room["id"] in range(len(self.active_zones))
                    ):
                        self.zooming = bool(self.active_zones[self.robot_in_room["id"]])
                    else:
                        self.zooming = False
                    return temp
            # Fallback to bounding box check if no outline data
            elif all(k in self.robot_in_room for k in ["left", "right", "up", "down"]):
                if (
                    self.robot_in_room["right"]
                    <= int(robot_x)
                    <= self.robot_in_room["left"]
                ) and (
                    self.robot_in_room["up"]
                    <= int(robot_y)
                    <= self.robot_in_room["down"]
                ):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.robot_in_room["room"],
                    }
                    # Handle active zones
                    self.active_zones = self.shared.rand256_active_zone
                    self.zooming = False
                    if self.active_zones and (
                        self.robot_in_room["id"] in range(len(self.active_zones))
                    ):
                        self.zooming = bool(self.active_zones[self.robot_in_room["id"]])
                    else:
                        self.zooming = False
                    return temp

        # If we don't have a cached room or the robot is not in it, search all rooms
        last_room = None
        room_count = 0
        if self.robot_in_room:
            last_room = self.robot_in_room

        # Check if the robot is far outside the normal map boundaries
        # This helps prevent false positives for points very far from any room
        map_boundary = 50000  # Typical map size is around 25000-30000 units for Rand25
        if abs(robot_x) > map_boundary or abs(robot_y) > map_boundary:
            _LOGGER.debug(
                "%s robot position (%s, %s) is far outside map boundaries.",
                self.file_name,
                robot_x,
                robot_y,
            )
            self.robot_in_room = last_room
            self.zooming = False
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": last_room["room"] if last_room else "unknown",
            }
            return temp

        # Search through all rooms to find which one contains the robot
        if not self.rooms_pos:
            _LOGGER.debug(
                "%s: No rooms data available for robot position detection.",
                self.file_name,
            )
            self.robot_in_room = last_room
            self.zooming = False
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": last_room["room"] if last_room else "unknown",
            }
            return temp

        _LOGGER.debug("%s: Searching for robot in rooms...", self.file_name)
        for room in self.rooms_pos:
            # Check if the room has an outline (polygon points)
            if "outline" in room:
                outline = room["outline"]
                # Use point_in_polygon for accurate detection with complex shapes
                if self.point_in_polygon(int(robot_x), int(robot_y), outline):
                    # Robot is in this room
                    self.robot_in_room = {
                        "id": room_count,
                        "room": str(room["name"]),
                        "outline": outline,
                    }
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.robot_in_room["room"],
                    }

                    # Handle active zones - Set zooming based on active zones
                    self.active_zones = self.shared.rand256_active_zone
                    if self.active_zones and (
                        self.robot_in_room["id"] in range(len(self.active_zones))
                    ):
                        self.zooming = bool(self.active_zones[self.robot_in_room["id"]])
                    else:
                        self.zooming = False

                    _LOGGER.debug(
                        "%s is in %s room (polygon detection).",
                        self.file_name,
                        self.robot_in_room["room"],
                    )
                    return temp
            room_count += 1

        # Robot not found in any room
        _LOGGER.debug(
            "%s not located within any room coordinates.",
            self.file_name,
        )
        self.robot_in_room = last_room
        self.zooming = False
        temp = {
            "x": robot_x,
            "y": robot_y,
            "angle": angle,
            "in_room": last_room["room"] if last_room else "unknown",
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
        manage_drawable_elements(self, "disable", element_code=element_code)

    def set_elements(self, element_codes: list[DrawableElement]) -> None:
        """Enable only the specified elements, disable all others."""
        manage_drawable_elements(self, "set_elements", element_codes=element_codes)

    def set_element_property(
        self, element_code: DrawableElement, property_name: str, value
    ) -> None:
        """Set a drawing property for an element."""
        manage_drawable_elements(
            self,
            "set_property",
            element_code=element_code,
            property_name=property_name,
            value=value,
        )

    async def async_copy_array(self, original_array):
        """Copy the array using async utilities."""
        return await AsyncNumPy.async_copy(original_array)
