"""
Hypfer Image Handler Class.
It returns the PIL PNG image frame relative to the Map Data extrapolated from the vacuum json.
It also returns calibration, rooms data to the card and other images information to the camera.
Version: 0.1.9
"""

from __future__ import annotations

import asyncio

from PIL import Image

from .config.async_utils import AsyncNumPy, AsyncPIL
from .config.auto_crop import AutoCrop
from .config.drawable_elements import DrawableElement
from .config.shared import CameraShared
from .config.utils import pil_to_webp_bytes
from .config.types import (
    COLORS,
    LOGGER,
    CalibrationPoints,
    Colors,
    RoomsProperties,
    RoomStore,
    WebPBytes,
    JsonType,
)
from .config.utils import (
    BaseHandler,
    initialize_drawing_config,
    manage_drawable_elements,
    numpy_to_webp_bytes,
    prepare_resize_params,
)
from .hypfer_draw import ImageDraw as ImDraw
from .map_data import ImageData
from .rooms_handler import RoomsHandler


class HypferMapImageHandler(BaseHandler, AutoCrop):
    """Map Image Handler Class.
    This class is used to handle the image data and the drawing of the map."""

    def __init__(self, shared_data: CameraShared):
        """Initialize the Map Image Handler."""
        BaseHandler.__init__(self)
        self.shared = shared_data  # camera shared data
        AutoCrop.__init__(self, self)
        self.calibration_data = None  # camera shared data.
        self.data = ImageData  # imported Image Data Module.

        # Initialize drawing configuration using the shared utility function
        self.drawing_config, self.draw, self.enhanced_draw = initialize_drawing_config(
            self
        )

        self.go_to = None  # vacuum go to data
        self.img_hash = None  # hash of the image calculated to check differences.
        self.img_base_layer = None  # numpy array store the map base layer.
        self.active_zones = None  # vacuum active zones.
        self.svg_wait = False  # SVG image creation wait.
        self.imd = ImDraw(self)  # Image Draw class.
        self.color_grey = (128, 128, 128, 255)
        self.file_name = self.shared.file_name  # file name of the vacuum.
        self.rooms_handler = RoomsHandler(
            self.file_name, self.drawing_config
        )  # Room data handler

    @staticmethod
    def get_corners(x_max, x_min, y_max, y_min):
        """Get the corners of the room."""
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    async def async_extract_room_properties(self, json_data) -> RoomsProperties:
        """Extract room properties from the JSON data."""
        room_properties = await self.rooms_handler.async_extract_room_properties(
            json_data
        )
        if room_properties:
            rooms = RoomStore(self.file_name, room_properties)
            LOGGER.debug(
                "%s: Rooms data extracted! %s", self.file_name, rooms.get_rooms()
            )
            # Convert room_properties to the format expected by async_get_robot_in_room
            self.rooms_pos = []
            for room_id, room_data in room_properties.items():
                self.rooms_pos.append(
                    {
                        "id": room_id,
                        "name": room_data["name"],
                        "outline": room_data["outline"],
                    }
                )
        else:
            LOGGER.debug("%s: Rooms data not available!", self.file_name)
            self.rooms_pos = None
        return room_properties

    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    async def async_get_image_from_json(
        self,
        m_json: JsonType | None,
        return_webp: bool = False,
    ) -> WebPBytes | Image.Image | None:
        """Get the image from the JSON data.
        It uses the ImageDraw class to draw some of the elements of the image.
        The robot itself will be drawn in this function as per some of the values are needed for other tasks.
        @param m_json: The JSON data to use to draw the image.
        @param return_webp: If True, return WebP bytes; if False, return PIL Image (default).
        @return WebPBytes | Image.Image: WebP bytes or PIL Image depending on return_webp parameter.
        """
        # Initialize the colors.
        colors: Colors = {
            name: self.shared.user_colors[idx] for idx, name in enumerate(COLORS)
        }
        # Check if the JSON data is not None else process the image.
        try:
            if m_json is not None:
                LOGGER.debug("%s: Creating Image.", self.file_name)
                # buffer json data
                self.json_data = m_json
                # Get the image size from the JSON data
                size_x = int(m_json["size"]["x"])
                size_y = int(m_json["size"]["y"])
                self.img_size = {
                    "x": size_x,
                    "y": size_y,
                    "centre": [(size_x // 2), (size_y // 2)],
                }
                # Get the JSON ID from the JSON data.
                self.json_id = await self.imd.async_get_json_id(m_json)
                # Check entity data.
                entity_dict = await self.imd.async_get_entity_data(m_json)
                # Update the Robot position.
                (
                    robot_pos,
                    robot_position,
                    robot_position_angle,
                ) = await self.imd.async_get_robot_position(entity_dict)

                # Get the pixels size and layers from the JSON data
                pixel_size = int(m_json["pixelSize"])
                layers, active = self.data.find_layers(m_json["layers"], {}, [])
                # Populate active_zones from the JSON data
                self.active_zones = active
                new_frame_hash = await self.calculate_array_hash(layers, active)
                if self.frame_number == 0:
                    self.img_hash = new_frame_hash
                    # Create empty image
                    img_np_array = await self.draw.create_empty_image(
                        size_x, size_y, colors["background"]
                    )
                    # Draw layers and segments if enabled
                    room_id = 0
                    # Keep track of disabled rooms to skip their walls later
                    disabled_rooms = set()

                    if self.drawing_config.is_enabled(DrawableElement.FLOOR):
                        # First pass: identify disabled rooms
                        for layer_type, compressed_pixels_list in layers.items():
                            # Check if this is a room layer
                            if layer_type == "segment":
                                # The room_id is the current room being processed (0-based index)
                                # We need to check if ROOM_{room_id+1} is enabled (1-based in DrawableElement)
                                current_room_id = room_id + 1
                                if 1 <= current_room_id <= 15:
                                    room_element = getattr(
                                        DrawableElement, f"ROOM_{current_room_id}", None
                                    )
                                    if (
                                        room_element
                                        and not self.drawing_config.is_enabled(
                                            room_element
                                        )
                                    ):
                                        # Add this room to the disabled rooms set
                                        disabled_rooms.add(room_id)
                                        LOGGER.debug(
                                            "%s: Room %d is disabled and will be skipped",
                                            self.file_name,
                                            current_room_id,
                                        )
                                room_id = (
                                    room_id + 1
                                ) % 16  # Cycle room_id back to 0 after 15

                        # Reset room_id for the actual drawing pass
                        room_id = 0

                        # Second pass: draw enabled rooms and walls
                        for layer_type, compressed_pixels_list in layers.items():
                            # Check if this is a room layer
                            is_room_layer = layer_type == "segment"

                            # If it's a room layer, check if the specific room is enabled
                            if is_room_layer:
                                # The room_id is the current room being processed (0-based index)
                                # We need to check if ROOM_{room_id+1} is enabled (1-based in DrawableElement)
                                current_room_id = room_id + 1
                                if 1 <= current_room_id <= 15:
                                    room_element = getattr(
                                        DrawableElement, f"ROOM_{current_room_id}", None
                                    )

                                    # Skip this room if it's disabled
                                    if not self.drawing_config.is_enabled(room_element):
                                        room_id = (
                                            room_id + 1
                                        ) % 16  # Increment room_id even if we skip
                                        continue

                            # Check if this is a wall layer and if walls are enabled
                            is_wall_layer = layer_type == "wall"
                            if is_wall_layer:
                                if not self.drawing_config.is_enabled(
                                    DrawableElement.WALL
                                ):
                                    pass

                            # Draw the layer
                            (
                                room_id,
                                img_np_array,
                            ) = await self.imd.async_draw_base_layer(
                                img_np_array,
                                compressed_pixels_list,
                                layer_type,
                                colors["wall"],
                                colors["zone_clean"],
                                pixel_size,
                                disabled_rooms if layer_type == "wall" else None,
                            )

                    # Draw the virtual walls if enabled
                    if self.drawing_config.is_enabled(DrawableElement.VIRTUAL_WALL):
                        img_np_array = await self.imd.async_draw_virtual_walls(
                            m_json, img_np_array, colors["no_go"]
                        )

                    # Draw charger if enabled
                    if self.drawing_config.is_enabled(DrawableElement.CHARGER):
                        img_np_array = await self.imd.async_draw_charger(
                            img_np_array, entity_dict, colors["charger"]
                        )

                    # Draw obstacles if enabled
                    if self.drawing_config.is_enabled(DrawableElement.OBSTACLE):
                        self.shared.obstacles_pos = self.data.get_obstacles(entity_dict)
                        if self.shared.obstacles_pos:
                            img_np_array = await self.imd.async_draw_obstacle(
                                img_np_array, self.shared.obstacles_pos, colors["no_go"]
                            )
                    # Robot and rooms position
                    if (room_id > 0) and not self.room_propriety:
                        self.room_propriety = await self.async_extract_room_properties(
                            self.json_data
                        )

                    # Ensure room data is available for robot room detection (even if not extracted above)
                    if not self.rooms_pos and not self.room_propriety:
                        self.room_propriety = await self.async_extract_room_properties(
                            self.json_data
                        )

                    # Always check robot position for zooming (moved outside the condition)
                    if self.rooms_pos and robot_position and robot_position_angle:
                        self.robot_pos = await self.imd.async_get_robot_in_room(
                            robot_x=(robot_position[0]),
                            robot_y=(robot_position[1]),
                            angle=robot_position_angle,
                        )
                    LOGGER.info("%s: Completed base Layers", self.file_name)
                    # Copy the new array in base layer.
                    self.img_base_layer = await self.async_copy_array(img_np_array)
                self.shared.frame_number = self.frame_number
                self.frame_number += 1
                if (self.frame_number >= self.max_frames) or (
                    new_frame_hash != self.img_hash
                ):
                    self.frame_number = 0
                LOGGER.debug(
                    "%s: %s at Frame Number: %s",
                    self.file_name,
                    str(self.json_id),
                    str(self.frame_number),
                )
                # Copy the base layer to the new image.
                img_np_array = await self.async_copy_array(self.img_base_layer)

                # Prepare parallel data extraction tasks
                data_tasks = []

                # Prepare zone data extraction
                if self.drawing_config.is_enabled(DrawableElement.RESTRICTED_AREA):
                    data_tasks.append(self._prepare_zone_data(m_json))

                # Prepare go_to flag data extraction
                if self.drawing_config.is_enabled(DrawableElement.GO_TO_TARGET):
                    data_tasks.append(self._prepare_goto_data(entity_dict))

                # Prepare path data extraction
                path_enabled = self.drawing_config.is_enabled(DrawableElement.PATH)
                LOGGER.info("%s: PATH element enabled: %s", self.file_name, path_enabled)
                if path_enabled:
                    LOGGER.info("%s: Drawing path", self.file_name)
                    data_tasks.append(self._prepare_path_data(m_json))

                # Process drawing operations sequentially (since they modify the same array)
                # Draw zones if enabled
                if self.drawing_config.is_enabled(DrawableElement.RESTRICTED_AREA):
                    img_np_array = await self.imd.async_draw_zones(
                        m_json, img_np_array, colors["zone_clean"], colors["no_go"]
                    )

                # Draw the go_to target flag if enabled
                if self.drawing_config.is_enabled(DrawableElement.GO_TO_TARGET):
                    img_np_array = await self.imd.draw_go_to_flag(
                        img_np_array, entity_dict, colors["go_to"]
                    )

                # Draw paths if enabled
                if path_enabled:
                    img_np_array = await self.imd.async_draw_paths(
                        img_np_array, m_json, colors["move"], self.color_grey
                    )
                else:
                    LOGGER.info("%s: Skipping path drawing", self.file_name)

                # Check if the robot is docked.
                if self.shared.vacuum_state == "docked":
                    # Adjust the robot angle.
                    robot_position_angle -= 180

                # Draw the robot if enabled
                if robot_pos and self.drawing_config.is_enabled(DrawableElement.ROBOT):
                    # Get robot color (allows for customization)
                    robot_color = self.drawing_config.get_property(
                        DrawableElement.ROBOT, "color", colors["robot"]
                    )

                    # Draw the robot
                    img_np_array = await self.draw.robot(
                        layers=img_np_array,
                        x=robot_position[0],
                        y=robot_position[1],
                        angle=robot_position_angle,
                        fill=robot_color,
                        robot_state=self.shared.vacuum_state,
                    )

                    # Update element map for robot position
                    if (
                        hasattr(self.shared, "element_map")
                        and self.shared.element_map is not None
                    ):
                        update_element_map_with_robot(
                            self.shared.element_map,
                            robot_position,
                            DrawableElement.ROBOT,
                        )
                # Synchronize zooming state from ImageDraw to handler before auto-crop
                self.zooming = self.imd.img_h.zooming

                # Resize the image
                img_np_array = await self.async_auto_trim_and_zoom_image(
                    img_np_array,
                    colors["background"],
                    int(self.shared.margins),
                    int(self.shared.image_rotate),
                    self.zooming,
                )
            # If the image is None return None and log the error.
            if img_np_array is None:
                LOGGER.warning("%s: Image array is None.", self.file_name)
                return None

            # Handle resizing if needed, then return based on format preference
            if self.check_zoom_and_aspect_ratio():
                # Convert to PIL for resizing
                pil_img = await AsyncPIL.async_fromarray(img_np_array, mode="RGBA")
                del img_np_array
                resize_params = prepare_resize_params(self, pil_img, False)
                resized_image = await self.async_resize_images(resize_params)

                # Return WebP bytes or PIL Image based on parameter
                if return_webp:
                    webp_bytes = await pil_to_webp_bytes(resized_image)
                    return webp_bytes
                else:
                    return resized_image
            else:
                # Return WebP bytes or PIL Image based on parameter
                if return_webp:
                    # Convert directly from NumPy to WebP for better performance
                    webp_bytes = await numpy_to_webp_bytes(img_np_array)
                    del img_np_array
                    LOGGER.debug("%s: Frame Completed.", self.file_name)
                    return webp_bytes
                else:
                    # Convert to PIL Image (original behavior)
                    pil_img = await AsyncPIL.async_fromarray(img_np_array, mode="RGBA")
                    del img_np_array
                    LOGGER.debug("%s: Frame Completed.", self.file_name)
                    return pil_img
        except (RuntimeError, RuntimeWarning) as e:
            LOGGER.warning(
                "%s: Error %s during image creation.",
                self.file_name,
                str(e),
                exc_info=True,
            )
            return None

    async def async_get_rooms_attributes(self) -> RoomsProperties:
        """Get the rooms attributes from the JSON data.
        :return: The rooms attribute's."""
        if self.room_propriety:
            return self.room_propriety
        if self.json_data:
            LOGGER.debug("Checking %s Rooms data..", self.file_name)
            self.room_propriety = await self.async_extract_room_properties(
                self.json_data
            )
            if self.room_propriety:
                LOGGER.debug("Got %s Rooms Attributes.", self.file_name)
        return self.room_propriety

    def get_calibration_data(self) -> CalibrationPoints:
        """Get the calibration data from the JSON data.
        this will create the attribute calibration points."""
        calibration_data = []
        rotation_angle = self.shared.image_rotate
        LOGGER.info("Getting %s Calibrations points.", self.file_name)

        # Define the map points (fixed)
        map_points = self.get_map_points()
        # Calculate the calibration points in the vacuum coordinate system
        vacuum_points = self.get_vacuum_points(rotation_angle)

        # Create the calibration data for each point
        for vacuum_point, map_point in zip(vacuum_points, map_points):
            calibration_point = {"vacuum": vacuum_point, "map": map_point}
            calibration_data.append(calibration_point)
        del vacuum_points, map_points, calibration_point, rotation_angle  # free memory.
        return calibration_data

    # Element selection methods
    def enable_element(self, element_code: DrawableElement) -> None:
        """Enable drawing of a specific element."""
        self.drawing_config.enable_element(element_code)
        LOGGER.info(
            "%s: Enabled element %s, now enabled: %s",
            self.file_name,
            element_code.name,
            self.drawing_config.is_enabled(element_code),
        )

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

    @staticmethod
    async def async_copy_array(original_array):
        """Copy the array."""
        return await AsyncNumPy.async_copy(original_array)

    async def _prepare_zone_data(self, m_json):
        """Prepare zone data for parallel processing."""
        await asyncio.sleep(0)  # Yield control
        try:
            return self.data.find_zone_entities(m_json)
        except (ValueError, KeyError):
            return None

    @staticmethod
    async def _prepare_goto_data(entity_dict):
        """Prepare go-to flag data for parallel processing."""
        await asyncio.sleep(0)  # Yield control
        # Extract go-to target data from entity_dict
        return entity_dict.get("go_to_target", None)

    async def _prepare_path_data(self, m_json):
        """Prepare path data for parallel processing."""
        await asyncio.sleep(0)  # Yield control
        try:
            return self.data.find_paths_entities(m_json)
        except (ValueError, KeyError):
            return None
