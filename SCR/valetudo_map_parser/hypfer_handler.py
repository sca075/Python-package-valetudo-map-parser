"""
Hypfer Image Handler Class.
It returns the PIL PNG image frame relative to the Map Data extrapolated from the vacuum json.
It also returns calibration, rooms data to the card and other images information to the camera.
Version: 0.1.9
"""

from __future__ import annotations

import json
import logging

import numpy as np
from PIL import Image

from .config.auto_crop import AutoCrop
from .config.drawable import Drawable
from .config.drawable_elements import DrawableElement, DrawingConfig
from .config.enhanced_drawable import EnhancedDrawable
from .config.shared import CameraShared
from .config.types import COLORS, CalibrationPoints, Colors, RoomsProperties, RoomStore
from .config.utils import BaseHandler, prepare_resize_params
from .hypfer_draw import ImageDraw as ImDraw
from .map_data import ImageData


_LOGGER = logging.getLogger(__name__)


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

        # Initialize drawing configuration
        self.drawing_config = DrawingConfig()
        if hasattr(self.shared, 'device_info') and self.shared.device_info is not None:
            _LOGGER.info("%s: Initializing drawing config from device_info", self.file_name)
            _LOGGER.info("%s: device_info contains disable_obstacles: %s", self.file_name, 'disable_obstacles' in self.shared.device_info)
            _LOGGER.info("%s: device_info contains disable_path: %s", self.file_name, 'disable_path' in self.shared.device_info)
            _LOGGER.info("%s: device_info contains disable_elements: %s", self.file_name, 'disable_elements' in self.shared.device_info)

            if 'disable_obstacles' in self.shared.device_info:
                _LOGGER.info("%s: disable_obstacles value: %s", self.file_name, self.shared.device_info['disable_obstacles'])
            if 'disable_path' in self.shared.device_info:
                _LOGGER.info("%s: disable_path value: %s", self.file_name, self.shared.device_info['disable_path'])
            if 'disable_elements' in self.shared.device_info:
                _LOGGER.info("%s: disable_elements value: %s", self.file_name, self.shared.device_info['disable_elements'])

            self.drawing_config.update_from_device_info(self.shared.device_info)

            # Verify elements are disabled
            _LOGGER.info("%s: After initialization, PATH enabled: %s", self.file_name, self.drawing_config.is_enabled(DrawableElement.PATH))
            _LOGGER.info("%s: After initialization, OBSTACLE enabled: %s", self.file_name, self.drawing_config.is_enabled(DrawableElement.OBSTACLE))

        # Initialize both drawable systems for backward compatibility
        self.draw = Drawable()  # Legacy drawing utilities
        self.enhanced_draw = EnhancedDrawable(self.drawing_config)  # New enhanced drawing system

        self.go_to = None  # vacuum go to data
        self.img_hash = None  # hash of the image calculated to check differences.
        self.img_base_layer = None  # numpy array store the map base layer.
        self.active_zones = None  # vacuum active zones.
        self.svg_wait = False  # SVG image creation wait.
        self.imd = ImDraw(self)  # Image Draw class.
        self.color_grey = (128, 128, 128, 255)
        self.file_name = self.shared.file_name  # file name of the vacuum.
        self.element_map = None  # Map of element codes

    def get_corners(self, x_max, x_min, y_max, y_min):
        """Get the corners of the room."""
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    async def extract_room_outline_from_map(self, room_id_int, pixels, pixel_size):
        """Extract the outline of a room using the pixel data and element map.

        Args:
            room_id_int: The room ID as an integer
            pixels: List of pixel coordinates in the format [[x, y, z], ...]
            pixel_size: Size of each pixel

        Returns:
            List of points forming the outline of the room
        """
        # Calculate x and y min/max from compressed pixels for rectangular fallback
        x_values = []
        y_values = []
        for x, y, z in pixels:
            for i in range(z):
                x_values.append(x + i * pixel_size)
                y_values.append(y)

        if not x_values or not y_values:
            return []

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        # If we don't have an element map, return a rectangular outline
        if not hasattr(self, 'element_map') or self.element_map is None:
            # Return rectangular outline
            return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        # Create a binary mask for this room
        height, width = self.element_map.shape
        room_mask = np.zeros((height, width), dtype=np.uint8)

        # Get the DrawableElement for this room
        from .config.drawable_elements import DrawableElement
        room_element = getattr(DrawableElement, f"ROOM_{room_id_int}", None)

        # Fill the mask with room pixels
        if room_element:
            # Use the element map to identify room pixels
            room_mask[self.element_map == room_element] = 1
        else:
            # Fall back to using the pixel data if element map doesn't have room info
            for x, y, z in pixels:
                for i in range(z):
                    px = x + i * pixel_size
                    py = y
                    # Make sure we're within bounds
                    if 0 <= py < height and 0 <= px < width:
                        # Mark a pixel_size x pixel_size block in the mask
                        for dx in range(pixel_size):
                            for dy in range(pixel_size):
                                if py + dy < height and px + dx < width:
                                    room_mask[py + dy, px + dx] = 1

        # Use OpenCV's contour finding algorithm if available
        try:
            import cv2
            # Find contours in the binary mask
            contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours found, return rectangular outline
            if not contours:
                return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

            # Get the largest contour (main room outline)
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplify the contour to reduce the number of points
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Convert the contour to a list of (x, y) points
            outline = [(point[0][0], point[0][1]) for point in approx_contour]

            # Ensure the outline has a reasonable number of points
            if len(outline) > 20:
                # Subsample the outline to reduce the number of points
                step = len(outline) // 20
                outline = [outline[i] for i in range(0, len(outline), step)]

            return outline

        except (ImportError, NameError):
            # Fall back to a simpler algorithm if OpenCV is not available
            # Find the coordinates of all room pixels
            room_y, room_x = np.where(room_mask > 0)
            if len(room_y) == 0:
                return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

            # Get the bounding box of the room
            min_y, max_y = np.min(room_y), np.max(room_y)
            min_x, max_x = np.min(room_x), np.max(room_x)

            # Create a simplified outline using the bounding box corners
            outline = [
                (min_x, min_y),  # Top-left
                (max_x, min_y),  # Top-right
                (max_x, max_y),  # Bottom-right
                (min_x, max_y),  # Bottom-left
            ]

            return outline

    async def async_extract_room_properties(self, json_data) -> RoomsProperties:
        """Extract room properties from the JSON data."""

        room_properties = {}
        self.rooms_pos = []
        pixel_size = json_data.get("pixelSize", [])

        for layer in json_data.get("layers", []):
            if layer["__class"] == "MapLayer":
                meta_data = layer.get("metaData", {})
                segment_id = meta_data.get("segmentId")
                if segment_id is not None:
                    name = meta_data.get("name")
                    compressed_pixels = layer.get("compressedPixels", [])
                    pixels = self.data.sublist(compressed_pixels, 3)
                    # Calculate x and y min/max from compressed pixels
                    (
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                    ) = await self.data.async_get_rooms_coordinates(pixels, pixel_size)

                    # Get rectangular corners as a fallback
                    corners = self.get_corners(x_max, x_min, y_max, y_min)

                    # Try to extract a more accurate room outline from the element map
                    try:
                        # Extract the room outline using the element map
                        outline = await self.extract_room_outline_from_map(segment_id, pixels, pixel_size)
                        _LOGGER.debug(
                            "%s: Traced outline for room %s with %d points",
                            self.file_name, segment_id, len(outline)
                        )
                    except Exception as e:
                        _LOGGER.warning(
                            "%s: Failed to trace outline for room %s: %s",
                            self.file_name, segment_id, str(e)
                        )
                        outline = corners

                    room_id = str(segment_id)
                    self.rooms_pos.append(
                        {
                            "name": name,
                            "corners": corners,
                        }
                    )
                    room_properties[room_id] = {
                        "number": segment_id,
                        "outline": outline,  # Use the detailed outline from the element map
                        "name": name,
                        "x": ((x_min + x_max) // 2),
                        "y": ((y_min + y_max) // 2),
                    }
        if room_properties:
            rooms = RoomStore(self.file_name, room_properties)
            _LOGGER.debug(
                "%s: Rooms data extracted! %s", self.file_name, rooms.get_rooms()
            )
        else:
            _LOGGER.debug("%s: Rooms data not available!", self.file_name)
            self.rooms_pos = None
        return room_properties

    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    async def async_get_image_from_json(
        self,
        m_json: json | None,
    ) -> Image.Image | None:
        """Get the image from the JSON data.
        It uses the ImageDraw class to draw some of the elements of the image.
        The robot itself will be drawn in this function as per some of the values are needed for other tasks.
        @param m_json: The JSON data to use to draw the image.
        @return Image.Image: The image to display.
        """
        # Initialize the colors.
        colors: Colors = {
            name: self.shared.user_colors[idx] for idx, name in enumerate(COLORS)
        }
        # Check if the JSON data is not None else process the image.
        try:
            if m_json is not None:
                _LOGGER.debug("%s: Creating Image.", self.file_name)
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
                new_frame_hash = await self.calculate_array_hash(layers, active)
                if self.frame_number == 0:
                    self.img_hash = new_frame_hash
                    # Create empty image
                    img_np_array = await self.draw.create_empty_image(
                        size_x, size_y, colors["background"]
                    )

                    # Create element map for tracking what's drawn where
                    self.element_map = np.zeros((size_y, size_x), dtype=np.int32)
                    self.element_map[:] = DrawableElement.FLOOR

                    _LOGGER.info("%s: Drawing map with color blending", self.file_name)

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
                                if current_room_id >= 1 and current_room_id <= 15:
                                    room_element = getattr(DrawableElement, f"ROOM_{current_room_id}", None)
                                    if room_element and not self.drawing_config.is_enabled(room_element):
                                        # Add this room to the disabled rooms set
                                        disabled_rooms.add(room_id)
                                        _LOGGER.debug("%s: Room %d is disabled and will be skipped",
                                                    self.file_name, current_room_id)
                                room_id = (room_id + 1) % 16  # Cycle room_id back to 0 after 15

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
                                if current_room_id >= 1 and current_room_id <= 15:
                                    room_element = getattr(DrawableElement, f"ROOM_{current_room_id}", None)
                                    if room_element:
                                        # Log the room check for debugging
                                        _LOGGER.debug("%s: Checking if room %d is enabled: %s",
                                                    self.file_name, current_room_id,
                                                    self.drawing_config.is_enabled(room_element))

                                        # Skip this room if it's disabled
                                        if not self.drawing_config.is_enabled(room_element):
                                            room_id = (room_id + 1) % 16  # Increment room_id even if we skip
                                            continue

                            # Check if this is a wall layer and if walls are enabled
                            is_wall_layer = layer_type == "wall"
                            if is_wall_layer:
                                if not self.drawing_config.is_enabled(DrawableElement.WALL):
                                    _LOGGER.info("%s: Skipping wall layer because WALL element is disabled", self.file_name)
                                    continue

                                # Filter out walls for disabled rooms
                                if disabled_rooms:
                                    # We need to modify the compressed_pixels_list to exclude walls of disabled rooms
                                    # This is a complex task that would require knowledge of which walls belong to which rooms
                                    # For now, we'll just log that we're drawing walls for all rooms
                                    _LOGGER.debug("%s: Drawing walls for all rooms (including disabled ones)", self.file_name)
                                    # In a real implementation, we would filter the walls here

                            # Draw the layer
                            room_id, img_np_array = await self.imd.async_draw_base_layer(
                                img_np_array,
                                compressed_pixels_list,
                                layer_type,
                                colors["wall"],
                                colors["zone_clean"],
                                pixel_size,
                                disabled_rooms if layer_type == "wall" else None,
                            )

                            # Update element map for this layer
                            if is_room_layer and room_id > 0 and room_id <= 15:
                                # Mark the room in the element map
                                room_element = getattr(DrawableElement, f"ROOM_{room_id}", None)
                                if room_element:
                                    # This is a simplification - in a real implementation we would
                                    # need to identify the exact pixels that belong to this room
                                    pass

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
                        img_np_array = await self.imd.async_draw_obstacle(
                            img_np_array, entity_dict, colors["no_go"]
                        )
                    # Robot and rooms position
                    if (room_id > 0) and not self.room_propriety:
                        self.room_propriety = await self.async_extract_room_properties(
                            self.json_data
                        )
                        if self.rooms_pos and robot_position and robot_position_angle:
                            self.robot_pos = await self.imd.async_get_robot_in_room(
                                robot_x=(robot_position[0]),
                                robot_y=(robot_position[1]),
                                angle=robot_position_angle,
                            )
                    _LOGGER.info("%s: Completed base Layers", self.file_name)
                    # Copy the new array in base layer.
                    self.img_base_layer = await self.async_copy_array(img_np_array)
                self.shared.frame_number = self.frame_number
                self.frame_number += 1
                if (self.frame_number >= self.max_frames) or (
                    new_frame_hash != self.img_hash
                ):
                    self.frame_number = 0
                _LOGGER.debug(
                    "%s: %s at Frame Number: %s",
                    self.file_name,
                    str(self.json_id),
                    str(self.frame_number),
                )
                # Copy the base layer to the new image.
                img_np_array = await self.async_copy_array(self.img_base_layer)
                # All below will be drawn at each frame.
                # Draw zones if any and if enabled
                if self.drawing_config.is_enabled(DrawableElement.RESTRICTED_AREA):
                    img_np_array = await self.imd.async_draw_zones(
                        m_json,
                        img_np_array,
                        colors["zone_clean"],
                        colors["no_go"],
                    )

                # Draw the go_to target flag if enabled
                if self.drawing_config.is_enabled(DrawableElement.GO_TO_TARGET):
                    img_np_array = await self.imd.draw_go_to_flag(
                        img_np_array, entity_dict, colors["go_to"]
                    )

                # Draw path prediction and paths if enabled
                path_enabled = self.drawing_config.is_enabled(DrawableElement.PATH)
                _LOGGER.info("%s: PATH element enabled: %s", self.file_name, path_enabled)
                if path_enabled:
                    _LOGGER.info("%s: Drawing path", self.file_name)
                    img_np_array = await self.imd.async_draw_paths(
                        img_np_array, m_json, colors["move"], self.color_grey
                    )
                else:
                    _LOGGER.info("%s: Skipping path drawing", self.file_name)

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
                    robot_radius = 25  # Same as in the robot drawing method
                    for dy in range(-robot_radius, robot_radius + 1):
                        for dx in range(-robot_radius, robot_radius + 1):
                            if dx*dx + dy*dy <= robot_radius*robot_radius:
                                rx, ry = int(robot_position[0] + dx), int(robot_position[1] + dy)
                                if 0 <= ry < self.element_map.shape[0] and 0 <= rx < self.element_map.shape[1]:
                                    self.element_map[ry, rx] = DrawableElement.ROBOT
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
                _LOGGER.warning("%s: Image array is None.", self.file_name)
                return None

            # Convert the numpy array to a PIL image
            pil_img = Image.fromarray(img_np_array, mode="RGBA")
            del img_np_array
            # reduce the image size if the zoomed image is bigger then the original.
            if self.check_zoom_and_aspect_ratio():
                resize_params = prepare_resize_params(self, pil_img, False)
                resized_image = await self.async_resize_images(resize_params)
                return resized_image
            _LOGGER.debug("%s: Frame Completed.", self.file_name)
            return pil_img
        except (RuntimeError, RuntimeWarning) as e:
            _LOGGER.warning(
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
            _LOGGER.debug("Checking %s Rooms data..", self.file_name)
            self.room_propriety = await self.async_extract_room_properties(
                self.json_data
            )
            if self.room_propriety:
                _LOGGER.debug("Got %s Rooms Attributes.", self.file_name)
        return self.room_propriety

    def get_calibration_data(self) -> CalibrationPoints:
        """Get the calibration data from the JSON data.
        this will create the attribute calibration points."""
        calibration_data = []
        rotation_angle = self.shared.image_rotate
        _LOGGER.info("Getting %s Calibrations points.", self.file_name)

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
        _LOGGER.info("%s: Enabled element %s, now enabled: %s", self.file_name, element_code.name, self.drawing_config.is_enabled(element_code))

    def disable_element(self, element_code: DrawableElement) -> None:
        """Disable drawing of a specific element."""
        self.drawing_config.disable_element(element_code)

    def set_elements(self, element_codes: list[DrawableElement]) -> None:
        """Enable only the specified elements, disable all others."""
        self.drawing_config.set_elements(element_codes)

    def set_element_property(self, element_code: DrawableElement, property_name: str, value) -> None:
        """Set a drawing property for an element."""
        self.drawing_config.set_property(element_code, property_name, value)

    def get_element_at_position(self, x: int, y: int) -> DrawableElement:
        """Get the element code at a specific position."""
        if self.element_map is not None:
            if 0 <= y < self.element_map.shape[0] and 0 <= x < self.element_map.shape[1]:
                return self.element_map[y, x]
        return None

    def get_room_at_position(self, x: int, y: int) -> int:
        """Get the room ID at a specific position, or None if not a room."""
        element = self.get_element_at_position(x, y)
        if element is not None and DrawableElement.ROOM_1 <= element <= DrawableElement.ROOM_15:
            return element - DrawableElement.ROOM_1 + 1
        return None

    @staticmethod
    def blend_colors(base_color, overlay_color):
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
            int(max(0, min(255, a_out * 255)))
        )

    def blend_pixel(self, array, x, y, color, element):
        """
        Blend a pixel color with the existing color at the specified position.
        Also updates the element map if the new element has higher z-index.

        Args:
            array: The image array to modify
            x, y: Pixel coordinates
            color: RGBA color to blend
            element: The element being drawn
        """
        # Check bounds
        if not (0 <= y < array.shape[0] and 0 <= x < array.shape[1]):
            return

        # Get current element at this position
        if self.element_map is not None:
            current_element = self.element_map[y, x]

            # Get z-indices for comparison
            current_z = self.drawing_config.get_property(current_element, "z_index", 0) if current_element else 0
            new_z = self.drawing_config.get_property(element, "z_index", 0)
        else:
            current_element = None
            current_z = 0
            new_z = 0

        # Get current color at this position
        current_color = tuple(array[y, x])

        # Blend colors
        blended_color = self.blend_colors(current_color, color)

        # Update pixel color
        array[y, x] = blended_color

        # Update element map if new element has higher z-index
        if self.element_map is not None and new_z >= current_z:
            self.element_map[y, x] = element

    async def async_copy_array(self, array):
        """Copy the array."""
        return array.copy()
