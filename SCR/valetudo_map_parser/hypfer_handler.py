"""
Hypfer Image Handler Class.
It returns the PIL PNG image frame relative to the Map Data extrapolated from the vacuum json.
It also returns calibration, rooms data to the card and other images information to the camera.
Version: 0.1.9
"""

from __future__ import annotations

import json
import logging

from PIL import Image

from .config.auto_crop import AutoCrop
from .config.drawable import Drawable
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
        self.draw = Drawable  # imported Drawing utilities
        self.go_to = None  # vacuum go to data
        self.img_hash = None  # hash of the image calculated to check differences.
        self.img_base_layer = None  # numpy array store the map base layer.
        self.active_zones = None  # vacuum active zones.
        self.svg_wait = False  # SVG image creation wait.
        self.imd = ImDraw(self)  # Image Draw class.
        self.color_grey = (128, 128, 128, 255)
        self.file_name = self.shared.file_name  # file name of the vacuum.

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
                    corners = self.get_corners(x_max, x_min, y_max, y_min)
                    room_id = str(segment_id)
                    self.rooms_pos.append(
                        {
                            "name": name,
                            "corners": corners,
                        }
                    )
                    room_properties[room_id] = {
                        "number": segment_id,
                        "outline": corners,
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
                    # empty image
                    img_np_array = await self.draw.create_empty_image(
                        size_x, size_y, colors["background"]
                    )
                    # overlapping layers and segments
                    for layer_type, compressed_pixels_list in layers.items():
                        room_id, img_np_array = await self.imd.async_draw_base_layer(
                            img_np_array,
                            compressed_pixels_list,
                            layer_type,
                            colors["wall"],
                            colors["zone_clean"],
                            pixel_size,
                        )
                    # Draw the virtual walls if any.
                    img_np_array = await self.imd.async_draw_virtual_walls(
                        m_json, img_np_array, colors["no_go"]
                    )
                    # Draw charger.
                    img_np_array = await self.imd.async_draw_charger(
                        img_np_array, entity_dict, colors["charger"]
                    )
                    # Draw obstacles if any.
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
                # Draw zones if any.
                img_np_array = await self.imd.async_draw_zones(
                    m_json,
                    img_np_array,
                    colors["zone_clean"],
                    colors["no_go"],
                )
                # Draw the go_to target flag.
                img_np_array = await self.imd.draw_go_to_flag(
                    img_np_array, entity_dict, colors["go_to"]
                )
                # Draw path prediction and paths.
                img_np_array = await self.imd.async_draw_paths(
                    img_np_array, m_json, colors["move"], self.color_grey
                )
                # Check if the robot is docked.
                if self.shared.vacuum_state == "docked":
                    # Adjust the robot angle.
                    robot_position_angle -= 180

                if robot_pos:
                    # Draw the robot
                    img_np_array = await self.draw.robot(
                        layers=img_np_array,
                        x=robot_position[0],
                        y=robot_position[1],
                        angle=robot_position_angle,
                        fill=colors["robot"],
                        robot_state=self.shared.vacuum_state,
                    )
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
