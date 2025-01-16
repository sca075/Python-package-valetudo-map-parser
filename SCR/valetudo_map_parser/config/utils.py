"""Utility code for the valetudo map parser."""

import hashlib
import json
from logging import getLogger

from PIL import ImageOps

from ..images_utils import ImageUtils as ImUtils
from .types import ChargerPosition, ImageSize, NumpyArray, RobotPosition

_LOGGER = getLogger(__name__)


class BaseHandler:
    """Avoid Code duplication"""

    def __init__(self):
        self.file_name = None
        self.img_size = None
        self.json_data = None
        self.json_id = None
        self.path_pixels = None
        self.robot_in_room = None
        self.robot_pos = None
        self.room_propriety = None
        self.rooms_pos = None
        self.charger_pos = None
        self.frame_number = 0
        self.max_frames = 1024
        self.crop_img_size = [0, 0]
        self.imu = ImUtils(self)  # Image Utils

    def get_frame_number(self) -> int:
        """Return the frame number of the image."""
        return self.frame_number

    def get_robot_position(self) -> RobotPosition | None:
        """Return the robot position."""
        return self.robot_pos

    def get_charger_position(self) -> ChargerPosition | None:
        """Return the charger position."""
        return self.charger_pos

    def get_img_size(self) -> ImageSize | None:
        """Return the size of the image."""
        return self.img_size

    def get_json_id(self) -> str | None:
        """Return the JSON ID from the image."""
        return self.json_id

    async def async_resize_image(
        self, pil_img, width, height, aspect_ratio=None, is_rand=False
    ):
        """Resize the image to the given dimensions and aspect ratio."""
        if aspect_ratio:
            wsf, hsf = [int(x) for x in aspect_ratio.split(",")]
            if wsf == 0 or hsf == 0:
                return pil_img
            new_aspect_ratio = wsf / hsf
            if width / height > new_aspect_ratio:
                new_width = int(pil_img.height * new_aspect_ratio)
                new_height = pil_img.height
            else:
                new_width = pil_img.width
                new_height = int(pil_img.width / new_aspect_ratio)
                _LOGGER.debug(
                    "%s: Image Aspect Ratio: %s, %s",
                    self.file_name,
                    str(wsf),
                    str(hsf),
                )
                (
                    self.crop_img_size[0],
                    self.crop_img_size[1],
                ) = await self.async_map_coordinates_offset(
                    wsf, hsf, new_width, new_height, is_rand
                )
            return ImageOps.pad(pil_img, (new_width, new_height))
        return ImageOps.pad(pil_img, (width, height))

    async def async_map_coordinates_offset(
        self, wsf: int, hsf: int, width: int, height: int, rand256: bool = False
    ) -> tuple[int, int]:
        """
        Offset the coordinates to the map.
        """

        if wsf == 1 and hsf == 1:
            self.imu.set_image_offset_ratio_1_1(width, height, rand256)
        elif wsf == 2 and hsf == 1:
            self.imu.set_image_offset_ratio_2_1(width, height, rand256)
        elif wsf == 3 and hsf == 2:
            self.imu.set_image_offset_ratio_3_2(width, height, rand256)
        elif wsf == 5 and hsf == 4:
            self.imu.set_image_offset_ratio_5_4(width, height, rand256)
        elif wsf == 9 and hsf == 16:
            self.imu.set_image_offset_ratio_9_16(width, height, rand256=True)
        elif wsf == 16 and hsf == 9:
            self.imu.set_image_offset_ratio_16_9(width, height, rand256=True)
        return width, height

    @staticmethod
    async def calculate_array_hash(layers: dict, active: list[int] = None) -> str:
        """Calculate the hash of the image based on layers and active zones."""
        if layers and active:
            data_to_hash = {
                "layers": len(layers["wall"][0]),
                "active_segments": tuple(active),
            }
            data_json = json.dumps(data_to_hash, sort_keys=True)
            return hashlib.sha256(data_json.encode()).hexdigest()
        return None

    @staticmethod
    async def async_copy_array(original_array: NumpyArray) -> NumpyArray:
        """Copy the array."""
        return NumpyArray.copy(original_array)

    def get_map_points(self) -> dict:
        """Return the map points."""
        return [
            {"x": 0, "y": 0},  # Top-left corner 0
            {"x": self.crop_img_size[0], "y": 0},  # Top-right corner 1
            {
                "x": self.crop_img_size[0],
                "y": self.crop_img_size[1],
            },  # Bottom-right corner 2
            {"x": 0, "y": self.crop_img_size[1]},  # Bottom-left corner (optional) 3
        ]
