"""Utility code for the valetudo map parser."""

import hashlib
import json
from dataclasses import dataclass
from logging import getLogger
from typing import Callable, List, Optional

from PIL import ImageOps

from .types import ChargerPosition, ImageSize, NumpyArray, PilPNG, RobotPosition


_LOGGER = getLogger(__name__)


@dataclass
class ResizeParams:
    """Resize the image to the given dimensions and aspect ratio."""

    pil_img: PilPNG
    width: int
    height: int
    aspect_ratio: str
    crop_size: List[int]
    is_rand: Optional[bool] = False
    offset_func: Optional[Callable] = None


@dataclass
class OffsetParams:
    """Map parameters."""

    wsf: int
    hsf: int
    width: int
    height: int
    rand256: Optional[bool] = False


class BaseHandler:
    """Avoid Code duplication"""

    def __init__(self):
        self.file_name = None
        self.shared = None
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
        self.offset_x = 0
        self.offset_y = 0
        self.crop_area = None
        self.zooming = False
        self.async_resize_images = async_resize_image

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

    def check_zoom_and_aspect_ratio(self) -> bool:
        """Check if the image is zoomed and has an aspect ratio."""
        return (
            self.shared.image_auto_zoom
            and self.shared.vacuum_state == "cleaning"
            and self.zooming
            and self.shared.image_zoom_lock_ratio
            or self.shared.image_aspect_ratio != "None"
        )

    def _set_image_offset_ratio_1_1(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 1:1."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = self.crop_img_size[0] - width
                self.offset_x = (height - self.crop_img_size[1]) // 2
            elif rotation in [90, 270]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = (self.crop_img_size[1] - height) // 2
        else:
            if rotation in [0, 180]:
                self.offset_x = (width - self.crop_img_size[0]) // 2
                self.offset_y = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = self.crop_img_size[1] - height
        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_2_1(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 2:1."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_3_2(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 3:2."""

        rotation = self.shared.image_rotate

        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = ((height - self.crop_img_size[1]) // 2) - (
                    self.crop_img_size[1] // 10
                )
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = (self.crop_img_size[1] - height) + ((height // 10) // 2)
        else:
            if rotation in [0, 180]:
                self.offset_x = (width - self.crop_img_size[0]) // 2
                self.offset_y = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = self.crop_img_size[1] - height

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_5_4(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 5:4."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_x = ((width - self.crop_img_size[0]) // 2) - (
                    self.crop_img_size[0] // 2
                )
                self.offset_y = (self.crop_img_size[1] - height) - (
                    self.crop_img_size[1] // 2
                )
            elif rotation in [90, 270]:
                self.offset_y = ((self.crop_img_size[0] - width) // 2) - 10
                self.offset_x = (self.crop_img_size[1] - height) + (height // 10)
        else:
            if rotation in [0, 180]:
                self.offset_y = (width - self.crop_img_size[0]) // 2
                self.offset_x = self.crop_img_size[1] - height
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = self.crop_img_size[1] - height

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_9_16(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 9:16."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = (width - self.crop_img_size[0]) + (height // 10)
                self.offset_y = height - self.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_16_9(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 16:9."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    async def async_map_coordinates_offset(
        self, params: OffsetParams
    ) -> tuple[int, int]:
        """
        Offset the coordinates to the map.
        """
        if params.wsf == 1 and params.hsf == 1:
            self._set_image_offset_ratio_1_1(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 2 and params.hsf == 1:
            self._set_image_offset_ratio_2_1(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 3 and params.hsf == 2:
            self._set_image_offset_ratio_3_2(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 5 and params.hsf == 4:
            self._set_image_offset_ratio_5_4(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 9 and params.hsf == 16:
            self._set_image_offset_ratio_9_16(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 16 and params.hsf == 9:
            self._set_image_offset_ratio_16_9(
                params.width, params.height, params.rand256
            )
        return params.width, params.height

    @staticmethod
    async def calculate_array_hash(
        layers: dict, active: Optional[List[int]]
    ) -> str | None:
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

    def get_map_points(
        self,
    ) -> list[dict[str, int] | dict[str, int] | dict[str, int] | dict[str, int]]:
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

    def get_vacuum_points(self, rotation_angle: int) -> list[dict[str, int]]:
        """Calculate the calibration points based on the rotation angle."""

        # get_calibration_data
        vacuum_points = [
            {
                "x": self.crop_area[0] + self.offset_x,
                "y": self.crop_area[1] + self.offset_y,
            },  # Top-left corner 0
            {
                "x": self.crop_area[2] - self.offset_x,
                "y": self.crop_area[1] + self.offset_y,
            },  # Top-right corner 1
            {
                "x": self.crop_area[2] - self.offset_x,
                "y": self.crop_area[3] - self.offset_y,
            },  # Bottom-right corner 2
            {
                "x": self.crop_area[0] + self.offset_x,
                "y": self.crop_area[3] - self.offset_y,
            },  # Bottom-left corner (optional)3
        ]

        # Rotate the vacuum points based on the rotation angle
        if rotation_angle == 90:
            vacuum_points = [
                vacuum_points[1],
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
            ]
        elif rotation_angle == 180:
            vacuum_points = [
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
            ]
        elif rotation_angle == 270:
            vacuum_points = [
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
                vacuum_points[2],
            ]

        return vacuum_points

    def re_get_vacuum_points(self, rotation_angle: int) -> list[dict[str, int]]:
        """Recalculate the calibration points based on the rotation angle.
        RAND256 Vacuums Calibration Points are in 10th of a mm."""
        vacuum_points = [
            {
                "x": ((self.crop_area[0] + self.offset_x) * 10),
                "y": ((self.crop_area[1] + self.offset_y) * 10),
            },  # Top-left corner 0
            {
                "x": ((self.crop_area[2] - self.offset_x) * 10),
                "y": ((self.crop_area[1] + self.offset_y) * 10),
            },  # Top-right corner 1
            {
                "x": ((self.crop_area[2] - self.offset_x) * 10),
                "y": ((self.crop_area[3] - self.offset_y) * 10),
            },  # Bottom-right corner 2
            {
                "x": ((self.crop_area[0] + self.offset_x) * 10),
                "y": ((self.crop_area[3] - self.offset_y) * 10),
            },  # Bottom-left corner (optional)3
        ]

        # Rotate the vacuum points based on the rotation angle
        if rotation_angle == 90:
            vacuum_points = [
                vacuum_points[1],
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
            ]
        elif rotation_angle == 180:
            vacuum_points = [
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
            ]
        elif rotation_angle == 270:
            vacuum_points = [
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
                vacuum_points[2],
            ]

        return vacuum_points

    async def async_zone_propriety(self, zones_data) -> dict:
        """Get the zone propriety"""
        zone_properties = {}
        id_count = 1
        for zone in zones_data:
            zone_name = zone.get("name")
            coordinates = zone.get("coordinates")
            if coordinates and len(coordinates) > 0:
                coordinates[0].pop()
                x1, y1, x2, y2 = coordinates[0]
                zone_properties[zone_name] = {
                    "zones": coordinates,
                    "name": zone_name,
                    "x": ((x1 + x2) // 2),
                    "y": ((y1 + y2) // 2),
                }
                id_count += 1
            if id_count > 1:
                _LOGGER.debug("%s: Zones Properties updated.", self.file_name)
        return zone_properties

    async def async_points_propriety(self, points_data) -> dict:
        """Get the point propriety"""
        point_properties = {}
        id_count = 1
        for point in points_data:
            point_name = point.get("name")
            coordinates = point.get("coordinates")
            if coordinates and len(coordinates) > 0:
                coordinates = point.get("coordinates")
                x1, y1 = coordinates
                point_properties[id_count] = {
                    "position": coordinates,
                    "name": point_name,
                    "x": x1,
                    "y": y1,
                }
                id_count += 1
            if id_count > 1:
                _LOGGER.debug("%s: Point Properties updated.", self.file_name)
        return point_properties

    @staticmethod
    def get_corners(
        x_max: int, x_min: int, y_max: int, y_min: int
    ) -> list[tuple[int, int]]:
        """Return the corners of the image."""
        return [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]


async def async_resize_image(params: ResizeParams):
    """Resize the image to the given dimensions and aspect ratio."""
    if params.aspect_ratio:
        wsf, hsf = [int(x) for x in params.aspect_ratio.split(",")]

        if wsf == 0 or hsf == 0 or params.width <= 0 or params.height <= 0:
            _LOGGER.warning(
                "Invalid aspect ratio parameters: width=%s, height=%s, wsf=%s, hsf=%s. Returning original image.",
                params.width,
                params.height,
                wsf,
                hsf,
            )
            return params.pil_img  # Return original image if invalid
        if params.width == 0:
            params.width = params.pil_img.width
        if params.height == 0:
            params.height = params.pil_img.height
        new_aspect_ratio = wsf / hsf
        if params.width / params.height > new_aspect_ratio:
            new_width = int(params.pil_img.height * new_aspect_ratio)
            new_height = params.pil_img.height
        else:
            new_width = params.pil_img.width
            new_height = int(params.pil_img.width / new_aspect_ratio)

        _LOGGER.debug("Resizing image to aspect ratio: %s, %s", wsf, hsf)
        _LOGGER.debug("New image size: %s x %s", new_width, new_height)

        if (params.crop_size is not None) and (params.offset_func is not None):
            offset = OffsetParams(wsf, hsf, new_width, new_height, params.is_rand)
            params.crop_size[0], params.crop_size[1] = await params.offset_func(offset)

        return ImageOps.pad(params.pil_img, (new_width, new_height))

    return ImageOps.pad(params.pil_img, (params.width, params.height))


def prepare_resize_params(handler, pil_img, rand):
    """Prepare resize parameters for image resizing."""
    return ResizeParams(
        pil_img=pil_img,
        width=handler.shared.image_ref_width,
        height=handler.shared.image_ref_height,
        aspect_ratio=handler.shared.image_aspect_ratio,
        crop_size=handler.crop_img_size,
        offset_func=handler.async_map_coordinates_offset,
        is_rand=rand,
    )
