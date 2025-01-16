"""Utility code for the valetudo map parser."""

import hashlib
import json
from logging import getLogger

from PIL import ImageOps

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
        self.offset_x = 0
        self.offset_y = 0
        self.shared = None
        self.crop_area = [0, 0, 0, 0]

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
            self.set_image_offset_ratio_1_1(width, height, rand256)
        elif wsf == 2 and hsf == 1:
            self.set_image_offset_ratio_2_1(width, height, rand256)
        elif wsf == 3 and hsf == 2:
            self.set_image_offset_ratio_3_2(width, height, rand256)
        elif wsf == 5 and hsf == 4:
            self.set_image_offset_ratio_5_4(width, height, rand256)
        elif wsf == 9 and hsf == 16:
            self.set_image_offset_ratio_9_16(width, height, rand256)
        elif wsf == 16 and hsf == 9:
            self.set_image_offset_ratio_16_9(width, height, rand256)
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

    def set_image_offset_ratio_1_1(
            self, width: int, height: int, rand256: bool = False
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

    def set_image_offset_ratio_2_1(
            self, width: int, height: int, rand256: bool = False
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

    def set_image_offset_ratio_3_2(
            self, width: int, height: int, rand256: bool = False
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
                self.offset_x = (self.crop_img_size[1] - height) + (
                        (height // 10) // 2
                )
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

    def set_image_offset_ratio_5_4(
            self, width: int, height: int, rand256: bool = False
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
                self.offset_x = (self.crop_img_size[1] - height) + (
                        height // 10
                )
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

    def set_image_offset_ratio_9_16(
            self, width: int, height: int, rand256: bool = False
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

    def set_image_offset_ratio_16_9(
            self, width: int, height: int, rand256: bool = False
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