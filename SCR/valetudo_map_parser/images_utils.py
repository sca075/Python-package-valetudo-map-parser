"""
Image Utils Class for Valetudo Hypfer Image Handling.
This class is used to simplify the ImageHandler class.
Version: 0.1.6
"""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)


class ImageUtils:
    """Image Utils Class for Valetudo Hypfer Image Handler.
    It is used to simplify the ImageHandler class."""

    def __init__(self, image_handler):
        self.img = image_handler
        self.file_name = self.img.shared.file_name

    def get_vacuum_points(self, rotation_angle: int) -> list[dict[str, int]]:
        """Calculate the calibration points based on the rotation angle."""

        # get_calibration_data
        vacuum_points = [
            {
                "x": self.img.crop_area[0] + self.img.offset_x,
                "y": self.img.crop_area[1] + self.img.offset_y,
            },  # Top-left corner 0
            {
                "x": self.img.crop_area[2] - self.img.offset_x,
                "y": self.img.crop_area[1] + self.img.offset_y,
            },  # Top-right corner 1
            {
                "x": self.img.crop_area[2] - self.img.offset_x,
                "y": self.img.crop_area[3] - self.img.offset_y,
            },  # Bottom-right corner 2
            {
                "x": self.img.crop_area[0] + self.img.offset_x,
                "y": self.img.crop_area[3] - self.img.offset_y,
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
                "x": ((self.img.crop_area[0] + self.img.offset_x) * 10),
                "y": ((self.img.crop_area[1] + self.img.offset_y) * 10),
            },  # Top-left corner 0
            {
                "x": ((self.img.crop_area[2] - self.img.offset_x) * 10),
                "y": ((self.img.crop_area[1] + self.img.offset_y) * 10),
            },  # Top-right corner 1
            {
                "x": ((self.img.crop_area[2] - self.img.offset_x) * 10),
                "y": ((self.img.crop_area[3] - self.img.offset_y) * 10),
            },  # Bottom-right corner 2
            {
                "x": ((self.img.crop_area[0] + self.img.offset_x) * 10),
                "y": ((self.img.crop_area[3] - self.img.offset_y) * 10),
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

    def set_image_offset_ratio_1_1(
        self, width: int, height: int, rand256: bool = False
    ) -> None:
        """Set the image offset ratio to 1:1."""

        rotation = self.img.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.img.offset_y = self.img.crop_img_size[0] - width
                self.img.offset_x = (height - self.img.crop_img_size[1]) // 2
            elif rotation in [90, 270]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = (self.img.crop_img_size[1] - height) // 2
        else:
            if rotation in [0, 180]:
                self.img.offset_x = (width - self.img.crop_img_size[0]) // 2
                self.img.offset_y = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_y = (self.img.crop_img_size[0] - width) // 2
                self.img.offset_x = self.img.crop_img_size[1] - height
        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.img.offset_x,
            self.img.offset_y,
        )

    def set_image_offset_ratio_2_1(
        self, width: int, height: int, rand256: bool = False
    ) -> None:
        """Set the image offset ratio to 2:1."""

        rotation = self.img.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_x = width - self.img.crop_img_size[0]
                self.img.offset_y = height - self.img.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_x = width - self.img.crop_img_size[0]
                self.img.offset_y = height - self.img.crop_img_size[1]

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.img.offset_x,
            self.img.offset_y,
        )

    def set_image_offset_ratio_3_2(
        self, width: int, height: int, rand256: bool = False
    ) -> None:
        """Set the image offset ratio to 3:2."""

        rotation = self.img.shared.image_rotate

        if not rand256:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = ((height - self.img.crop_img_size[1]) // 2) - (
                    self.img.crop_img_size[1] // 10
                )
            elif rotation in [90, 270]:
                self.img.offset_y = (self.img.crop_img_size[0] - width) // 2
                self.img.offset_x = (self.img.crop_img_size[1] - height) + (
                    (height // 10) // 2
                )
        else:
            if rotation in [0, 180]:
                self.img.offset_x = (width - self.img.crop_img_size[0]) // 2
                self.img.offset_y = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_y = (self.img.crop_img_size[0] - width) // 2
                self.img.offset_x = self.img.crop_img_size[1] - height

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.img.offset_x,
            self.img.offset_y,
        )

    def set_image_offset_ratio_5_4(
        self, width: int, height: int, rand256: bool = False
    ) -> None:
        """Set the image offset ratio to 5:4."""

        rotation = self.img.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.img.offset_x = ((width - self.img.crop_img_size[0]) // 2) - (
                    self.img.crop_img_size[0] // 2
                )
                self.img.offset_y = (self.img.crop_img_size[1] - height) - (
                    self.img.crop_img_size[1] // 2
                )
            elif rotation in [90, 270]:
                self.img.offset_y = ((self.img.crop_img_size[0] - width) // 2) - 10
                self.img.offset_x = (self.img.crop_img_size[1] - height) + (
                    height // 10
                )
        else:
            if rotation in [0, 180]:
                self.img.offset_y = (width - self.img.crop_img_size[0]) // 2
                self.img.offset_x = self.img.crop_img_size[1] - height
            elif rotation in [90, 270]:
                self.img.offset_y = (self.img.crop_img_size[0] - width) // 2
                self.img.offset_x = self.img.crop_img_size[1] - height

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.img.offset_x,
            self.img.offset_y,
        )

    def set_image_offset_ratio_9_16(
        self, width: int, height: int, rand256: bool = False
    ) -> None:
        """Set the image offset ratio to 9:16."""

        rotation = self.img.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_x = (width - self.img.crop_img_size[0]) + (height // 10)
                self.img.offset_y = height - self.img.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_x = width - self.img.crop_img_size[0]
                self.img.offset_y = height - self.img.crop_img_size[1]

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.img.offset_x,
            self.img.offset_y,
        )

    def set_image_offset_ratio_16_9(
        self, width: int, height: int, rand256: bool = False
    ) -> None:
        """Set the image offset ratio to 16:9."""

        rotation = self.img.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_x = width - self.img.crop_img_size[0]
                self.img.offset_y = height - self.img.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.img.offset_y = width - self.img.crop_img_size[0]
                self.img.offset_x = height - self.img.crop_img_size[1]
            elif rotation in [90, 270]:
                self.img.offset_x = width - self.img.crop_img_size[0]
                self.img.offset_y = height - self.img.crop_img_size[1]

        _LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.img.offset_x,
            self.img.offset_y,
        )

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
