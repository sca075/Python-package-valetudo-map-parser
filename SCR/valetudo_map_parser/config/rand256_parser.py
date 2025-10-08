"""New Rand256 Map Parser -
Based on Xiaomi/Roborock implementation with precise binary parsing."""

import math
import struct
from enum import Enum
from typing import Any, Dict, List, Optional


class RRMapParser:
    """New Rand256 Map Parser using Xiaomi/Roborock approach for precise data extraction."""

    class Types(Enum):
        """Map data block types."""

        CHARGER_LOCATION = 1
        IMAGE = 2
        PATH = 3
        GOTO_PATH = 4
        GOTO_PREDICTED_PATH = 5
        CURRENTLY_CLEANED_ZONES = 6
        GOTO_TARGET = 7
        ROBOT_POSITION = 8
        FORBIDDEN_ZONES = 9
        VIRTUAL_WALLS = 10
        CURRENTLY_CLEANED_BLOCKS = 11
        FORBIDDEN_MOP_ZONES = 12
        OBSTACLES = 13
        IGNORED_OBSTACLES = 14
        OBSTACLES_WITH_PHOTO = 15
        IGNORED_OBSTACLES_WITH_PHOTO = 16
        CARPET_MAP = 17
        MOP_PATH = 18
        NO_CARPET_AREAS = 19
        DIGEST = 1024

    class Tools:
        """Tools for coordinate transformations."""

        DIMENSION_PIXELS = 1024
        DIMENSION_MM = 50 * 1024

    def __init__(self):
        """Initialize the parser."""
        self.is_valid = False
        self.map_data: Dict[str, Any] = {}

    # Xiaomi/Roborock style byte extraction methods
    @staticmethod
    def _get_bytes(data: bytes, start_index: int, size: int) -> bytes:
        """Extract bytes from data."""
        return data[start_index : start_index + size]

    @staticmethod
    def _get_int8(data: bytes, address: int) -> int:
        """Get an 8-bit integer from data using Xiaomi method."""
        return data[address] & 0xFF

    @staticmethod
    def _get_int16(data: bytes, address: int) -> int:
        """Get a 16-bit little-endian integer using Xiaomi method."""
        return ((data[address + 0] << 0) & 0xFF) | ((data[address + 1] << 8) & 0xFFFF)

    @staticmethod
    def _get_int32(data: bytes, address: int) -> int:
        """Get a 32-bit little-endian integer using Xiaomi method."""
        return (
            ((data[address + 0] << 0) & 0xFF)
            | ((data[address + 1] << 8) & 0xFFFF)
            | ((data[address + 2] << 16) & 0xFFFFFF)
            | ((data[address + 3] << 24) & 0xFFFFFFFF)
        )

    @staticmethod
    def _get_int32_signed(data: bytes, address: int) -> int:
        """Get a 32-bit signed integer."""
        value = RRMapParser._get_int32(data, address)
        return value if value < 0x80000000 else value - 0x100000000

    @staticmethod
    def _parse_carpet_map(data: bytes) -> set[int]:
        """Parse carpet map using Xiaomi method."""
        carpet_map = set()

        for i, v in enumerate(data):
            if v:
                carpet_map.add(i)
        return carpet_map

    @staticmethod
    def _parse_area(header: bytes, data: bytes) -> list:
        """Parse area using Xiaomi method."""
        area_pairs = RRMapParser._get_int16(header, 0x08)
        areas = []
        for area_start in range(0, area_pairs * 16, 16):
            x0 = RRMapParser._get_int16(data, area_start + 0)
            y0 = RRMapParser._get_int16(data, area_start + 2)
            x1 = RRMapParser._get_int16(data, area_start + 4)
            y1 = RRMapParser._get_int16(data, area_start + 6)
            x2 = RRMapParser._get_int16(data, area_start + 8)
            y2 = RRMapParser._get_int16(data, area_start + 10)
            x3 = RRMapParser._get_int16(data, area_start + 12)
            y3 = RRMapParser._get_int16(data, area_start + 14)
            areas.append(
                [
                    x0,
                    RRMapParser.Tools.DIMENSION_MM - y0,
                    x1,
                    RRMapParser.Tools.DIMENSION_MM - y1,
                    x2,
                    RRMapParser.Tools.DIMENSION_MM - y2,
                    x3,
                    RRMapParser.Tools.DIMENSION_MM - y3,
                ]
            )
        return areas

    @staticmethod
    def _parse_zones(data: bytes, header: bytes) -> list:
        """Parse zones using Xiaomi method."""
        zone_pairs = RRMapParser._get_int16(header, 0x08)
        zones = []
        for zone_start in range(0, zone_pairs * 8, 8):
            x0 = RRMapParser._get_int16(data, zone_start + 0)
            y0 = RRMapParser._get_int16(data, zone_start + 2)
            x1 = RRMapParser._get_int16(data, zone_start + 4)
            y1 = RRMapParser._get_int16(data, zone_start + 6)
            zones.append(
                [
                    x0,
                    RRMapParser.Tools.DIMENSION_MM - y0,
                    x1,
                    RRMapParser.Tools.DIMENSION_MM - y1,
                ]
            )
        return zones

    @staticmethod
    def _parse_object_position(block_data_length: int, data: bytes) -> Dict[str, Any]:
        """Parse object position using Xiaomi method."""
        x = RRMapParser._get_int32(data, 0x00)
        y = RRMapParser._get_int32(data, 0x04)
        angle = 0
        if block_data_length > 8:
            raw_angle = RRMapParser._get_int32(data, 0x08)
            # Apply Xiaomi angle normalization
            if raw_angle > 0xFF:
                angle = (raw_angle & 0xFF) - 256
            else:
                angle = raw_angle
        return {"position": [x, y], "angle": angle}

    @staticmethod
    def _parse_walls(data: bytes, header: bytes) -> list:
        """Parse walls using Xiaomi method."""
        wall_pairs = RRMapParser._get_int16(header, 0x08)
        walls = []
        for wall_start in range(0, wall_pairs * 8, 8):
            x0 = RRMapParser._get_int16(data, wall_start + 0)
            y0 = RRMapParser._get_int16(data, wall_start + 2)
            x1 = RRMapParser._get_int16(data, wall_start + 4)
            y1 = RRMapParser._get_int16(data, wall_start + 6)
            walls.append(
                [
                    x0,
                    RRMapParser.Tools.DIMENSION_MM - y0,
                    x1,
                    RRMapParser.Tools.DIMENSION_MM - y1,
                ]
            )
        return walls

    @staticmethod
    def _parse_path_block(buf: bytes, offset: int, length: int) -> Dict[str, Any]:
        """Parse path block using EXACT same method as working parser."""
        points = [
            [
                struct.unpack("<H", buf[offset + 20 + i : offset + 22 + i])[0],
                struct.unpack("<H", buf[offset + 22 + i : offset + 24 + i])[0],
            ]
            for i in range(0, length, 4)
        ]
        return {
            "current_angle": struct.unpack("<I", buf[offset + 16 : offset + 20])[0],
            "points": points,
        }

    @staticmethod
    def _parse_goto_target(data: bytes) -> List[int]:
        """Parse goto target using Xiaomi method."""
        try:
            x = RRMapParser._get_int16(data, 0x00)
            y = RRMapParser._get_int16(data, 0x02)
            return [x, y]
        except (struct.error, IndexError):
            return [0, 0]

    def parse(self, map_buf: bytes) -> Dict[str, Any]:
        """Parse the map header data using Xiaomi method."""
        if len(map_buf) < 18 or map_buf[0:2] != b"rr":
            return {}

        try:
            return {
                "header_length": self._get_int16(map_buf, 0x02),
                "data_length": self._get_int16(map_buf, 0x04),
                "version": {
                    "major": self._get_int16(map_buf, 0x08),
                    "minor": self._get_int16(map_buf, 0x0A),
                },
                "map_index": self._get_int32(map_buf, 0x0C),
                "map_sequence": self._get_int32(map_buf, 0x10),
            }
        except (struct.error, IndexError):
            return {}

    def parse_blocks(self, raw: bytes, pixels: bool = True) -> Dict[int, Any]:
        """Parse all blocks using Xiaomi method."""
        blocks = {}
        map_header_length = self._get_int16(raw, 0x02)
        block_start_position = map_header_length
        while block_start_position < len(raw):
            try:
                block_header_length = self._get_int16(raw, block_start_position + 0x02)
                header = self._get_bytes(raw, block_start_position, block_header_length)
                block_type = self._get_int16(header, 0x00)
                block_data_length = self._get_int32(header, 0x04)
                block_data_start = block_start_position + block_header_length
                data = self._get_bytes(raw, block_data_start, block_data_length)
                match block_type:
                    case self.Types.DIGEST.value:
                        self.is_valid = True
                    case (
                        self.Types.ROBOT_POSITION.value
                        | self.Types.CHARGER_LOCATION.value
                    ):
                        blocks[block_type] = self._parse_object_position(
                            block_data_length, data
                        )
                    case self.Types.PATH.value | self.Types.GOTO_PREDICTED_PATH.value:
                        blocks[block_type] = self._parse_path_block(
                            raw, block_start_position, block_data_length
                        )
                    case self.Types.CURRENTLY_CLEANED_ZONES.value:
                        blocks[block_type] = {"zones": self._parse_zones(data, header)}
                    case self.Types.FORBIDDEN_ZONES.value:
                        blocks[block_type] = {
                            "forbidden_zones": self._parse_area(header, data)
                        }
                    case self.Types.FORBIDDEN_MOP_ZONES.value:
                        blocks[block_type] = {
                            "forbidden_mop_zones": self._parse_area(header, data)
                        }
                    case self.Types.GOTO_TARGET.value:
                        blocks[block_type] = {"position": self._parse_goto_target(data)}
                    case self.Types.VIRTUAL_WALLS.value:
                        blocks[block_type] = {
                            "virtual_walls": self._parse_walls(data, header)
                        }
                    case self.Types.CARPET_MAP.value:
                        data = RRMapParser._get_bytes(
                            raw, block_data_start, block_data_length
                        )
                        blocks[block_type] = {
                            "carpet_map": self._parse_carpet_map(data)
                        }
                    case self.Types.IMAGE.value:
                        header_length = self._get_int8(header, 2)
                        blocks[block_type] = self._parse_image_block(
                            raw,
                            block_start_position,
                            block_data_length,
                            header_length,
                            pixels,
                        )

                block_start_position = (
                    block_start_position + block_data_length + self._get_int8(header, 2)
                )
            except (struct.error, IndexError):
                break
        return blocks

    def _parse_image_block(
        self, buf: bytes, offset: int, length: int, hlength: int, pixels: bool = True
    ) -> Dict[str, Any]:
        """Parse image block using EXACT logic from working parser."""
        try:
            # CRITICAL: Gen1 vs Gen3 detection like working parser
            g3offset = 4 if hlength > 24 else 0

            # Use EXACT same structure as working parser
            parameters = {
                "segments": {
                    "count": (
                        struct.unpack("<i", buf[offset + 8 : offset + 12])[0]
                        if g3offset
                        else 0
                    ),
                    "id": [],
                },
                "position": {
                    "top": struct.unpack(
                        "<i", buf[offset + 8 + g3offset : offset + 12 + g3offset]
                    )[0],
                    "left": struct.unpack(
                        "<i", buf[offset + 12 + g3offset : offset + 16 + g3offset]
                    )[0],
                },
                "dimensions": {
                    "height": struct.unpack(
                        "<i", buf[offset + 16 + g3offset : offset + 20 + g3offset]
                    )[0],
                    "width": struct.unpack(
                        "<i", buf[offset + 20 + g3offset : offset + 24 + g3offset]
                    )[0],
                },
                "pixels": {"floor": [], "walls": [], "segments": {}},
            }

            # Apply EXACT working parser coordinate transformation
            parameters["position"]["top"] = (
                self.Tools.DIMENSION_PIXELS
                - parameters["position"]["top"]
                - parameters["dimensions"]["height"]
            )

            # Extract pixels using optimized sequential processing
            if (
                parameters["dimensions"]["height"] > 0
                and parameters["dimensions"]["width"] > 0
            ):
                # Process data sequentially - segments are organized as blocks
                current_segments = {}

                for i in range(length):
                    pixel_byte = struct.unpack(
                        "<B",
                        buf[offset + 24 + g3offset + i : offset + 25 + g3offset + i],
                    )[0]

                    segment_type = pixel_byte & 0x07
                    if segment_type == 0:
                        continue

                    if segment_type == 1 and pixels:
                        # Wall pixel
                        parameters["pixels"]["walls"].append(i)
                    else:
                        # Floor or room segment
                        segment_id = pixel_byte >> 3
                        if segment_id == 0 and pixels:
                            # Floor pixel
                            parameters["pixels"]["floor"].append(i)
                        elif segment_id != 0:
                            # Room segment - segments are sequential blocks
                            if segment_id not in current_segments:
                                parameters["segments"]["id"].append(segment_id)
                                parameters["segments"][
                                    "pixels_seg_" + str(segment_id)
                                ] = []
                                current_segments[segment_id] = True

                            if pixels:
                                parameters["segments"][
                                    "pixels_seg_" + str(segment_id)
                                ].append(i)

            parameters["segments"]["count"] = len(parameters["segments"]["id"])
            return parameters

        except (struct.error, IndexError):
            return {
                "segments": {"count": 0, "id": []},
                "position": {"top": 0, "left": 0},
                "dimensions": {"height": 0, "width": 0},
                "pixels": {"floor": [], "walls": [], "segments": {}},
            }

    def parse_rrm_data(
        self, map_buf: bytes, pixels: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Parse the complete map data and return in your JSON format."""
        if not self.parse(map_buf).get("map_index"):
            return None

        try:
            parsed_map_data = {}
            blocks = self.parse_blocks(map_buf, pixels)

            # Parse robot position
            if self.Types.ROBOT_POSITION.value in blocks:
                robot_data = blocks[self.Types.ROBOT_POSITION.value]
                parsed_map_data["robot"] = robot_data["position"]

            # Parse path data with coordinate transformation FIRST
            transformed_path_points = []
            if self.Types.PATH.value in blocks:
                path_data = blocks[self.Types.PATH.value].copy()
                # Apply coordinate transformation like current parser
                transformed_path_points = [
                    [point[0], self.Tools.DIMENSION_MM - point[1]]
                    for point in path_data["points"]
                ]
                path_data["points"] = transformed_path_points

                # Calculate current angle from transformed points
                if len(transformed_path_points) >= 2:
                    last_point = transformed_path_points[-1]
                    second_last = transformed_path_points[-2]
                    dx = last_point[0] - second_last[0]
                    dy = last_point[1] - second_last[1]
                    if dx != 0 or dy != 0:
                        angle_rad = math.atan2(dy, dx)
                        path_data["current_angle"] = math.degrees(angle_rad)
                parsed_map_data["path"] = path_data

            # Get robot angle from TRANSFORMED path data (like current implementation)
            robot_angle = 0
            if len(transformed_path_points) >= 2:
                last_point = transformed_path_points[-1]
                second_last = transformed_path_points[-2]
                dx = last_point[0] - second_last[0]
                dy = last_point[1] - second_last[1]
                if dx != 0 or dy != 0:
                    angle_rad = math.atan2(dy, dx)
                    robot_angle = int(math.degrees(angle_rad))

            parsed_map_data["robot_angle"] = robot_angle

            # Parse charger position
            if self.Types.CHARGER_LOCATION.value in blocks:
                charger_data = blocks[self.Types.CHARGER_LOCATION.value]
                parsed_map_data["charger"] = charger_data["position"]

            # Parse image data
            if self.Types.IMAGE.value in blocks:
                parsed_map_data["image"] = blocks[self.Types.IMAGE.value]

            # Parse goto predicted path
            if self.Types.GOTO_PREDICTED_PATH.value in blocks:
                goto_path_data = blocks[self.Types.GOTO_PREDICTED_PATH.value].copy()
                # Apply coordinate transformation
                goto_path_data["points"] = [
                    [point[0], self.Tools.DIMENSION_MM - point[1]]
                    for point in goto_path_data["points"]
                ]
                # Calculate current angle from transformed points (like working parser)
                if len(goto_path_data["points"]) >= 2:
                    points = goto_path_data["points"]
                    last_point = points[-1]
                    second_last = points[-2]
                    dx = last_point[0] - second_last[0]
                    dy = last_point[1] - second_last[1]
                    if dx != 0 or dy != 0:
                        angle_rad = math.atan2(dy, dx)
                        goto_path_data["current_angle"] = math.degrees(angle_rad)
                parsed_map_data["goto_predicted_path"] = goto_path_data

            # Parse goto target
            if self.Types.GOTO_TARGET.value in blocks:
                parsed_map_data["goto_target"] = blocks[self.Types.GOTO_TARGET.value][
                    "position"
                ]

            # Add missing fields to match expected JSON format
            parsed_map_data["currently_cleaned_zones"] = (
                blocks[self.Types.CURRENTLY_CLEANED_ZONES.value]["zones"]
                if self.Types.CURRENTLY_CLEANED_ZONES.value in blocks
                else []
            )
            parsed_map_data["forbidden_zones"] = (
                blocks[self.Types.FORBIDDEN_ZONES.value]["forbidden_zones"]
                if self.Types.FORBIDDEN_ZONES.value in blocks
                else []
            )
            parsed_map_data["forbidden_mop_zones"] = (
                blocks[self.Types.FORBIDDEN_MOP_ZONES.value]["forbidden_mop_zones"]
                if self.Types.FORBIDDEN_MOP_ZONES.value in blocks
                else []
            )
            parsed_map_data["virtual_walls"] = (
                blocks[self.Types.VIRTUAL_WALLS.value]["virtual_walls"]
                if self.Types.VIRTUAL_WALLS.value in blocks
                else []
            )
            parsed_map_data["carpet_areas"] = (
                blocks[self.Types.CARPET_MAP.value]["carpet_map"]
                if self.Types.CARPET_MAP.value in blocks
                else []
            )
            parsed_map_data["is_valid"] = self.is_valid

            return parsed_map_data

        except (struct.error, IndexError, ValueError):
            return None

    def parse_data(
        self, payload: Optional[bytes] = None, pixels: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get the map data from MQTT and return dictionary like old parsers."""
        if payload:
            try:
                self.map_data = self.parse(payload)
                parsed_data = self.parse_rrm_data(payload, pixels)
                if parsed_data:
                    self.map_data.update(parsed_data)
                # Return dictionary directly - faster!
                return self.map_data
            except (struct.error, IndexError, ValueError):
                return None
        return self.map_data
