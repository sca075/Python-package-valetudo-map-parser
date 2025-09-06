"""
Collections of Json and List routines
ImageData is part of the Image_Handler
used functions to search data in the json
provided for the creation of the new camera frame
Version: v0.1.10
"""

from __future__ import annotations

import numpy as np
from typing import List, Sequence, TypeVar, Any, TypedDict, NotRequired, Literal

from dataclasses import dataclass, field

from .config.types import ImageSize, JsonType

T = TypeVar("T")

# --- Common Nested Structures ---


class RangeStats(TypedDict):
    min: int
    max: int
    mid: int
    avg: int


class Dimensions(TypedDict):
    x: RangeStats
    y: RangeStats
    pixelCount: int


# --- Layer Types ---


class FloorWallMeta(TypedDict, total=False):
    area: int


class SegmentMeta(TypedDict, total=False):
    segmentId: str
    active: bool
    source: str
    area: int


class MapLayerBase(TypedDict):
    __class__: Literal["MapLayer"]
    type: str
    pixels: list[int]
    compressedPixels: list[int]
    dimensions: Dimensions


class FloorWallLayer(MapLayerBase):
    metaData: FloorWallMeta
    type: Literal["floor", "wall"]


class SegmentLayer(MapLayerBase):
    metaData: SegmentMeta
    type: Literal["segment"]


# --- Entity Types ---


class PointMeta(TypedDict, total=False):
    angle: float
    label: str
    id: str


class PointMapEntity(TypedDict):
    __class__: Literal["PointMapEntity"]
    type: str
    points: list[int]
    metaData: NotRequired[PointMeta]


class PathMapEntity(TypedDict):
    __class__: Literal["PathMapEntity"]
    type: str
    points: list[int]
    metaData: dict[str, object]  # flexible for now


Entity = PointMapEntity | PathMapEntity

# --- Top-level Map ---


class MapMeta(TypedDict, total=False):
    version: int
    totalLayerArea: int


class Size(TypedDict):
    x: int
    y: int


class ValetudoMap(TypedDict):
    __class__: Literal["ValetudoMap"]
    metaData: MapMeta
    size: Size
    pixelSize: int
    layers: list[FloorWallLayer | SegmentLayer]
    entities: list[Entity]


class ImageData:
    """Class to handle the image data."""

    @staticmethod
    def sublist(lst: Sequence[T], n: int) -> list[Sequence[T]]:
        """Sub lists of specific n number of elements"""
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    @staticmethod
    def sublist_join(lst: Sequence[T], n: int) -> list[list[T]]:
        """Join the lists in a unique list of n elements"""
        arr = np.array(lst)
        num_windows = len(lst) - n + 1
        result = [arr[i : i + n].tolist() for i in range(num_windows)]
        return result

    # The below functions are basically the same ech one
    # of them is allowing filtering and putting together in a
    # list the specific Layers, Paths, Zones and Pints in the
    # Vacuums Json in parallel.

    @staticmethod
    def get_image_size(json_data: JsonType) -> dict[str, int | list[int]]:
        """Get the image size from the json."""
        if json_data:
            size_x = int(json_data["size"]["x"])
            size_y = int(json_data["size"]["y"])
            return {
                "x": size_x,
                "y": size_y,
                "centre": [(size_x // 2), (size_y // 2)],
            }
        return {"x": 0, "y": 0, "centre": [0, 0]}

    @staticmethod
    def get_json_id(json_data: JsonType) -> str | None:
        """Get the json id from the json."""
        try:
            json_id = json_data["metaData"]["nonce"]
        except (ValueError, KeyError):
            json_id = None
        return json_id

    @staticmethod
    def get_obstacles(
        entity_dict: dict[str, list[PointMapEntity]],
    ) -> list[dict[str, str | int | None]]:
        """
        Extract obstacle positions from Valetudo entity data.

        Args:
            entity_dict: Parsed JSON-like dict containing obstacle data.

        Returns:
            A list of obstacle dicts with keys:
                - 'label': obstacle label string
                - 'points': dict with 'x' and 'y' coordinates
                - 'id': obstacle image/metadata ID (if any)
            Returns an empty list if no valid obstacles found.
        """
        obstacle_data = entity_dict.get("obstacle")  # .get() won't raise KeyError
        if not obstacle_data:
            return []

        obstacle_positions: list[dict[str, Any]] = []

        for obstacle in obstacle_data:
            meta = obstacle.get("metaData", {}) or {}
            label = meta.get("label")
            image_id = meta.get("id")
            points = obstacle.get("points") or []

            # Expecting at least two coordinates for a valid obstacle
            if label and len(points) >= 2:
                obstacle_positions.append(
                    {
                        "label": label,
                        "points": {"x": points[0], "y": points[1]},
                        "id": image_id,
                    }
                )

        return obstacle_positions

    @staticmethod
    def find_layers(
        json_obj: JsonType,
        layer_dict: dict[str, list[Any]] | None,
        active_list: list[int] | None,
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """
        Recursively traverse a JSON-like structure to find MapLayer entries.

        Args:
            json_obj: The JSON-like object (dicts/lists) to search.
            layer_dict: Optional mapping of layer_type to a list of compressed pixel data.
            active_list: Optional list of active segment flags.

        Returns:
            A tuple:
                - dict mapping layer types to their compressed pixel arrays.
                - list of integers marking active segment layers.
        """
        if layer_dict is None:
            layer_dict = {}
            active_list = []

        if isinstance(json_obj, dict):
            if json_obj.get("__class") == "MapLayer":
                layer_type = json_obj.get("type")
                meta_data = json_obj.get("metaData") or {}
                if layer_type:
                    layer_dict.setdefault(layer_type, []).append(
                        json_obj.get("compressedPixels", [])
                    )
                    # Safely extract "active" flag if present and convertible to int
                    if layer_type == "segment":
                        try:
                            active_list.append(int(meta_data.get("active", 0)))
                        except (ValueError, TypeError):
                            pass  # skip invalid/missing 'active' values

            # json_obj.items() yields (key, value), so we only want the values
            for _, value in json_obj.items():
                ImageData.find_layers(value, layer_dict, active_list)

        elif isinstance(json_obj, list):
            for item in json_obj:
                ImageData.find_layers(item, layer_dict, active_list)

        return layer_dict, active_list

    @staticmethod
    def find_points_entities(
        json_obj: ValetudoMap, entity_dict: dict = None
    ) -> dict[str, list[PointMapEntity]]:
        """
        Traverse a ValetudoMap and collect PointMapEntity objects by their `type`.

        Args:
            json_obj: The full parsed JSON structure of a ValetudoMap.
            entity_dict: Optional starting dict to append into.

        Returns:
            A dict mapping entity type strings to lists of PointMapEntitys.
        """
        if entity_dict is None:
            entity_dict = {}
        if isinstance(json_obj, dict):
            if json_obj.get("__class") == "PointMapEntity":
                entity_type = json_obj.get("type")
                if entity_type:
                    entity_dict.setdefault(entity_type, []).append(json_obj)
            for value in json_obj.values():
                ImageData.find_points_entities(value, entity_dict)
        elif isinstance(json_obj, list):
            for item in json_obj:
                ImageData.find_points_entities(item, entity_dict)
        return entity_dict

    @staticmethod
    def find_paths_entities(
        json_obj: JsonType, entity_dict: dict[str, list[Entity]] | None = None
    ) -> dict[str, list[Entity]]:
        """Find the paths entities in the json object."""

        if entity_dict is None:
            entity_dict = {}
        if isinstance(json_obj, dict):
            if json_obj.get("__class") == "PathMapEntity":
                entity_type = json_obj.get("type")
                if entity_type:
                    entity_dict.setdefault(entity_type, []).append(json_obj)
            for value in json_obj.values():
                ImageData.find_paths_entities(value, entity_dict)
        elif isinstance(json_obj, list):
            for item in json_obj:
                ImageData.find_paths_entities(item, entity_dict)
        return entity_dict

    @staticmethod
    def find_zone_entities(
        json_obj: JsonType, entity_dict: dict[str, list[Entity]] | None = None
    ) -> dict[str, list[Entity]]:
        """Find the zone entities in the json object."""
        if entity_dict is None:
            entity_dict = {}
        if isinstance(json_obj, dict):
            if json_obj.get("__class") == "PolygonMapEntity":
                entity_type = json_obj.get("type")
                if entity_type:
                    entity_dict.setdefault(entity_type, []).append(json_obj)
            for value in json_obj.values():
                ImageData.find_zone_entities(value, entity_dict)
        elif isinstance(json_obj, list):
            for item in json_obj:
                ImageData.find_zone_entities(item, entity_dict)
        return entity_dict

    @staticmethod
    def find_virtual_walls(json_obj: JsonType) -> list[list[tuple[float, float]]]:
        """
        Recursively search a JSON-like structure for virtual wall line entities.

        Args:
            json_obj: The JSON-like data (dicts/lists) to search.

        Returns:
            A list of point lists, where each point list belongs to a virtual wall.
        """
        virtual_walls: list[list[tuple[float, float]]] = []

        def _recurse(obj: Any) -> None:
            if isinstance(obj, dict):
                if (
                    obj.get("__class") == "LineMapEntity"
                    and obj.get("type") == "virtual_wall"
                ):
                    points = obj.get("points")
                    if isinstance(points, list):
                        virtual_walls.append(
                            points
                        )  # Type checkers may refine further here

                for value in obj.values():
                    _recurse(value)

            elif isinstance(obj, list):
                for item in obj:
                    _recurse(item)

        _recurse(json_obj)
        return virtual_walls

    @staticmethod
    async def async_get_rooms_coordinates(
        pixels: Sequence[tuple[int, int, int]], pixel_size: int = 5, rand: bool = False
    ) -> tuple[int, int, int, int] | tuple[tuple[int, int], tuple[int, int]]:
        """
        Extract the room bounding box coordinates from vacuum pixel data.

        Args:
            pixels: Sequence of (x, y, z) values representing pixels.
            pixel_size: Size of each pixel in mm. Defaults to 5.
            rand: If True, return coordinates in rand256 format.

        Returns:
            If rand is True:
                ((max_x_mm, max_y_mm), (min_x_mm, min_y_mm))
            Else:
                (min_x_mm, min_y_mm, max_x_mm, max_y_mm)
        """
        if not pixels:
            raise ValueError("Pixels list cannot be empty.")

        # Initialise min/max using the first pixel
        first_x, first_y, _ = pixels[0]
        min_x = max_x = first_x
        min_y = max_y = first_y

        for x, y, z in pixels:
            if rand:
                max_x = max(max_x, x)
                max_y = max(max_y, y + pixel_size)
            else:
                max_x = max(max_x, x + z)
                max_y = max(max_y, y + pixel_size)

            min_x = min(min_x, x)
            min_y = min(min_y, y)

        if rand:
            to_mm = lambda v: v * pixel_size * 10
            return (to_mm(max_x), to_mm(max_y)), (to_mm(min_x), to_mm(min_y))

        return (
            min_x * pixel_size,
            min_y * pixel_size,
            max_x * pixel_size,
            max_y * pixel_size,
        )


class RandImageData:
    """This functions read directly the data from the json created
    from the parser for Valetudo Re. They allow to use the
    functions to draw the image without changes on the drawing class."""

    @staticmethod
    def from_rrm_to_compressed_pixels(
        pixel_data: list,
        image_width: int = 0,
        image_height: int = 0,
        image_top: int = 0,
        image_left: int = 0,
    ) -> list:
        """Convert the pixel data to compressed pixels."""
        compressed_pixels = []

        tot_pixels = 0
        current_x, current_y, count = None, None, 0
        for index in pixel_data:
            x = (index % image_width) + image_left
            y = ((image_height - 1) - (index // image_width)) + image_top

            if current_x == x and current_y == y:
                count += 1
            else:
                if current_x is not None:
                    compressed_pixels.append([current_x, current_y, count])
                current_x, current_y, count = x, y, 1
            tot_pixels += 1
        if current_x is not None:
            compressed_pixels.append([current_x, current_y, count])
        return compressed_pixels

    @staticmethod
    def _calculate_max_x_y(coord_array):
        """Calculate the max and min x and y coordinates."""
        max_x = -float("inf")
        max_y = -float("inf")

        for x, y, _ in coord_array:
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        return (max_x * 6), (max_y * 6)

    @staticmethod
    def rrm_coordinates_to_valetudo(points):
        """Transform the coordinates from RRM to Valetudo."""
        transformed_points = []
        dimension_mm = 50 * 1024
        for i, p in enumerate(points):
            if i % 2 == 0:
                transformed_points.append(round(p / 10))
            else:
                transformed_points.append(round((dimension_mm - p) / 10))
        return transformed_points

    @staticmethod
    def rrm_valetudo_path_array(points):
        """Transform the path coordinates from RRM to Valetudo."""
        transformed_points = []
        for point in points:
            transformed_x = round(point[0] / 10)
            transformed_y = round(point[1] / 10)
            transformed_points.extend([[transformed_x, transformed_y]])
        return transformed_points

    @staticmethod
    def get_rrm_image(json_data: JsonType) -> JsonType:
        """Get the image data from the json."""
        if isinstance(json_data, tuple):
            return {}
        return json_data.get("image", {})

    @staticmethod
    def get_rrm_path(json_data: JsonType) -> JsonType:
        """Get the path data from the json."""
        return json_data.get("path", {})

    @staticmethod
    def get_rrm_goto_predicted_path(json_data: JsonType) -> List or None:
        """Get the predicted path data from the json."""
        try:
            predicted_path = json_data.get("goto_predicted_path", {})
            points = predicted_path["points"]
        except KeyError:
            return None
        predicted_path = ImageData.sublist_join(
            RandImageData.rrm_valetudo_path_array(points), 2
        )
        return predicted_path

    @staticmethod
    def get_rrm_charger_position(json_data: JsonType) -> JsonType:
        """Get the charger position from the json."""
        return json_data.get("charger", {})

    @staticmethod
    def get_rrm_robot_position(json_data: JsonType) -> JsonType:
        """Get the robot position from the json."""
        return json_data.get("robot", {})

    @staticmethod
    def get_rrm_robot_angle(json_data: JsonType) -> tuple:
        """
        Get the robot angle from the json.
        Return the calculated angle and original angle.
        """
        angle_c = round(json_data.get("robot_angle", 0))
        # Convert negative values: -10 -> 350, -180 -> 359, but keep positive: 24 -> 24
        if angle_c < 0:
            if angle_c == -180:
                angle = 359  # -180 becomes 359 (avoiding 360)
            else:
                angle = 360 + angle_c  # -10 -> 350, -90 -> 270
        else:
            angle = angle_c

        angle = (angle + 90) % 360
        return angle, json_data.get("robot_angle", 0)

    @staticmethod
    def get_rrm_goto_target(json_data: JsonType) -> list or None:
        """Get the goto target from the json."""
        try:
            path_data = json_data.get("goto_target", {})
        except KeyError:
            return None

        if path_data and path_data != []:
            path_data = RandImageData.rrm_coordinates_to_valetudo(path_data)
            return path_data
        return None

    @staticmethod
    def get_rrm_currently_cleaned_zones(json_data: JsonType) -> dict:
        """Get the currently cleaned zones from the json."""
        re_zones = json_data.get("currently_cleaned_zones", [])
        formatted_zones = RandImageData._rrm_valetudo_format_zone(re_zones)
        return formatted_zones

    @staticmethod
    def get_rrm_forbidden_zones(json_data: JsonType) -> dict:
        """Get the forbidden zones from the json."""
        re_zones = json_data.get("forbidden_zones", [])
        formatted_zones = RandImageData._rrm_valetudo_format_zone(re_zones)
        return formatted_zones

    @staticmethod
    def _rrm_valetudo_format_zone(coordinates: list) -> Any:
        """Format the zones from RRM to Valetudo."""
        formatted_zones = []
        for zone_data in coordinates:
            if len(zone_data) == 4:  # This is a zone_clean (4 coordinates)
                formatted_zone = {
                    "__class": "PolygonMapEntity",
                    "metaData": {},
                    "points": [
                        zone_data[0] // 10,
                        zone_data[1] // 10,
                        zone_data[2] // 10,
                        zone_data[1] // 10,
                        zone_data[2] // 10,
                        zone_data[3] // 10,
                        zone_data[0] // 10,
                        zone_data[3] // 10,
                    ],
                    "type": "zone_clean",
                }
                formatted_zones.append(formatted_zone)
            elif len(zone_data) == 8:  # This is a no_go_area (8 coordinates)
                formatted_zone = {
                    "__class": "PolygonMapEntity",
                    "metaData": {},
                    "points": [
                        zone_data[0] // 10,
                        zone_data[1] // 10,
                        zone_data[2] // 10,
                        zone_data[3] // 10,
                        zone_data[4] // 10,
                        zone_data[5] // 10,
                        zone_data[6] // 10,
                        zone_data[7] // 10,
                    ],
                    "type": "no_go_area",
                }
                formatted_zones.append(formatted_zone)

        return formatted_zones

    @staticmethod
    def _rrm_valetudo_lines(coordinates: list) -> list:
        """Format the lines from RRM to Valetudo."""
        formatted_lines = []
        for lines in coordinates:
            line = [lines[0] // 10, lines[1] // 10, lines[2] // 10, lines[3] // 10]
            formatted_lines.append(line)
        return formatted_lines

    @staticmethod
    def get_rrm_virtual_walls(json_data: JsonType) -> list or None:
        """Get the virtual walls from the json."""
        try:
            tmp_data = json_data.get("virtual_walls", [])
        except KeyError:
            return None
        virtual_walls = RandImageData._rrm_valetudo_lines(tmp_data)
        return virtual_walls

    @staticmethod
    def get_rrm_currently_cleaned_blocks(json_data: JsonType) -> list:
        """Get the currently cleaned blocks from the json."""
        return json_data.get("currently_cleaned_blocks", [])

    @staticmethod
    def get_rrm_forbidden_mop_zones(json_data: JsonType) -> list:
        """Get the forbidden mop zones from the json."""
        return json_data.get("forbidden_mop_zones", [])

    @staticmethod
    def get_rrm_image_size(json_data: JsonType) -> ImageSize:
        """Get the image size from the json."""
        if isinstance(json_data, tuple):
            return 0, 0
        image = RandImageData.get_rrm_image(json_data)
        if image == {}:
            return 0, 0
        dimensions = image.get("dimensions", {})
        return dimensions.get("width", 0), dimensions.get("height", 0)

    @staticmethod
    def get_rrm_image_position(json_data: JsonType) -> tuple:
        """Get the image position from the json."""
        image = RandImageData.get_rrm_image(json_data)
        position = image.get("position", {})
        return position.get("top", 0), position.get("left", 0)

    @staticmethod
    def get_rrm_floor(json_data: JsonType) -> list:
        """Get the floor data from the json."""
        img = RandImageData.get_rrm_image(json_data)
        return img.get("pixels", {}).get("floor", [])

    @staticmethod
    def get_rrm_walls(json_data: JsonType) -> list:
        """Get the walls data from the json."""
        img = RandImageData.get_rrm_image(json_data)
        return img.get("pixels", {}).get("walls", [])

    @staticmethod
    async def async_get_rrm_segments(
        json_data: JsonType,
        size_x: int,
        size_y: int,
        pos_top: int,
        pos_left: int,
        out_lines: bool = False,
    ) -> tuple or list:
        """Get the segments data from the json."""

        img = RandImageData.get_rrm_image(json_data)
        seg_data = img.get("segments", {})
        seg_ids = seg_data.get("id")
        segments = []
        outlines = []
        count_seg = 0
        for id_seg in seg_ids:
            tmp_data = seg_data.get("pixels_seg_" + str(id_seg))
            segments.append(
                RandImageData.from_rrm_to_compressed_pixels(
                    tmp_data,
                    image_width=size_x,
                    image_height=size_y,
                    image_top=pos_top,
                    image_left=pos_left,
                )
            )
            if out_lines:
                room_coords = await ImageData.async_get_rooms_coordinates(
                    pixels=segments[count_seg], rand=True
                )
                outlines.append(room_coords)
                count_seg += 1
        if count_seg > 0:
            if out_lines:
                return segments, outlines
            return segments
        return []

    @staticmethod
    def get_rrm_segments_ids(json_data: JsonType) -> list or None:
        """Get the segments ids from the json."""
        try:
            img = RandImageData.get_rrm_image(json_data)
            seg_ids = img.get("segments", {}).get("id", [])
        except KeyError:
            return None
        return seg_ids


@dataclass
class HyperMapData:
    """Class to handle the map data snapshots."""

    json_data: Any = None
    json_id: str = "" or None
    obstacles: dict[str, list[Any]] = field(default_factory=dict)
    paths: dict[str, list[Any]] = field(default_factory=dict)
    image_size: dict[str, int | list[int]] = field(default_factory=dict)
    areas: dict[str, list[Any]] = field(default_factory=dict)
    pixel_size: int = 0
    entity_dict: dict[str, list[Any]] = field(default_factory=dict)
    layers: dict[str, list[Any]] = field(default_factory=dict)
    active_zones: list[int] = field(default_factory=list)
    virtual_walls: list[list[tuple[float, float]]] = field(default_factory=list)

    @classmethod
    async def async_from_valetudo_json(cls, json_data: Any) -> "HyperMapData":
        """
        Build a fully-populated MapSnapshot from raw Valetudo JSON
        using ImageData's helper functions.
        """

        # Call into your refactored static/class methods
        json_id = ImageData.get_json_id(json_data)
        paths = ImageData.find_paths_entities(json_data)
        image_size = ImageData.get_image_size(json_data)
        areas = ImageData.find_zone_entities(json_data)
        layers = {}
        active_zones = []
        # Hypothetical obstacles finder, if you have one
        obstacles = getattr(ImageData, "find_obstacles_entities", lambda *_: {})(
            json_data
        )
        virtual_walls = ImageData.find_virtual_walls(json_data)
        pixel_size = int(json_data["pixelSize"])
        layers, active_zones = ImageData.find_layers(
            json_data["layers"], layers, active_zones
        )
        entity_dict = ImageData.find_points_entities(json_data)

        return cls(
            json_data=json_data,
            json_id=json_id,
            image_size=image_size,
            obstacles=obstacles,
            paths=paths,
            areas=areas,
            virtual_walls=virtual_walls,
            entity_dict=entity_dict,
            pixel_size=pixel_size,
            layers=layers,
            active_zones=active_zones,
        )
