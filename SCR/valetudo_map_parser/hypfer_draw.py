"""
Image Draw Class for Valetudo Hypfer Image Handling.
This class is used to simplify the ImageHandler class.
Version: 0.1.10
"""

from __future__ import annotations

import logging

from .config.drawable_elements import DrawableElement
from .config.types import Color, JsonType, NumpyArray, RobotPosition, RoomStore

MAP_BOUNDARY = 20000  # typical map extent ~5000–10000 units
_LOGGER = logging.getLogger(__name__)


class ImageDraw:
    """Class to handle the image creation.
    It Draws each element of the images, like the walls, zones, paths, etc."""

    def __init__(self, image_handler):
        self.img_h = image_handler
        self.file_name = self.img_h.shared.file_name

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

    async def draw_go_to_flag(
        self, np_array: NumpyArray, entity_dict: dict, color_go_to: Color
    ) -> NumpyArray:
        """Draw the goto target flag on the map."""
        go_to = entity_dict.get("go_to_target")
        if go_to:
            np_array = await self.img_h.draw.go_to_flag(
                np_array,
                (go_to[0]["points"][0], go_to[0]["points"][1]),
                self.img_h.shared.image_rotate,
                color_go_to,
            )
        return np_array

    async def async_draw_base_layer(
        self,
        img_np_array: NumpyArray,
        compressed_pixels_list: list[list[int]],
        layer_type: str,
        color_wall: Color,
        color_zone_clean: Color,
        pixel_size: int,
        disabled_rooms: set[int] | None = None,
    ) -> tuple[int, NumpyArray]:
        """
        Draw the base layer of the map.
    
        Sequential for rooms/segments to maintain room_id state. Returns (last_room_id, image_array).
        """
        if not compressed_pixels_list:
            return 0, img_np_array
    
        room_id = 0
        SEGMENT_LAYERS = ("segment", "floor")
        
        for compressed_pixels in compressed_pixels_list:
            pixels = self.img_h.data.sublist(compressed_pixels, 3)
    
            try:
                if layer_type in SEGMENT_LAYERS:
                    img_np_array, room_id = await self._process_room_layer(
                        img_np_array, pixels, layer_type, room_id, pixel_size, color_zone_clean
                    )
                elif layer_type == "wall":
                    img_np_array = await self._process_wall_layer(
                        img_np_array, pixels, pixel_size, color_wall, disabled_rooms
                    )
            except Exception as e:
                _LOGGER.warning("%s: Failed processing %s layer: %s", self.file_name, layer_type, e)
    
        return room_id, img_np_array

    async def _process_room_layer(
        self,
        img_np_array: NumpyArray,
        pixels: list[tuple[int, int, int]],
        layer_type: str,
        room_id: int,
        pixel_size: int,
        color_zone_clean: Color
    ) -> tuple[NumpyArray, int]:
        """
        Draw a room layer (segment or floor) onto the image array.
    
        Returns a tuple of (updated image array, next room_id).
        """
        draw_room = True
        drawing_config = getattr(self.img_h, "drawing_config", None)
    
        # Segment-specific enable check
        if layer_type == "segment" and drawing_config:
            current_room_id = room_id + 1  # 1-based for DrawableElement
            if 1 <= current_room_id <= 15:
                room_element = getattr(DrawableElement, f"ROOM_{current_room_id}", None)
                if room_element and hasattr(drawing_config, "is_enabled"):
                    draw_room = drawing_config.is_enabled(room_element)
    
        try:
            room_color = self.img_h.shared.rooms_colors[room_id]
        except IndexError:
            _LOGGER.warning(
                "%s: Invalid room_id %d for layer_type '%s'",
                self.file_name, room_id, layer_type
            )
            return img_np_array, room_id
    
        if layer_type == "segment":
            room_color = self._get_active_room_color(room_id, room_color, color_zone_clean)
    
        # Draw only if enabled
        if draw_room:
            try:
                img_np_array = await self.img_h.draw.from_json_to_image(
                    img_np_array, pixels, pixel_size, room_color
                )
            except IndexError as e:
                _LOGGER.warning(
                    "%s: Image draw error for room_id %d in '%s': %s",
                    self.file_name, room_id, layer_type, e
                )
    
        # Cycle room_id back to 0 after 15
        return img_np_array, (room_id + 1) % 16

    def _get_active_room_color(self, room_id, room_color, color_zone_clean):
        """Adjust the room color if the room is active."""
        if self.img_h.active_zones and room_id < len(self.img_h.active_zones):
            if self.img_h.active_zones[room_id] == 1:
                return tuple(
                    ((2 * room_color[i]) + color_zone_clean[i]) // 3 for i in range(4)
                )
        return room_color

    async def _process_wall_layer(
        self,
        img_np_array: NumpyArray,
        pixels: list[tuple[int, int, int]],
        pixel_size: int,
        color_wall: Color,
        disabled_rooms: set[int] | None = None,
    ) -> NumpyArray:
        """Draw a wall layer, optionally filtering out walls near disabled rooms."""
        _LOGGER.debug("%s: Drawing walls with color %s", self.file_name, color_wall)
    
        if not disabled_rooms:
            return await self.img_h.draw.from_json_to_image(
                img_np_array, pixels, pixel_size, color_wall
            )
    
        element_map = getattr(self.img_h, "element_map", None)
        if element_map is None:
            _LOGGER.warning("%s: Element map not available, drawing all walls", self.file_name)
            return await self.img_h.draw.from_json_to_image(
                img_np_array, pixels, pixel_size, color_wall
            )
    
        shape_y, shape_x = element_map.shape
        filtered_pixels = []
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == dy == 0)]
    
        for x, y, z in pixels:
            for dx, dy in offsets:
                cx, cy = x + dx, y + dy
                if 0 <= cx < shape_x and 0 <= cy < shape_y:
                    elem = element_map[cy, cx]
                    if 101 <= elem <= 115 and (elem - 101) in disabled_rooms:
                        break
            else:
                filtered_pixels.append((x, y, z))
    
        _LOGGER.debug(
            "%s: Drawing %d of %d wall pixels after filtering",
            self.file_name,
            len(filtered_pixels),
            len(pixels),
        )
        if filtered_pixels:
            return await self.img_h.draw.from_json_to_image(
                img_np_array, filtered_pixels, pixel_size, color_wall
            )
        return img_np_array

    async def async_draw_obstacle(
        self, np_array: NumpyArray, obstacle_positions: list[dict], color_no_go: Color
    ) -> NumpyArray:
        """Draw the obstacle positions from the entity data."""
        if obstacle_positions:
            await self.img_h.draw.async_draw_obstacles(
                np_array, obstacle_positions, color_no_go
            )
        return np_array

    async def async_draw_charger(
        self,
        np_array: NumpyArray,
        entity_dict: dict,
        color_charger: Color,
    ) -> NumpyArray:
        """Get the charger position from the entity data."""
        try:
            charger_pos = entity_dict.get("charger_location")
        except KeyError:
            _LOGGER.warning("%s: No charger position found.", self.file_name)
            return np_array
        if charger_pos:
            charger_pos = charger_pos[0]["points"]
            self.img_h.charger_pos = {
                "x": charger_pos[0],
                "y": charger_pos[1],
            }
            np_array = await self.img_h.draw.battery_charger(
                np_array, charger_pos[0], charger_pos[1], color_charger
            )
            return np_array
        return np_array

    async def async_draw_zones(
        self,
        m_json: JsonType,
        np_array: NumpyArray,
        color_zone_clean: Color,
        color_no_go: Color,
    ) -> NumpyArray:
        """Draw active, no-go, and no-mop zones in-place on the map array."""
        try:
            zones = self.img_h.data.find_zone_entities(m_json) or {}
        except (ValueError, KeyError) as e:
            _LOGGER.debug("%s: No zones found (%s)", self.file_name, e)
            return np_array
    
        _LOGGER.info("%s: Got zones.", self.file_name)
    
        for key, color in [
            ("active_zone", color_zone_clean),
            ("no_go_area", color_no_go),
            ("no_mop_area", color_no_go),
        ]:
            if key in zones:
                np_array = await self.img_h.draw.zones(np_array, zones[key], color)
    
        return np_array

    def _prepare_xyz_sequences(points: list[int]) -> list[tuple[int, int, int]]:
        """Convert a flat [x1, y1, z1, x2, y2, z2, ...] list into (x, y, z) triples."""
        return self.img_h.data.sublist(points, 3)
        
    async def async_draw_virtual_walls(
        self, m_json: JsonType, np_array: NumpyArray, color_no_go: Color
    ) -> NumpyArray:
        """Draw any configured virtual walls on the map."""
        try:
            virtual_walls = self.img_h.data.find_virtual_walls(m_json)
        except (ValueError, KeyError) as e:
            _LOGGER.debug("%s: No virtual walls found (%s)", self.file_name, e)
            return np_array
    
        if not virtual_walls:
            return np_array
    
        _LOGGER.info("%s: Drawing %d virtual wall(s).", self.file_name, len(virtual_walls))
        return await self.img_h.draw.draw_virtual_walls(np_array, virtual_walls, color_no_go)
    
    
    async def async_draw_paths(
        self,
        np_array: NumpyArray,
        m_json: JsonType,
        color_move: Color,
        color_gray: Color,
    ) -> NumpyArray:
        """Draw predicted and actual paths from the JSON data."""
        try:
            paths_data = self.img_h.data.find_paths_entities(m_json)
        except KeyError as e:
            _LOGGER.warning("%s: Error extracting paths data: %s", self.file_name, e)
            return np_array
    
        predicted_path = (paths_data.get("predicted_path") or [])
        if predicted_path and "points" in predicted_path[0]:
            coords = _prepare_xy_sequences(predicted_path[0]["points"])
            joined = self.img_h.data.sublist_join(coords, 2)
            np_array = await self.img_h.draw.lines(np_array, joined, 2, color_gray)
    
        path_pixels = paths_data.get("path") or []
        for path in path_pixels:
            coords = _prepare_xy_sequences(path.get("points", []))
            self.img_h.shared.map_new_path = self.img_h.data.sublist_join(coords, 2)
            np_array = await self.img_h.draw.lines(
                np_array, self.img_h.shared.map_new_path, 5, color_move
            )
    
        return np_array

    def _check_active_zone_and_set_zooming(self) -> None:
        """Check active zones and update zooming state accordingly."""
        zooming = False
    
        if self.img_h.active_zones and self.img_h.robot_in_room:
            segment_id = str(self.img_h.robot_in_room.get("id"))
            room_keys = list(RoomStore(self.file_name).get_rooms().keys())
    
            _LOGGER.debug(
                "%s: Active zones debug - segment_id: %s, room_keys: %s, active_zones: %s",
                self.file_name, segment_id, room_keys, self.img_h.active_zones
            )
    
            if segment_id in room_keys:
                pos = room_keys.index(segment_id)
                in_bounds = pos < len(self.img_h.active_zones)
    
                _LOGGER.debug(
                    "%s: Segment ID %s found at position %s, active_zones[%s] = %s",
                    self.file_name, segment_id, pos, pos,
                    self.img_h.active_zones[pos] if in_bounds else "OUT_OF_BOUNDS"
                )
    
                if in_bounds:
                    zooming = bool(self.img_h.active_zones[pos])
            else:
                _LOGGER.warning(
                    "%s: Segment ID %s not found in room_keys %s",
                    self.file_name, segment_id, room_keys
                )
    
        self.img_h.zooming = zooming


    @staticmethod
    def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        """
        Determine if (x, y) lies inside a polygon using ray casting.
    
        Args:
            x, y: Coordinates of the point to test.
            polygon: List of (x, y) vertices; first and last may be the same but need not be.
    
        Returns:
            True if inside or on the boundary, False otherwise.
        """
        n = len(polygon)
        if n < 3:
            return False
    
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if ((p1y > y) != (p2y > y)):  # crosses horizontal ray
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    async def async_get_robot_in_room(
        self,
        robot_y: int = 0,
        robot_x: int = 0,
        angle: float = 0.0
    ) -> RobotPosition:
        """Return robot position and the room it’s currently in."""
        def make_pos(room_name: str | None) -> RobotPosition:
            return {"x": robot_x, "y": robot_y, "angle": angle, "in_room": room_name}
    
        last_room = self.img_h.robot_in_room or None
    
        # Reuse cached polygon or bbox
        if last_room:
            if "outline" in last_room and self.point_in_polygon(int(robot_x), int(robot_y), last_room["outline"]):
                self._check_active_zone_and_set_zooming()
                return make_pos(last_room["room"])
            if all(k in last_room for k in ("left", "right", "up", "down")):
                if (last_room["left"] <= robot_x <= last_room["right"] and
                    last_room["up"]   <= robot_y <= last_room["down"]):
                    self._check_active_zone_and_set_zooming()
                    return make_pos(last_room["room"])
    
        # Reject obviously off‑map points
        if abs(robot_x) > MAP_BOUNDARY or abs(robot_y) > MAP_BOUNDARY:
            _LOGGER.debug("%s: position (%s, %s) outside map bounds", self.file_name, robot_x, robot_y)
            self.img_h.robot_in_room = last_room
            self.img_h.zooming = False
            return make_pos(last_room["room"] if last_room else None)
    
        # No room geometry? bail early
        if self.img_h.rooms_pos is None:
            _LOGGER.debug("%s: No rooms data for position detection", self.file_name)
            self.img_h.robot_in_room = last_room
            self.img_h.zooming = False
            return make_pos(last_room["room"] if last_room else None)
    
        # Search all rooms
        for idx, room in enumerate(self.img_h.rooms_pos):
            if "outline" in room:
                if self.point_in_polygon(int(robot_x), int(robot_y), room["outline"]):
                    self.img_h.robot_in_room = {
                        "id": room.get("id", idx),
                        "room": str(room["name"]),
                        "outline": room["outline"],
                    }
                    self._check_active_zone_and_set_zooming()
                    _LOGGER.debug("%s is in %s room (polygon)", self.file_name, room["name"])
                    return make_pos(room["name"])
            elif "corners" in room:
                left, up = map(int, room["corners"][0])
                right, down = map(int, room["corners"][2])
                if left <= robot_x <= right and up <= robot_y <= down:
                    self.img_h.robot_in_room = {
                        "id": room.get("id", idx),
                        "left": left, "right": right,
                        "up": up, "down": down,
                        "room": str(room["name"]),
                    }
                    self._check_active_zone_and_set_zooming()
                    _LOGGER.debug("%s is in %s room (bbox)", self.file_name, room["name"])
                    return make_pos(room["name"])
    
        # Not found
        _LOGGER.debug("%s not located in any room", self.file_name)
        self.img_h.robot_in_room = last_room
        self.img_h.zooming = False
        return make_pos(last_room["room"] if last_room else None)
    

    async def async_get_robot_position(self, entity_dict: dict) -> tuple | None:
        """Get the robot position from the entity data."""
        robot_pos = None
        robot_position = None
        robot_position_angle = None
        try:
            robot_pos = entity_dict.get("robot_position")
        except KeyError:
            _LOGGER.warning("%s No robot position found.", self.file_name)
            return None, None, None
        finally:
            if robot_pos:
                robot_position = robot_pos[0]["points"]
                robot_position_angle = round(
                    float(robot_pos[0]["metaData"]["angle"]), 1
                )
                if self.img_h.rooms_pos is None:
                    self.img_h.robot_pos = {
                        "x": robot_position[0],
                        "y": robot_position[1],
                        "angle": robot_position_angle,
                    }
                else:
                    self.img_h.robot_pos = await self.async_get_robot_in_room(
                        robot_y=(robot_position[1]),
                        robot_x=(robot_position[0]),
                        angle=robot_position_angle,
                    )

        return robot_pos, robot_position, robot_position_angle
