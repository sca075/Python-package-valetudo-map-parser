"""
Image Draw Class for Valetudo Hypfer Image Handling.
This class is used to simplify the ImageHandler class.
Version: 0.1.10
"""

from __future__ import annotations

import logging

from .config.drawable_elements import DrawableElement
from .config.types import Color, JsonType, NumpyArray, RobotPosition, RoomStore


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
        img_np_array,
        compressed_pixels_list,
        layer_type,
        color_wall,
        color_zone_clean,
        pixel_size,
        disabled_rooms=None,
    ):
        """Draw the base layer of the map with parallel processing for rooms.

        Args:
            img_np_array: The image array to draw on
            compressed_pixels_list: The list of compressed pixels to draw
            layer_type: The type of layer to draw (segment, floor, wall)
            color_wall: The color to use for walls
            color_zone_clean: The color to use for clean zones
            pixel_size: The size of each pixel
            disabled_rooms: A set of room IDs that are disabled

        Returns:
            A tuple of (room_id, img_np_array)
        """
        room_id = 0

        # Sequential processing for rooms/segments (dependencies require this)
        for compressed_pixels in compressed_pixels_list:
            pixels = self.img_h.data.sublist(compressed_pixels, 3)

            if layer_type in ["segment", "floor"]:
                img_np_array, room_id = await self._process_room_layer(
                    img_np_array,
                    pixels,
                    layer_type,
                    room_id,
                    pixel_size,
                    color_zone_clean,
                )
            elif layer_type == "wall":
                img_np_array = await self._process_wall_layer(
                    img_np_array, pixels, pixel_size, color_wall, disabled_rooms
                )

        return room_id, img_np_array

    async def _process_room_layer(
        self, img_np_array, pixels, layer_type, room_id, pixel_size, color_zone_clean
    ):
        """Process a room layer (segment or floor)."""
        # Check if this room should be drawn
        draw_room = True
        if layer_type == "segment" and hasattr(self.img_h, "drawing_config"):
            # The room_id is 0-based, but DrawableElement.ROOM_x is 1-based
            current_room_id = room_id + 1
            if 1 <= current_room_id <= 15:
                # Use the DrawableElement imported at the top of the file

                room_element = getattr(DrawableElement, f"ROOM_{current_room_id}", None)
                if room_element and hasattr(self.img_h.drawing_config, "is_enabled"):
                    draw_room = self.img_h.drawing_config.is_enabled(room_element)

        # Get the room color
        room_color = self.img_h.shared.rooms_colors[room_id]

        try:
            if layer_type == "segment":
                room_color = self._get_active_room_color(
                    room_id, room_color, color_zone_clean
                )

            # Only draw the room if it's enabled
            if draw_room:
                img_np_array = await self.img_h.draw.from_json_to_image(
                    img_np_array, pixels, pixel_size, room_color
                )

            # Always increment the room_id, even if the room is not drawn
            room_id = (room_id + 1) % 16  # Cycle room_id back to 0 after 15

        except IndexError as e:
            _LOGGER.warning("%s: Image Draw Error: %s", self.file_name, str(e))

        return img_np_array, room_id

    def _get_active_room_color(self, room_id, room_color, color_zone_clean):
        """Adjust the room color if the room is active."""
        if self.img_h.active_zones and room_id < len(self.img_h.active_zones):
            if self.img_h.active_zones[room_id] == 1:
                return tuple(
                    ((2 * room_color[i]) + color_zone_clean[i]) // 3 for i in range(4)
                )
        return room_color

    async def _process_wall_layer(
        self, img_np_array, pixels, pixel_size, color_wall, disabled_rooms=None
    ):
        """Process a wall layer.

        Args:
            img_np_array: The image array to draw on
            pixels: The pixels to draw
            pixel_size: The size of each pixel
            color_wall: The color to use for the walls
            disabled_rooms: A set of room IDs that are disabled

        Returns:
            The updated image array
        """
        # Log the wall color to verify alpha is being passed correctly
        _LOGGER.debug("%s: Drawing walls with color %s", self.file_name, color_wall)

        # If there are no disabled rooms, draw all walls
        if not disabled_rooms:
            return await self.img_h.draw.from_json_to_image(
                img_np_array, pixels, pixel_size, color_wall
            )

        # If there are disabled rooms, we need to check each wall pixel
        # to see if it belongs to a disabled room
        _LOGGER.debug(
            "%s: Filtering walls for disabled rooms: %s", self.file_name, disabled_rooms
        )

        # Get the element map if available
        element_map = getattr(self.img_h, "element_map", None)
        if element_map is None:
            _LOGGER.warning(
                "%s: Element map not available, drawing all walls", self.file_name
            )
            return await self.img_h.draw.from_json_to_image(
                img_np_array, pixels, pixel_size, color_wall
            )

        # Filter out walls that belong to disabled rooms
        filtered_pixels = []
        for x, y, z in pixels:
            # Check if this wall pixel is adjacent to a disabled room
            # by checking the surrounding pixels in the element map
            is_disabled_room_wall = False

            # Check the element map at this position and surrounding positions
            # to see if this wall is adjacent to a disabled room
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    # Skip the center pixel
                    if dx == 0 and dy == 0:
                        continue

                    # Calculate the position to check
                    check_x = x + dx
                    check_y = y + dy

                    # Make sure the position is within bounds
                    if (
                        check_x < 0
                        or check_y < 0
                        or check_x >= element_map.shape[1]
                        or check_y >= element_map.shape[0]
                    ):
                        continue

                    # Get the element at this position
                    element = element_map[check_y, check_x]

                    # Check if this element is a disabled room
                    # Room elements are in the range 101-115 (ROOM_1 to ROOM_15)
                    if 101 <= element <= 115:
                        room_id = element - 101  # Convert to 0-based index
                        if room_id in disabled_rooms:
                            is_disabled_room_wall = True
                            break

                if is_disabled_room_wall:
                    break

            # If this wall is not adjacent to a disabled room, add it to the filtered pixels
            if not is_disabled_room_wall:
                filtered_pixels.append((x, y, z))

        # Draw the filtered walls
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
        """Get the zone clean from the JSON data with parallel processing."""

        try:
            zone_clean = self.img_h.data.find_zone_entities(m_json)
        except (ValueError, KeyError):
            zone_clean = None
        else:
            _LOGGER.info("%s: Got zones.", self.file_name)

        if zone_clean:
            # Process zones sequentially to avoid memory-intensive array copies
            # This is more memory-efficient than parallel processing with copies

            # Active zones
            zones_active = zone_clean.get("active_zone")
            if zones_active:
                np_array = await self.img_h.draw.zones(
                    np_array, zones_active, color_zone_clean
                )

            # No-go zones
            no_go_zones = zone_clean.get("no_go_area")
            if no_go_zones:
                np_array = await self.img_h.draw.zones(
                    np_array, no_go_zones, color_no_go
                )

            # No-mop zones
            no_mop_zones = zone_clean.get("no_mop_area")
            if no_mop_zones:
                np_array = await self.img_h.draw.zones(
                    np_array, no_mop_zones, color_no_go
                )

        return np_array

    async def async_draw_virtual_walls(
        self, m_json: JsonType, np_array: NumpyArray, color_no_go: Color
    ) -> NumpyArray:
        """Get the virtual walls from the JSON data."""
        try:
            virtual_walls = self.img_h.data.find_virtual_walls(m_json)
        except (ValueError, KeyError):
            virtual_walls = None
        else:
            _LOGGER.info("%s: Got virtual walls.", self.file_name)
        if virtual_walls:
            np_array = await self.img_h.draw.draw_virtual_walls(
                np_array, virtual_walls, color_no_go
            )
        return np_array

    async def async_draw_paths(
        self,
        np_array: NumpyArray,
        m_json: JsonType,
        color_move: Color,
        color_gray: Color,
    ) -> NumpyArray:
        """Get the paths from the JSON data."""
        # Initialize the variables
        path_pixels = None
        predicted_path = None
        # Extract the paths data from the JSON data.
        try:
            paths_data = self.img_h.data.find_paths_entities(m_json)
            predicted_path = paths_data.get("predicted_path", [])
            path_pixels = paths_data.get("path", [])
        except KeyError as e:
            _LOGGER.warning("%s: Error extracting paths data:", str(e))

        if predicted_path:
            predicted_path = predicted_path[0]["points"]
            predicted_path = self.img_h.data.sublist(predicted_path, 2)
            predicted_pat2 = self.img_h.data.sublist_join(predicted_path, 2)
            np_array = await self.img_h.draw.lines(
                np_array, predicted_pat2, 2, color_gray
            )
        if path_pixels:
            for path in path_pixels:
                # Get the points from the current path and extend multiple paths.
                points = path.get("points", [])
                sublist = self.img_h.data.sublist(points, 2)
                self.img_h.shared.map_new_path = self.img_h.data.sublist_join(
                    sublist, 2
                )
                np_array = await self.img_h.draw.lines(
                    np_array, self.img_h.shared.map_new_path, 5, color_move
                )
        return np_array

    def _check_active_zone_and_set_zooming(self) -> None:
        """Helper function to check active zones and set zooming state."""
        if self.img_h.active_zones and self.img_h.robot_in_room:
            segment_id = str(self.img_h.robot_in_room["id"])
            room_store = RoomStore(self.file_name)
            room_keys = list(room_store.get_rooms().keys())

            _LOGGER.debug(
                "%s: Active zones debug - segment_id: %s, room_keys: %s, active_zones: %s",
                self.file_name,
                segment_id,
                room_keys,
                self.img_h.active_zones,
            )

            if segment_id in room_keys:
                position = room_keys.index(segment_id)
                _LOGGER.debug(
                    "%s: Segment ID %s found at position %s, active_zones[%s] = %s",
                    self.file_name,
                    segment_id,
                    position,
                    position,
                    self.img_h.active_zones[position]
                    if position < len(self.img_h.active_zones)
                    else "OUT_OF_BOUNDS",
                )
                if position < len(self.img_h.active_zones):
                    self.img_h.zooming = bool(self.img_h.active_zones[position])
                else:
                    self.img_h.zooming = False
            else:
                _LOGGER.warning(
                    "%s: Segment ID %s not found in room_keys %s",
                    self.file_name,
                    segment_id,
                    room_keys,
                )
                self.img_h.zooming = False
        else:
            self.img_h.zooming = False

    @staticmethod
    def point_in_polygon(x: int, y: int, polygon: list) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            polygon: List of (x, y) tuples forming the polygon

        Returns:
            True if the point is inside the polygon, False otherwise
        """
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        xinters = None  # Initialize with default value
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or (xinters is not None and x <= xinters):
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    async def async_get_robot_in_room(
        self, robot_y: int = 0, robot_x: int = 0, angle: float = 0.0
    ) -> RobotPosition:
        """Get the robot position and return in what room is."""
        # First check if we already have a cached room and if the robot is still in it
        if self.img_h.robot_in_room:
            # If we have outline data, use point_in_polygon for accurate detection
            if "outline" in self.img_h.robot_in_room:
                outline = self.img_h.robot_in_room["outline"]
                if self.point_in_polygon(int(robot_x), int(robot_y), outline):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }
                    # Handle active zones
                    self._check_active_zone_and_set_zooming()
                    return temp
            # Fallback to bounding box check if no outline data
            elif all(
                k in self.img_h.robot_in_room for k in ["left", "right", "up", "down"]
            ):
                if (
                    (self.img_h.robot_in_room["right"] >= int(robot_x))
                    and (self.img_h.robot_in_room["left"] <= int(robot_x))
                ) and (
                    (self.img_h.robot_in_room["down"] >= int(robot_y))
                    and (self.img_h.robot_in_room["up"] <= int(robot_y))
                ):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }
                    # Handle active zones
                    self._check_active_zone_and_set_zooming()
                    return temp

        # If we don't have a cached room or the robot is not in it, search all rooms
        last_room = None
        room_count = 0
        if self.img_h.robot_in_room:
            last_room = self.img_h.robot_in_room

        # Check if the robot is far outside the normal map boundaries
        # This helps prevent false positives for points very far from any room
        map_boundary = 20000  # Typical map size is around 5000-10000 units
        if abs(robot_x) > map_boundary or abs(robot_y) > map_boundary:
            _LOGGER.debug(
                "%s robot position (%s, %s) is far outside map boundaries.",
                self.file_name,
                robot_x,
                robot_y,
            )
            self.img_h.robot_in_room = last_room
            self.img_h.zooming = False
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": last_room["room"] if last_room else None,
            }
            return temp

        # Search through all rooms to find which one contains the robot
        if self.img_h.rooms_pos is None:
            _LOGGER.debug(
                "%s: No rooms data available for robot position detection.",
                self.file_name,
            )
            self.img_h.robot_in_room = last_room
            self.img_h.zooming = False
            temp = {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "in_room": last_room["room"] if last_room else None,
            }
            return temp

        for room in self.img_h.rooms_pos:
            # Check if the room has an outline (polygon points)
            if "outline" in room:
                outline = room["outline"]
                # Use point_in_polygon for accurate detection with complex shapes
                if self.point_in_polygon(int(robot_x), int(robot_y), outline):
                    # Robot is in this room
                    self.img_h.robot_in_room = {
                        "id": room.get(
                            "id", room_count
                        ),  # Use actual segment ID if available
                        "room": str(room["name"]),
                        "outline": outline,
                    }
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }

                    # Handle active zones - Map segment ID to active_zones position
                    if self.img_h.active_zones:
                        segment_id = str(self.img_h.robot_in_room["id"])
                        room_store = RoomStore(self.file_name)
                        room_keys = list(room_store.get_rooms().keys())

                        _LOGGER.debug(
                            "%s: Active zones debug - segment_id: %s, room_keys: %s, active_zones: %s",
                            self.file_name,
                            segment_id,
                            room_keys,
                            self.img_h.active_zones,
                        )

                        if segment_id in room_keys:
                            position = room_keys.index(segment_id)
                            _LOGGER.debug(
                                "%s: Segment ID %s found at position %s, active_zones[%s] = %s",
                                self.file_name,
                                segment_id,
                                position,
                                position,
                                self.img_h.active_zones[position]
                                if position < len(self.img_h.active_zones)
                                else "OUT_OF_BOUNDS",
                            )
                            if position < len(self.img_h.active_zones):
                                self.img_h.zooming = bool(
                                    self.img_h.active_zones[position]
                                )
                            else:
                                self.img_h.zooming = False
                        else:
                            _LOGGER.warning(
                                "%s: Segment ID %s not found in room_keys %s",
                                self.file_name,
                                segment_id,
                                room_keys,
                            )
                            self.img_h.zooming = False
                    else:
                        self.img_h.zooming = False

                    _LOGGER.debug(
                        "%s is in %s room (polygon detection).",
                        self.file_name,
                        self.img_h.robot_in_room["room"],
                    )
                    return temp
            # Fallback to bounding box if no outline is available
            elif "corners" in room:
                corners = room["corners"]
                # Create a bounding box from the corners
                self.img_h.robot_in_room = {
                    "id": room.get(
                        "id", room_count
                    ),  # Use actual segment ID if available
                    "left": int(corners[0][0]),
                    "right": int(corners[2][0]),
                    "up": int(corners[0][1]),
                    "down": int(corners[2][1]),
                    "room": str(room["name"]),
                }
                # Check if the robot is inside the bounding box
                if (
                    (self.img_h.robot_in_room["right"] >= int(robot_x))
                    and (self.img_h.robot_in_room["left"] <= int(robot_x))
                ) and (
                    (self.img_h.robot_in_room["down"] >= int(robot_y))
                    and (self.img_h.robot_in_room["up"] <= int(robot_y))
                ):
                    temp = {
                        "x": robot_x,
                        "y": robot_y,
                        "angle": angle,
                        "in_room": self.img_h.robot_in_room["room"],
                    }

                    # Handle active zones
                    self._check_active_zone_and_set_zooming()

                    _LOGGER.debug(
                        "%s is in %s room (bounding box detection).",
                        self.file_name,
                        self.img_h.robot_in_room["room"],
                    )
                    return temp
            room_count += 1

        # Robot not found in any room
        _LOGGER.debug(
            "%s not located within any room coordinates.",
            self.file_name,
        )
        self.img_h.robot_in_room = last_room
        self.img_h.zooming = False
        temp = {
            "x": robot_x,
            "y": robot_y,
            "angle": angle,
            "in_room": last_room["room"] if last_room else None,
        }
        return temp

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
