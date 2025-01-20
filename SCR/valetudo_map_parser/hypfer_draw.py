"""
Image Draw Class for Valetudo Hypfer Image Handling.
This class is used to simplify the ImageHandler class.
Version: 2024.07.2
"""

from __future__ import annotations

import logging

from .config.types import Color, JsonType, NumpyArray, RobotPosition

_LOGGER = logging.getLogger(__name__)


class ImageDraw:
    """Class to handle the image creation.
    It Draws each element of the images, like the walls, zones, paths, etc."""

    def __init__(self, image_handler):
        self.img_h = image_handler
        self.file_name = self.img_h.shared.file_name

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
    ):
        """Draw the base layer of the map."""
        room_id = 0

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
                    img_np_array, pixels, pixel_size, color_wall
                )

        return room_id, img_np_array

    async def _process_room_layer(
        self, img_np_array, pixels, layer_type, room_id, pixel_size, color_zone_clean
    ):
        """Process a room layer (segment or floor)."""
        room_color = self.img_h.shared.rooms_colors[room_id]

        try:
            if layer_type == "segment":
                room_color = self._get_active_room_color(
                    room_id, room_color, color_zone_clean
                )

            img_np_array = await self.img_h.draw.from_json_to_image(
                img_np_array, pixels, pixel_size, room_color
            )
            room_id = (room_id + 1) % 16  # Cycle room_id back to 0 after 15

        except IndexError as e:
            _LOGGER.warning("%s: Image Draw Error: %s", self.file_name, str(e))
        _LOGGER.debug(
            "%s Active Zones: %s and Room ID: %s",
            self.file_name,
            str(self.img_h.active_zones),
            str(room_id),
        )

        return img_np_array, room_id

    def _get_active_room_color(self, room_id, room_color, color_zone_clean):
        """Adjust the room color if the room is active."""
        if self.img_h.active_zones and room_id < len(self.img_h.active_zones):
            if self.img_h.active_zones[room_id] == 1:
                return tuple(
                    ((2 * room_color[i]) + color_zone_clean[i]) // 3 for i in range(4)
                )
        return room_color

    async def _process_wall_layer(self, img_np_array, pixels, pixel_size, color_wall):
        """Process a wall layer."""
        return await self.img_h.draw.from_json_to_image(
            img_np_array, pixels, pixel_size, color_wall
        )

    async def async_draw_obstacle(
        self, np_array: NumpyArray, entity_dict: dict, color_no_go: Color
    ) -> NumpyArray:
        """Get the obstacle positions from the entity data."""
        try:
            obstacle_data = entity_dict.get("obstacle")
        except KeyError:
            _LOGGER.info("%s No obstacle found.", self.file_name)
        else:
            obstacle_positions = []
            if obstacle_data:
                for obstacle in obstacle_data:
                    label = obstacle.get("metaData", {}).get("label")
                    points = obstacle.get("points", [])

                    if label and points:
                        obstacle_pos = {
                            "label": label,
                            "points": {"x": points[0], "y": points[1]},
                        }
                        obstacle_positions.append(obstacle_pos)

            # List of dictionaries containing label and points for each obstacle
            # and draw obstacles on the map
            if obstacle_positions:
                self.img_h.draw.draw_obstacles(
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
        else:
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

    async def async_get_json_id(self, my_json: JsonType) -> str | None:
        """Return the JSON ID from the image."""
        try:
            json_id = my_json["metaData"]["nonce"]
        except (ValueError, KeyError) as e:
            _LOGGER.debug("%s: No JsonID provided: %s", self.file_name, str(e))
            json_id = None
        return json_id

    async def async_draw_zones(
        self,
        m_json: JsonType,
        np_array: NumpyArray,
        color_zone_clean: Color,
        color_no_go: Color,
    ) -> NumpyArray:
        """Get the zone clean from the JSON data."""
        try:
            zone_clean = self.img_h.data.find_zone_entities(m_json)
        except (ValueError, KeyError):
            zone_clean = None
        else:
            _LOGGER.info("%s: Got zones.", self.file_name)
        if zone_clean:
            try:
                zones_active = zone_clean.get("active_zone")
            except KeyError:
                zones_active = None
            if zones_active:
                np_array = await self.img_h.draw.zones(
                    np_array, zones_active, color_zone_clean
                )
            try:
                no_go_zones = zone_clean.get("no_go_area")
            except KeyError:
                no_go_zones = None

            if no_go_zones:
                np_array = await self.img_h.draw.zones(
                    np_array, no_go_zones, color_no_go
                )

            try:
                no_mop_zones = zone_clean.get("no_mop_area")
            except KeyError:
                no_mop_zones = None

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

    async def async_get_entity_data(self, m_json: JsonType) -> dict or None:
        """Get the entity data from the JSON data."""
        try:
            entity_dict = self.img_h.data.find_points_entities(m_json)
        except (ValueError, KeyError):
            return None
        _LOGGER.info("%s: Got the points in the json.", self.file_name)
        return entity_dict

    async def async_get_robot_in_room(
        self, robot_y: int = 0, robot_x: int = 0, angle: float = 0.0
    ) -> RobotPosition:
        """Get the robot position and return in what room is."""
        if self.img_h.robot_in_room:
            # Check if the robot coordinates are inside the room's corners
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
                if self.img_h.active_zones and (
                    self.img_h.robot_in_room["id"]
                    in range(len(self.img_h.active_zones))
                ):  # issue #100 Index out of range.
                    self.img_h.zooming = bool(
                        self.img_h.active_zones[self.img_h.robot_in_room["id"]]
                    )
                else:
                    self.img_h.zooming = False
                return temp
        # else we need to search and use the async method.
        if self.img_h.rooms_pos:
            last_room = None
            room_count = 0
            if self.img_h.robot_in_room:
                last_room = self.img_h.robot_in_room
            for room in self.img_h.rooms_pos:
                corners = room["corners"]
                self.img_h.robot_in_room = {
                    "id": room_count,
                    "left": int(corners[0][0]),
                    "right": int(corners[2][0]),
                    "up": int(corners[0][1]),
                    "down": int(corners[2][1]),
                    "room": str(room["name"]),
                }
                room_count += 1
                # Check if the robot coordinates are inside the room's corners
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
                    _LOGGER.debug(
                        "%s is in %s room.",
                        self.file_name,
                        self.img_h.robot_in_room["room"],
                    )
                    del room, corners, robot_x, robot_y  # free memory.
                    return temp
            del room, corners  # free memory.
            _LOGGER.debug(
                "%s not located within Camera Rooms coordinates.",
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
            # If the robot is not inside any room, return a default value
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
