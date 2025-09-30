"""
Class Camera Shared.
Keep the data between the modules.
Version: v0.1.12
"""

import asyncio
import logging
from typing import List
from PIL import Image

from .types import (
    ATTR_CALIBRATION_POINTS,
    ATTR_CAMERA_MODE,
    ATTR_MARGINS,
    ATTR_OBSTACLES,
    ATTR_POINTS,
    ATTR_ROOMS,
    ATTR_ROTATE,
    ATTR_SNAPSHOT,
    ATTR_VACUUM_BATTERY,
    ATTR_VACUUM_CHARGING,
    ATTR_VACUUM_JSON_ID,
    ATTR_VACUUM_POSITION,
    ATTR_VACUUM_STATUS,
    ATTR_ZONES,
    CONF_ASPECT_RATIO,
    CONF_AUTO_ZOOM,
    CONF_OFFSET_BOTTOM,
    CONF_OFFSET_LEFT,
    CONF_OFFSET_RIGHT,
    CONF_OFFSET_TOP,
    CONF_SNAPSHOTS_ENABLE,
    CONF_VAC_STAT,
    CONF_VAC_STAT_FONT,
    CONF_VAC_STAT_POS,
    CONF_VAC_STAT_SIZE,
    CONF_ZOOM_LOCK_RATIO,
    DEFAULT_VALUES,
    CameraModes,
    Colors,
    TrimsData,
    PilPNG,
)


_LOGGER = logging.getLogger(__name__)


class CameraShared:
    """
    CameraShared class to keep the data between the classes.
    Implements a kind of Thread Safe data shared area.
    """

    def __init__(self, file_name):
        self.camera_mode: str = CameraModes.MAP_VIEW
        self.frame_number: int = 0
        self.destinations: list = []
        self.rand256_active_zone: list = []
        self.rand256_zone_coordinates: list = []
        self.is_rand: bool = False
        self._new_mqtt_message = False
        self.last_image = Image.new("RGBA", (250, 150), (128, 128, 128, 255))
        self.new_image: PilPNG | None = None
        self.binary_image: bytes | None = None
        self.image_last_updated: float = 0.0
        self.image_format = "image/pil"
        self.image_size = None
        self.robot_size = None
        self.image_auto_zoom: bool = False
        self.image_zoom_lock_ratio: bool = True
        self.image_ref_height: int = 0
        self.image_ref_width: int = 0
        self.image_aspect_ratio: str = "None"
        self.image_grab = True
        self.image_rotate: int = 0
        self.drawing_limit: float = 0.0
        self.current_room = None
        self.user_colors = Colors
        self.rooms_colors = Colors
        self.vacuum_battery = 0
        self.vacuum_connection = False
        self.vacuum_state = None
        self.charger_position = None
        self.show_vacuum_state = None
        self.vacuum_status_font: str = (
            "custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf"
        )
        self.vacuum_status_size: int = 50
        self.vacuum_status_position: bool = True
        self.snapshot_take = False
        self.vacuum_error = None
        self.vacuum_api = None
        self.vacuum_ips = None
        self.vac_json_id = None
        self.margins = "100"
        self.obstacles_data = None
        self.obstacles_pos = None
        self.offset_top = 0
        self.offset_down = 0
        self.offset_left = 0
        self.offset_right = 0
        self.export_svg = False
        self.svg_path = None
        self.enable_snapshots = False
        self.file_name = file_name
        self.attr_calibration_points = None
        self.map_rooms = None
        self.map_pred_zones = None
        self.map_pred_points = None
        self.map_new_path = None
        self.map_old_path = None
        self.user_language = None
        self.trim_crop_data = None
        self.trims = TrimsData.from_dict(DEFAULT_VALUES["trims_data"])
        self.skip_room_ids: List[str] = []
        self.device_info = None

    def vacuum_bat_charged(self) -> bool:
        """Check if the vacuum is charging."""
        return (self.vacuum_state == "docked") and (int(self.vacuum_battery) < 100)

    @staticmethod
    def _compose_obstacle_links(vacuum_host_ip: str, obstacles: list) -> list | None:
        """Compose JSON with obstacle details including the image link."""
        obstacle_links = []
        if not obstacles or not vacuum_host_ip:
            return None

        for obstacle in obstacles:
            label = obstacle.get("label", "")
            points = obstacle.get("points", {})
            image_id = obstacle.get("id", "None")

            if label and points and image_id and vacuum_host_ip:
                if image_id != "None":
                    image_link = (
                        f"http://{vacuum_host_ip}"
                        f"/api/v2/robot/capabilities/ObstacleImagesCapability/img/{image_id}"
                    )
                    obstacle_links.append(
                        {"point": points, "label": label, "link": image_link}
                    )
                else:
                    obstacle_links.append({"point": points, "label": label})
        return obstacle_links

    def update_user_colors(self, user_colors):
        """Update user colors palette"""
        self.user_colors = user_colors

    def get_user_colors(self):
        """Return user colors"""
        return self.user_colors

    def update_rooms_colors(self, user_colors):
        """Update the rooms colors."""
        self.rooms_colors = user_colors

    def get_rooms_colors(self):
        """Return rooms colors"""
        return self.rooms_colors

    def reset_trims(self) -> dict:
        """Reset the trims."""
        self.trims = TrimsData.from_dict(DEFAULT_VALUES["trims_data"])
        return self.trims

    async def batch_update(self, **kwargs):
        """Update the data of Shared in Batch"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    async def batch_get(self, *args):
        """Batch get multiple attributes."""
        return {key: getattr(self, key) for key in args}

    def generate_attributes(self) -> dict:
        """Generate and return the shared attribute's dictionary."""
        attrs = {
            ATTR_CAMERA_MODE: self.camera_mode,
            ATTR_VACUUM_BATTERY: f"{self.vacuum_battery}%",
            ATTR_VACUUM_CHARGING: self.vacuum_bat_charged(),
            ATTR_VACUUM_POSITION: self.current_room,
            ATTR_VACUUM_STATUS: self.vacuum_state,
            ATTR_VACUUM_JSON_ID: self.vac_json_id,
            ATTR_CALIBRATION_POINTS: self.attr_calibration_points,
        }
        if self.obstacles_pos and self.vacuum_ips:
            self.obstacles_data = self._compose_obstacle_links(
                self.vacuum_ips, self.obstacles_pos
            )
            attrs[ATTR_OBSTACLES] = self.obstacles_data

        attrs[ATTR_SNAPSHOT] = self.snapshot_take if self.enable_snapshots else False

        shared_attrs = {
            ATTR_ROOMS: self.map_rooms,
            ATTR_ZONES: self.map_pred_zones,
            ATTR_POINTS: self.map_pred_points,
        }
        for key, value in shared_attrs.items():
            if value is not None:
                attrs[key] = value

        return attrs

    def to_dict(self) -> dict:
        """Return a dictionary with image and attributes data."""
        return {
            "image": {
                "binary": self.binary_image,
                "pil_image_size": self.new_image.size,
                "size": self.new_image.size if self.new_image else None,
                "format": self.image_format,
                "updated": self.image_last_updated,
            },
            "attributes": self.generate_attributes(),
        }


class CameraSharedManager:
    """Camera Shared Manager class."""

    def __init__(self, file_name: str, device_info: dict = None):
        self._instances = {}
        self._lock = asyncio.Lock()
        self.file_name = file_name
        if device_info:
            self.device_info = device_info
            self.update_shared_data(device_info)

        # Automatically initialize shared data for the instance
        # self._init_shared_data(device_info)

    def update_shared_data(self, device_info):
        """Initialize the shared data with device_info."""
        instance = self.get_instance()  # Retrieve the correct instance

        try:
            # Store the device_info in the instance
            instance.device_info = device_info
            _LOGGER.info(
                "%s: Stored device_info in shared instance", instance.file_name
            )

            instance.attr_calibration_points = None

            # Initialize shared data with defaults from DEFAULT_VALUES
            instance.offset_top = device_info.get(
                CONF_OFFSET_TOP, DEFAULT_VALUES["offset_top"]
            )
            instance.offset_down = device_info.get(
                CONF_OFFSET_BOTTOM, DEFAULT_VALUES["offset_bottom"]
            )
            instance.offset_left = device_info.get(
                CONF_OFFSET_LEFT, DEFAULT_VALUES["offset_left"]
            )
            instance.offset_right = device_info.get(
                CONF_OFFSET_RIGHT, DEFAULT_VALUES["offset_right"]
            )
            instance.image_auto_zoom = device_info.get(
                CONF_AUTO_ZOOM, DEFAULT_VALUES["auto_zoom"]
            )
            instance.image_zoom_lock_ratio = device_info.get(
                CONF_ZOOM_LOCK_RATIO, DEFAULT_VALUES["zoom_lock_ratio"]
            )
            instance.image_aspect_ratio = device_info.get(
                CONF_ASPECT_RATIO, DEFAULT_VALUES["aspect_ratio"]
            )
            instance.image_rotate = int(
                device_info.get(ATTR_ROTATE, DEFAULT_VALUES["rotate_image"])
            )
            instance.margins = int(
                device_info.get(ATTR_MARGINS, DEFAULT_VALUES["margins"])
            )
            instance.show_vacuum_state = device_info.get(
                CONF_VAC_STAT, DEFAULT_VALUES["show_vac_status"]
            )
            instance.vacuum_status_font = device_info.get(
                CONF_VAC_STAT_FONT, DEFAULT_VALUES["vac_status_font"]
            )
            instance.vacuum_status_size = device_info.get(
                CONF_VAC_STAT_SIZE, DEFAULT_VALUES["vac_status_size"]
            )
            instance.vacuum_status_position = device_info.get(
                CONF_VAC_STAT_POS, DEFAULT_VALUES["vac_status_position"]
            )
            # If enable_snapshots, check for png in www.
            instance.enable_snapshots = device_info.get(
                CONF_SNAPSHOTS_ENABLE, DEFAULT_VALUES["enable_www_snapshots"]
            )
            # Ensure trims are updated correctly
            trim_data = device_info.get("trims_data", DEFAULT_VALUES["trims_data"])
            instance.trims = TrimsData.from_dict(trim_data)
            # Robot size
            robot_size = device_info.get("robot_size", 25)
            try:
                robot_size = int(robot_size)
            except (ValueError, TypeError):
                robot_size = 25
            # Clamp robot_size to [8, 25]
            if robot_size < 8:
                robot_size = 8
            elif robot_size > 25:
                robot_size = 25
            instance.robot_size = robot_size

        except TypeError as ex:
            _LOGGER.warning(
                "Shared data can't be initialized due to a TypeError! %s", ex
            )
        except AttributeError as ex:
            _LOGGER.warning(
                "Shared data can't be initialized due to an AttributeError! %s", ex
            )
        except RuntimeError as ex:
            _LOGGER.warning(
                "An unexpected error occurred while initializing shared data %s:", ex
            )

    def get_instance(self):
        """Get the shared instance."""
        if self.file_name not in self._instances:
            self._instances[self.file_name] = CameraShared(self.file_name)
            self._instances[self.file_name].file_name = self.file_name
        return self._instances[self.file_name]

    async def update_instance(self, **kwargs):
        """Update the shared instance."""
        async with self._lock:
            instance = self.get_instance()
            await instance.batch_update(**kwargs)
