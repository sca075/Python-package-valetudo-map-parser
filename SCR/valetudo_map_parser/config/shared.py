"""
Class Camera Shared.
Keep the data between the modules.
Version: v2024.12.0
"""

import asyncio
import logging
from typing import List

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
)


_LOGGER = logging.getLogger(__name__)


class CameraShared:
    """
    CameraShared class to keep the data between the classes.
    Implements a kind of Thread Safe data shared area.
    """

    def __init__(self, file_name):
        self.camera_mode: str = CameraModes.MAP_VIEW  # Camera mode
        self.frame_number: int = 0  # camera Frame number
        self.destinations: list = []  # MQTT rand destinations
        self.rand256_active_zone: list = []  # Active zone for rand256
        self.is_rand: bool = False  # MQTT rand data
        self._new_mqtt_message = False  # New MQTT message
        self.last_image = None  # Last image received
        self.image_size = None  # Image size
        self.image_auto_zoom: bool = False  # Auto zoom image
        self.image_zoom_lock_ratio: bool = True  # Zoom lock ratio
        self.image_ref_height: int = 0  # Image reference height
        self.image_ref_width: int = 0  # Image reference width
        self.image_aspect_ratio: str = "None"  # Change Image aspect ratio
        self.image_grab = True  # Grab image from MQTT
        self.image_rotate: int = 0  # Rotate image
        self.drawing_limit: float = 0.0  # Drawing CPU limit
        self.current_room = None  # Current room of rhe vacuum
        self.user_colors = Colors  # User base colors
        self.rooms_colors = Colors  # Rooms colors
        self.vacuum_battery = None  # Vacuum battery state
        self.vacuum_bat_charged: bool = True  # Vacuum charged and ready
        self.vacuum_connection = None  # Vacuum connection state
        self.vacuum_state = None  # Vacuum state
        self.charger_position = None  # Vacuum Charger position
        self.show_vacuum_state = None  # Show vacuum state on the map
        self.vacuum_status_font: str = (
            "custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf"  # Font
        )
        self.vacuum_status_size: int = 50  # Vacuum status size
        self.vacuum_status_position: bool = True  # Vacuum status text image top
        self.snapshot_take = False  # Take snapshot
        self.vacuum_error = None  # Vacuum error
        self.vacuum_api = None  # Vacuum API
        self.vacuum_ips = None  # Vacuum IPs
        self.vac_json_id = None  # Vacuum json id
        self.margins = "100"  # Image margins
        self.obstacles_data = None  # Obstacles data
        self.obstacles_pos = None  # Obstacles position
        self.offset_top = 0  # Image offset top
        self.offset_down = 0  # Image offset down
        self.offset_left = 0  # Image offset left
        self.offset_right = 0  # Image offset right
        self.export_svg = False  # Export SVG
        self.svg_path = None  # SVG Export path
        self.enable_snapshots = False  # Enable snapshots
        self.file_name = file_name  # vacuum friendly name as File name
        self.attr_calibration_points = None  # Calibration points of the image
        self.map_rooms = None  # Rooms data from the vacuum
        self.map_pred_zones = None  # Predefined zones data
        self.map_pred_points = None  # Predefined points data
        self.map_new_path = None  # New path data
        self.map_old_path = None  # Old path data
        self.user_language = None  # User language
        self.trim_crop_data = None
        self.trims = TrimsData.from_dict(DEFAULT_VALUES["trims_data"])  # Trims data
        self.skip_room_ids: List[str] = []
        self.device_info = None  # Store the device_info

    @staticmethod
    def _compose_obstacle_links(vacuum_host_ip: str, obstacles: list) -> list | None:
        """
        Compose JSON with obstacle details including the image link.
        """
        obstacle_links = []
        if not obstacles or not vacuum_host_ip:
            return None

        for obstacle in obstacles:
            # Extract obstacle details
            label = obstacle.get("label", "")
            points = obstacle.get("points", {})
            image_id = obstacle.get("id", "None")

            if label and points and image_id and vacuum_host_ip:
                # Append formatted obstacle data
                if image_id != "None":
                    # Compose the link
                    image_link = (
                        f"http://{vacuum_host_ip}"
                        f"/api/v2/robot/capabilities/ObstacleImagesCapability/img/{image_id}"
                    )
                    obstacle_links.append(
                        {
                            "point": points,
                            "label": label,
                            "link": image_link,
                        }
                    )
                else:
                    obstacle_links.append(
                        {
                            "point": points,
                            "label": label,
                        }
                    )
        return obstacle_links

    def update_user_colors(self, user_colors):
        """Update the user colors."""
        self.user_colors = user_colors

    def get_user_colors(self):
        """Get the user colors."""
        return self.user_colors

    def update_rooms_colors(self, user_colors):
        """Update the rooms colors."""
        self.rooms_colors = user_colors

    def get_rooms_colors(self):
        """Get the rooms colors."""
        return self.rooms_colors

    def reset_trims(self) -> dict:
        """Reset the trims."""
        self.trims = TrimsData.from_dict(DEFAULT_VALUES["trims_data"])
        return self.trims

    async def batch_update(self, **kwargs):
        """Batch update multiple attributes."""
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
            ATTR_VACUUM_POSITION: self.current_room,
            ATTR_VACUUM_STATUS: self.vacuum_state,
            ATTR_VACUUM_JSON_ID: self.vac_json_id,
            ATTR_CALIBRATION_POINTS: self.attr_calibration_points,
        }
        if self.obstacles_pos and self.vacuum_ips:
            _LOGGER.debug("Generating obstacle links from: %s", self.obstacles_pos)
            self.obstacles_data =  self._compose_obstacle_links(
                self.vacuum_ips, self.obstacles_pos
            )
            attrs[ATTR_OBSTACLES] = self.obstacles_data

        if self.enable_snapshots:
            attrs[ATTR_SNAPSHOT] = self.snapshot_take
        else:
            attrs[ATTR_SNAPSHOT] = False

        # Add dynamic shared attributes if they are available
        shared_attrs = {
            ATTR_ROOMS: self.map_rooms,
            ATTR_ZONES: self.map_pred_zones,
            ATTR_POINTS: self.map_pred_points,
        }

        for key, value in shared_attrs.items():
            if value is not None:
                attrs[key] = value

        return attrs


class CameraSharedManager:
    """Camera Shared Manager class."""

    def __init__(self, file_name, device_info):
        self._instances = {}
        self._lock = asyncio.Lock()
        self.file_name = file_name
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
            _LOGGER.debug(
                "%s: Updating shared trims with: %s", instance.file_name, trim_data
            )
            instance.trims = TrimsData.from_dict(trim_data)

        except TypeError as ex:
            _LOGGER.error("Shared data can't be initialized due to a TypeError! %s", ex)
        except AttributeError as ex:
            _LOGGER.error(
                "Shared data can't be initialized due to an AttributeError! %s", ex
            )
        except RuntimeError as ex:
            _LOGGER.error(
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
