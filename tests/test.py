from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import cProfile
import pstats

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from SCR package
from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import RoomStore
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler
from SCR.valetudo_map_parser.hypfer_draw import ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensures DEBUG logs are displayed
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


class TestImageHandler:
    def __init__(self):
        self.test_data = None
        self.image = None

    def setUp(self):
        # Load the test.json file
        test_file_path = os.path.join(os.path.dirname(__file__), "test.json")
        _LOGGER.info(f"Loading test data from {test_file_path}")

        with open(test_file_path, "r") as file:
            self.test_data = json.load(file)
            _LOGGER.debug("Test data loaded.")

    async def test_image_handler(self):
        _LOGGER.info("Starting test_image_handler...")

        device_info = {
            'platform': 'mqtt_vacuum_camera',
            'unique_id': 'rockrobo_camera',
            'vacuum_config_entry': '9abe4e81972b00d1682c2363d3584041',
            'vacuum_map': 'valetudo/rockrobo',
            'vacuum_identifiers': {('mqtt', 'rockrobo')},
            'coordinator': "<custom_components.mqtt_vacuum_camera.coordinator.MQTTVacuumCoordinator object>",
            'is_rand256': False,
            'unsub_options_update_listener': "<function ConfigEntry.add_update_listener.<locals>.<lambda>>",
            'alpha_background': 255.0,
            'alpha_charger': 255.0,
            'alpha_go_to': 255.0,
            'alpha_move': 150.0,  # Higher alpha for better visibility
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 150.0,  # Testing with a lower alpha value
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
            'zoom_lock_ratio': True,
            'color_background': [0, 125, 255],
            'color_charger': [255, 128, 0],
            'color_go_to': [0, 255, 0],
            'color_move': [238, 247, 255],  # More vibrant blue for better visibility
            'color_no_go': [255, 0, 0],
            'color_robot': [255, 255, 204],
            'color_text': [164, 25, 25],
            'color_wall': [255, 255, 0],
            'color_zone_clean': [255, 255, 255],
            'color_room_0': [135, 206, 250],
            'color_room_1': [176, 226, 255],
            'color_room_2': [165, 141, 18],
            'color_room_3': [164, 211, 238],
            'color_room_4': [141, 182, 205],
            'color_room_5': [96, 123, 139],
            'color_room_6': [224, 255, 255],
            'color_room_7': [209, 238, 238],
            'color_room_8': [180, 205, 205],
            'color_room_9': [122, 139, 139],
            'color_room_10': [175, 238, 238],
            'color_room_11': [84, 153, 199],
            'color_room_12': [133, 193, 233],
            'color_room_13': [245, 176, 65],
            'color_room_14': [82, 190, 128],
            'color_room_15': [72, 201, 176],
            'alpha_room_0': 155.0,
            'alpha_room_1': 155.0,
            'alpha_room_2': 155.0,
            'alpha_room_3': 155.0,
            'alpha_room_4': 155.0,
            'alpha_room_5': 155.0,
            'alpha_room_6': 155.0,
            'alpha_room_7': 155.0,
            'alpha_room_8': 255.0,
            'alpha_room_9': 255.0,
            'alpha_room_10': 255.0,
            'alpha_room_11': 255.0,
            'alpha_room_12': 255.0,
            'alpha_room_13': 255.0,
            'alpha_room_14': 255.0,
            'alpha_room_15': 255.0,
            'offset_top': 0,
            'offset_bottom': 0,
            'offset_left': 10,
            'offset_right': 0,
            'rotate_image': '180',
            'margins': '100',
            'show_vac_status': False,
            'vac_status_font': 'custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf',
            'vac_status_position': True,
            'vac_status_size': 50.0,
            'enable_www_snapshots': False,
            'get_svg_file': False,
            'trims_data': {
                'trim_left': 1980,
                'trim_up': 1650,
                'trim_right': 3974,
                'trim_down': 3474
            },
            'disable_floor': False,
            'disable_wall': False,
            'disable_robot': False,
            'disable_charger': False,
            'disable_virtual_walls': False,
            'disable_restricted_areas': False,
            'disable_no_mop_areas': False,
            'disable_obstacles': False,
            'disable_path': False,
            'disable_predicted_path': False,
            'disable_go_to_target': False,
            'disable_room_1': False,
            'disable_room_2': False,
            'disable_room_3': False,
            'disable_room_4': False,
            'disable_room_5': False,
            'disable_room_6': False,
            'disable_room_7': False,
            'disable_room_8': False,
            'disable_room_9': False,
            'disable_room_10': False,
            'disable_room_11': False,
            'disable_room_12': False,
            'disable_room_13': False,
            'disable_room_14': False,
            'disable_room_15': False
        }

        shared_data = CameraSharedManager("test_vacuum", device_info)
        shared = shared_data.get_instance()
        shared.vacuum_state = "cleaning"
        shared.vacuum_ips = "192.168.1.1"

        # Set up active zones for rooms - True means zoom will be enabled for that room
        # For Hypfer handler, we need to set the active zones in the handler instance
        shared.active_zones = [1] * 15  # Enable all rooms as active zones

        _LOGGER.debug(f"Shared instance trims: {shared.trims}")

        colors = ColorsManagement(shared)
        colors.set_initial_colours(device_info)

        _LOGGER.debug(f"Colors initialized: {shared.user_colors}")

        handler = HypferMapImageHandler(shared)

        # Set active zones in the handler instance
        handler.imd = ImageDraw(handler)
        handler.imd.file_name = "test_vacuum"
        handler.imd.img_h.active_zones = [1] * 15  # Enable all rooms as active zones

        # Get the image with elements disabled from device_info
        self.image = await handler.async_get_image_from_json(self.test_data)
        if self.image is None:
            _LOGGER.error("Failed to generate image from JSON data")
            return

        _LOGGER.info(f"Calibration_data: {handler.get_calibration_data()}")
        _LOGGER.info(f"Image size: {self.image.size}")
        store = RoomStore("test_vacuum")
        rooms_data = await handler.async_get_rooms_attributes()
        _LOGGER.info(f"Room Properties: {rooms_data}")
        count = store.get_rooms_count()
        _LOGGER.info(f"Room Store Properties: {count}")
        rooms = store.get_rooms()
        instaces = RoomStore.get_all_instances()
        _LOGGER.info(f"Room Store Rooms {instaces}: {rooms}")
        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")
        calibration_data = handler.get_calibration_data()
        _LOGGER.info(f"Calibration Data: {calibration_data}")
        robot_x = handler.robot_pos.get("x")
        robot_y = handler.robot_pos.get("y")
        robot_angle = handler.robot_pos.get("angle")
        _LOGGER.info(f"Robot Position: ({robot_x}, {robot_y}, {robot_angle})")
        # Test the robot detection function
        result = await handler.imd.async_get_robot_in_room(robot_y=robot_y, robot_x=robot_x, angle=robot_angle)
        _LOGGER.info(f"Robot in room: {result}")

        # Check if zooming is enabled
        _LOGGER.info(f"Zooming enabled: {handler.imd.img_h.zooming}")

        # Check if all zooming conditions are met
        _LOGGER.info(f"Zoom conditions: zoom={handler.imd.img_h.zooming}, vacuum_state={handler.shared.vacuum_state}, image_auto_zoom={handler.shared.image_auto_zoom}")

        # Debug: Check the device_info auto_zoom setting
        _LOGGER.info(f"Device info auto_zoom: {device_info.get('auto_zoom')}")
        _LOGGER.info(f"Shared image_auto_zoom: {handler.shared.image_auto_zoom}")

        # Debug: Check the zooming flags
        _LOGGER.info(f"Handler zooming: {handler.zooming}")
        _LOGGER.info(f"ImageDraw img_h zooming: {handler.imd.img_h.zooming}")
        _LOGGER.info(f"Active zones: {handler.imd.img_h.active_zones}")

        # Generate attributes to populate obstacles_data
        if shared.obstacles_pos:
            _LOGGER.info(f"Obstacles positions found: {shared.obstacles_pos}")
            # Call generate_attributes to populate obstacles_data
            attrs = shared.generate_attributes()
            _LOGGER.info(f"Generated attributes: {attrs}")
        else:
            _LOGGER.info("No obstacles positions found in the map data")

        _LOGGER.info(f"Obstacles data: {shared.obstacles_data}")
        #_LOGGER.info("Testing robot detection in each room...")
        success_count = 0

        # # Set the rooms_pos attribute in the handler.imd object
        # # Convert the rooms dictionary to a list of room objects
        # rooms_list = []
        # for room_id, props in store.get_rooms().items():
        #     room_obj = {
        #         'name': props['name'],
        #         'outline': props['outline']
        #     }
        #     rooms_list.append(room_obj)
        #
        # # Set the rooms_pos attribute
        # handler.imd.img_h.rooms_pos = rooms_list
        #
        # for room_id, props in store.get_rooms().items():
        #     # Use the room's center coordinates as the robot position
        #     robot_x = props['x']
        #     robot_y = props['y']
        #
        #     # Verify that the point is actually inside the polygon using our algorithm
        #     is_inside = handler.imd.point_in_polygon(robot_x, robot_y, props['outline'])
        #     if not is_inside:
        #         _LOGGER.warning(f"⚠️ Center point ({robot_x}, {robot_y}) is not inside room {room_id}: {props['name']}")
        #         # Try to find a better test point by averaging some points from the outline
        #         points = props['outline']
        #         if len(points) >= 3:
        #             # Use the average of the first 3 points as an alternative test point
        #             alt_x = sum(p[0] for p in points[:3]) // 3
        #             alt_y = sum(p[1] for p in points[:3]) // 3
        #             if handler.imd.point_in_polygon(alt_x, alt_y, props['outline']):
        #                 _LOGGER.info(f"   Using alternative point ({alt_x}, {alt_y}) for testing")
        #                 robot_x, robot_y = alt_x, alt_y
        #
        #     # Call the function to detect which room the robot is in
        #     #result = await handler.imd.async_get_robot_in_room(robot_y=robot_y, robot_x=robot_x)
        #     #_LOGGER.info(f"Robot in room: {result}")



        self.image.show()


def __main__():
    test = TestImageHandler()
    test.setUp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        loop.run_until_complete(test.test_image_handler())
    finally:
        profiler.disable()
        loop.close()

        # Save profiling data
        profile_output = "hypfer_profile_output.prof"
        profiler.dump_stats(profile_output)

        # Print profiling summary
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)  # Show top 50 functions


if __name__ == "__main__":
    __main__()
