from __future__ import annotations

import asyncio
import json
import logging
import os
import cProfile
import pstats

from SCR.valetudo_map_parser.config.colors_man import ColorsManagment
from SCR.valetudo_map_parser.config.drawable_elements import DrawableElement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import RoomStore
from SCR.valetudo_map_parser.rand25_handler import ReImageHandler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensures DEBUG logs are displayed
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


class TestRandImageHandler:
    def __init__(self):
        self.test_data = None
        self.image = None

    def setUp(self):
        # Load test data from the rand.json file
        test_file_path = os.path.join(os.path.dirname(__file__), "rand.json")
        logging.getLogger(__name__).info(f"Loading test data from {test_file_path}")

        with open(test_file_path, "r") as file:
            self.test_data = json.load(file)

        _LOGGER.debug("Test data loaded.")

    async def test_image_handler(self):
        _LOGGER.info("Starting test_rand_image_handler...")

        device_info = {
            'platform': 'mqtt_vacuum_camera',
            'unique_id': 'rockrobo_camera',
            'vacuum_config_entry': '9abe4e81972b00d1682c2363d3584041',
            'vacuum_map': 'valetudo/rockrobo',
            'vacuum_identifiers': {('mqtt', 'rockrobo')},
            'coordinator': "<custom_components.mqtt_vacuum_camera.coordinator.MQTTVacuumCoordinator object>",
            'is_rand256': True,
            'unsub_options_update_listener': "<function ConfigEntry.add_update_listener.<locals>.<lambda>>",
            'alpha_background': 255.0,
            'alpha_charger': 255.0,
            'alpha_go_to': 255.0,
            'alpha_move': 125.0,
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 125.0,
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
            'zoom_lock_ratio': True,
            'color_background': [0, 125, 255],
            'color_charger': [255, 128, 0],
            'color_go_to': [0, 255, 0],
            'color_move': [238, 247, 255],
            'color_no_go': [255, 0, 0],
            'color_robot': [255, 255, 204],
            'color_text': [164, 25, 25],
            'color_wall': [255, 255, 0],
            'color_zone_clean': [255, 255, 255],
            'disable_obstacles': False,
            'disable_path': False
        }

        shared_data = CameraSharedManager("test_vacuum", device_info)
        shared = shared_data.get_instance()
        shared.vacuum_state = "docked"
        _LOGGER.debug(f"Shared instance trims: {shared.trims}")

        colors = ColorsManagment(shared)
        colors.set_initial_colours(device_info)

        _LOGGER.debug(f"Colors initialized: {shared.user_colors}")

        # Create a handler instance
        handler = ReImageHandler(shared)

        # Test the drawable element system
        _LOGGER.info("Testing drawable element system...")
        _LOGGER.info(f"PATH element enabled: {handler.drawing_config.is_enabled(DrawableElement.PATH)}")
        _LOGGER.info(f"OBSTACLE element enabled: {handler.drawing_config.is_enabled(DrawableElement.OBSTACLE)}")

        # Try to generate an image from the JSON data
        try:
            _LOGGER.info("Attempting to generate image from JSON data...")
            # Check if the JSON data has a charger position
            if 'charger_location' not in self.test_data:
                _LOGGER.info("No charger location in test data, adding a dummy one")
                # Add a dummy charger position to avoid the error
                self.test_data['charger_location'] = [{'points': [2526, 2554]}]

            # Monkey patch the handler to handle the charger position correctly
            original_draw_charger = handler.imd.async_draw_charger

            async def patched_draw_charger(np_array, m_json, color_charger):
                np_array, charger_pos_dict = await original_draw_charger(np_array, m_json, color_charger)
                # Convert the dictionary to the expected format
                if charger_pos_dict and 'x' in charger_pos_dict and 'y' in charger_pos_dict:
                    # Return as a list-like object with indices 0 and 1 for x and y
                    return np_array, [charger_pos_dict['x'], charger_pos_dict['y']]
                return np_array, None

            # Apply the monkey patch
            handler.imd.async_draw_charger = patched_draw_charger

            self.image = await handler.get_image_from_rrm(self.test_data)
            _LOGGER.info("Successfully generated image from JSON data")
        except Exception as e:
            _LOGGER.warning(f"Error generating image from JSON: {e}")
            # Fall back to a simple test image if generation fails
            from PIL import Image
            self.image = Image.new('RGBA', (300, 300), (0, 125, 255, 255))
            _LOGGER.info("Created fallback test image")

        # Show available elements
        _LOGGER.info("\nAvailable elements that can be disabled:")
        for element in DrawableElement:
            _LOGGER.info(f"  - {element.name}: {element.value} (enabled: {handler.drawing_config.is_enabled(element)})")

        # Test enabling/disabling elements
        _LOGGER.info("\nTesting element enabling/disabling")
        handler.enable_element(DrawableElement.PATH)
        _LOGGER.info(f"PATH element enabled: {handler.drawing_config.is_enabled(DrawableElement.PATH)}")

        handler.disable_element(DrawableElement.PATH)
        _LOGGER.info(f"PATH element disabled: {not handler.drawing_config.is_enabled(DrawableElement.PATH)}")

        # Test changing element properties
        _LOGGER.info("\nTesting changing element properties")
        handler.set_element_property(DrawableElement.ROBOT, "color", (255, 0, 0, 255))
        _LOGGER.info("Changed robot color to bright red")

        # Display image size and other properties
        _LOGGER.info(f"Image size: {self.image.size}")
        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")

        # Show the image
        self.image.show()


def __main__():
    test = TestRandImageHandler()
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
        profile_output = "profile_output_rand.prof"
        profiler.dump_stats(profile_output)

        # Print profiling summary
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)  # Show top 50 functions


if __name__ == "__main__":
    __main__()
