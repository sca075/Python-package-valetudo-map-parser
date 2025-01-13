from __future__ import annotations

import json
import os

from SCR.valetudo_map_parser.config.shared import CameraSharedManager
import asyncio
from SCR.valetudo_map_parser.hypfer_handler import (
    HypferMapImageHandler,
)  # Adjust the import based on your project structure
import logging

_LOGGER = logging.getLogger(__name__)


class TestImageHandler:
    def __init__(self):
        self.test_data = None
        self.image = None

    def setUp(self):
        # Load the test.json file

        test_file_path = os.getcwd() + "/test.json"
        print(test_file_path)
        with open(test_file_path, "r") as file:
            self.test_data = json.load(file)

    async def test_image_handler(self):
        shared_data = CameraSharedManager("test_vacuum", {})
        shared = shared_data.get_instance()
        handler = HypferMapImageHandler(shared)
        self.image = await handler.async_get_image_from_json(self.test_data)
        print(f"Calibration_data: {handler.get_calibration_data()}")
        print(f"Image size: {self.image.size}")
        print(f"Room Proprietes: {await handler.async_get_rooms_attributes()}")
        self.image.show()


def __main__():
    test = TestImageHandler()
    test.setUp()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test.test_image_handler())
    loop.close()


if __name__ == "__main__":
    __main__()
