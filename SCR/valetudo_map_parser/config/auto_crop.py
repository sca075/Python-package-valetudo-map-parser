"""Auto Crop Class for trimming and zooming images.
Version: 2024.10.0"""

from __future__ import annotations

import logging

import numpy as np
from numpy import rot90

from .types import Color, NumpyArray, TrimCropData, TrimsData
from .utils import BaseHandler


_LOGGER = logging.getLogger(__name__)


class TrimError(Exception):
    """Exception raised for errors in the trim process."""

    def __init__(self, message, image):
        super().__init__(message)
        self.image = image


class AutoCrop:
    """Auto Crop Class for trimming and zooming images."""

    def __init__(self, handler: BaseHandler):
        self.auto_crop = None  # auto crop data to be calculate once.
        self.crop_area = None
        self.handler = handler
        trim_data = self.handler.shared.trims.to_dict()  # trims data
        self.trim_up = trim_data.get("trim_up", 0)  # trim up
        self.trim_down = trim_data.get("trim_down", 0)  # trim down
        self.trim_left = trim_data.get("trim_left", 0)  # trim left
        self.trim_right = trim_data.get("trim_right", 0)  # trim right
        self.offset_top = self.handler.shared.offset_top  # offset top
        self.offset_bottom = self.handler.shared.offset_down  # offset bottom
        self.offset_left = self.handler.shared.offset_left  # offset left
        self.offset_right = self.handler.shared.offset_right  # offset right

    @staticmethod
    def validate_crop_dimensions(shared):
        """Ensure width and height are valid before processing cropping."""
        if shared.image_ref_width <= 0 or shared.image_ref_height <= 0:
            _LOGGER.warning(
                "Auto-crop failed: Invalid dimensions (width=%s, height=%s). Using original image.",
                shared.image_ref_width,
                shared.image_ref_height,
            )
            return False
        return True

    def check_trim(
        self, trimmed_height, trimmed_width, margin_size, image_array, file_name, rotate
    ):
        """Check if the trim is okay."""
        if trimmed_height <= margin_size or trimmed_width <= margin_size:
            self.crop_area = [0, 0, image_array.shape[1], image_array.shape[0]]
            self.handler.img_size = (image_array.shape[1], image_array.shape[0])
            raise TrimError(
                f"{file_name}: Trimming failed at rotation {rotate}.", image_array
            )

    def _calculate_trimmed_dimensions(self):
        """Calculate and update the dimensions after trimming."""
        trimmed_width = max(
            1,  # Ensure at least 1px
            (self.trim_right - self.offset_right) - (self.trim_left + self.offset_left),
        )
        trimmed_height = max(
            1,  # Ensure at least 1px
            (self.trim_down - self.offset_bottom) - (self.trim_up + self.offset_top),
        )

        # Ensure shared reference dimensions are updated
        if hasattr(self.handler.shared, "image_ref_height") and hasattr(
            self.handler.shared, "image_ref_width"
        ):
            self.handler.shared.image_ref_height = trimmed_height
            self.handler.shared.image_ref_width = trimmed_width
        else:
            _LOGGER.warning(
                "Shared attributes for image dimensions are not initialized."
            )

        return trimmed_width, trimmed_height

    async def _async_auto_crop_data(self, tdata: TrimsData):  # , tdata=None
        """Load the auto crop data from the Camera config."""
        _LOGGER.debug("Auto Crop data: %s, %s", str(tdata), str(self.auto_crop))
        if not self.auto_crop:
            trims_data = TrimCropData.from_dict(dict(tdata.to_dict())).to_list()
            (
                self.trim_left,
                self.trim_up,
                self.trim_right,
                self.trim_down,
            ) = trims_data
            _LOGGER.debug("Auto Crop trims data: %s", trims_data)
            if trims_data != [0, 0, 0, 0]:
                self._calculate_trimmed_dimensions()
            else:
                trims_data = None
            return trims_data
        return None

    def auto_crop_offset(self):
        """Calculate the offset for the auto crop."""
        if self.auto_crop:
            self.auto_crop[0] += self.offset_left
            self.auto_crop[1] += self.offset_top
            self.auto_crop[2] -= self.offset_right
            self.auto_crop[3] -= self.offset_bottom

    async def _init_auto_crop(self):
        """Initialize the auto crop data."""
        _LOGGER.debug("Auto Crop Init data: %s", str(self.auto_crop))
        _LOGGER.debug(
            "Auto Crop Init trims data: %r", self.handler.shared.trims.to_dict()
        )
        if not self.auto_crop:  # and self.handler.shared.vacuum_state == "docked":
            self.auto_crop = await self._async_auto_crop_data(self.handler.shared.trims)
            if self.auto_crop:
                self.auto_crop_offset()
        else:
            self.handler.max_frames = 5

        # Fallback: Ensure auto_crop is valid
        if not self.auto_crop or any(v < 0 for v in self.auto_crop):
            _LOGGER.debug("Auto-crop data unavailable. Scanning full image.")
            self.auto_crop = None

        return self.auto_crop

    async def async_image_margins(
        self, image_array: NumpyArray, detect_colour: Color
    ) -> tuple[int, int, int, int]:
        """Crop the image based on the auto crop area."""
        nonzero_coords = np.column_stack(np.where(image_array != list(detect_colour)))
        # Calculate the trim box based on the first and last occurrences
        min_y, min_x, _ = NumpyArray.min(nonzero_coords, axis=0)
        max_y, max_x, _ = NumpyArray.max(nonzero_coords, axis=0)
        del nonzero_coords
        _LOGGER.debug(
            "%s: Found trims max and min values (y,x) (%s, %s) (%s, %s)...",
            self.handler.file_name,
            int(max_y),
            int(max_x),
            int(min_y),
            int(min_x),
        )
        return min_y, min_x, max_x, max_y

    async def async_check_if_zoom_is_on(
        self,
        image_array: NumpyArray,
        margin_size: int = 100,
        zoom: bool = False,
        rand256: bool = False,
    ) -> NumpyArray:
        """Check if the image needs to be zoomed."""

        if (
            zoom
            and self.handler.shared.vacuum_state == "cleaning"
            and self.handler.shared.image_auto_zoom
        ):
            _LOGGER.debug(
                "%s: Zooming the image on room %s.",
                self.handler.file_name,
                self.handler.robot_in_room["room"],
            )

            if rand256:
                trim_left = (
                    round(self.handler.robot_in_room["right"] / 10) - margin_size
                )
                trim_right = (
                    round(self.handler.robot_in_room["left"] / 10) + margin_size
                )
                trim_up = round(self.handler.robot_in_room["down"] / 10) - margin_size
                trim_down = round(self.handler.robot_in_room["up"] / 10) + margin_size
            else:
                trim_left = self.handler.robot_in_room["left"] - margin_size
                trim_right = self.handler.robot_in_room["right"] + margin_size
                trim_up = self.handler.robot_in_room["up"] - margin_size
                trim_down = self.handler.robot_in_room["down"] + margin_size

            # Ensure valid trim values
            trim_left, trim_right = sorted([trim_left, trim_right])
            trim_up, trim_down = sorted([trim_up, trim_down])

            # Prevent zero-sized images
            if trim_right - trim_left < 1 or trim_down - trim_up < 1:
                _LOGGER.warning(
                    "Zooming resulted in an invalid crop area. Using full image."
                )
                return image_array  # Return original image

            trimmed = image_array[trim_up:trim_down, trim_left:trim_right]

        else:
            trimmed = image_array[
                self.auto_crop[1] : self.auto_crop[3],
                self.auto_crop[0] : self.auto_crop[2],
            ]

        return trimmed

    async def async_rotate_the_image(
        self, trimmed: NumpyArray, rotate: int
    ) -> NumpyArray:
        """Rotate the image and return the new array."""
        if rotate == 90:
            rotated = rot90(trimmed)
            self.crop_area = [
                self.trim_left,
                self.trim_up,
                self.trim_right,
                self.trim_down,
            ]
        elif rotate == 180:
            rotated = rot90(trimmed, 2)
            self.crop_area = self.auto_crop
        elif rotate == 270:
            rotated = rot90(trimmed, 3)
            self.crop_area = [
                self.trim_left,
                self.trim_up,
                self.trim_right,
                self.trim_down,
            ]
        else:
            rotated = trimmed
            self.crop_area = self.auto_crop
        return rotated

    async def async_auto_trim_and_zoom_image(
        self,
        image_array: NumpyArray,
        detect_colour: Color = (93, 109, 126, 255),
        margin_size: int = 0,
        rotate: int = 0,
        zoom: bool = False,
        rand256: bool = False,
    ):
        """
        Automatically crops and trims a numpy array and returns the processed image.
        """
        try:
            self.auto_crop = await self._init_auto_crop()
            if (self.auto_crop is None) or (self.auto_crop == [0, 0, 0, 0]):
                _LOGGER.debug("%s: Calculating auto trim box", self.handler.file_name)
                # Find the coordinates of the first occurrence of a non-background color
                min_y, min_x, max_x, max_y = await self.async_image_margins(
                    image_array, detect_colour
                )
                # Calculate and store the trims coordinates with margins
                self.trim_left = int(min_x) - margin_size
                self.trim_up = int(min_y) - margin_size
                self.trim_right = int(max_x) + margin_size
                self.trim_down = int(max_y) + margin_size
                del min_y, min_x, max_x, max_y

                # Calculate the dimensions after trimming using min/max values
                trimmed_width, trimmed_height = self._calculate_trimmed_dimensions()

                # Test if the trims are okay or not
                try:
                    self.check_trim(
                        trimmed_height,
                        trimmed_width,
                        margin_size,
                        image_array,
                        self.handler.file_name,
                        rotate,
                    )
                except TrimError as e:
                    return e.image

                # Store Crop area of the original image_array we will use from the next frame.
                self.auto_crop = TrimCropData(
                    self.trim_left,
                    self.trim_up,
                    self.trim_right,
                    self.trim_down,
                ).to_list()
                # Update the trims data in the shared instance
                self.handler.shared.trims = TrimsData.from_dict(
                    {
                        "trim_left": self.trim_left,
                        "trim_up": self.trim_up,
                        "trim_right": self.trim_right,
                        "trim_down": self.trim_down,
                    }
                )
                self.auto_crop_offset()
            # If it is needed to zoom the image.
            trimmed = await self.async_check_if_zoom_is_on(
                image_array, margin_size, zoom, rand256
            )
            del image_array  # Free memory.
            # Rotate the cropped image based on the given angle
            rotated = await self.async_rotate_the_image(trimmed, rotate)
            del trimmed  # Free memory.
            _LOGGER.debug(
                "%s: Auto Trim Box data: %s", self.handler.file_name, self.crop_area
            )
            self.handler.crop_img_size = [rotated.shape[1], rotated.shape[0]]
            _LOGGER.debug(
                "%s: Auto Trimmed image size: %s",
                self.handler.file_name,
                self.handler.crop_img_size,
            )

        except RuntimeError as e:
            _LOGGER.warning(
                "%s: Error %s during auto trim and zoom.",
                self.handler.file_name,
                e,
                exc_info=True,
            )
            return None
        return rotated
