"""
Conga Image Handler Class.
Extends HypferMapImageHandler for Conga vacuums.
Conga maps have pixelSize=1 and canvas size 800×800.  Before rendering we
deep-copy the JSON and multiply pixelSize, canvas size, and all entity
coordinates by CONGA_SCALE so that every map pixel becomes a CONGA_SCALE×CONGA_SCALE
block, producing solid filled rooms.  Entity positions stay consistent with
the scaled map coordinate space, so no post-upscale is required.
Version: 0.1.4
"""

from __future__ import annotations

import copy

from PIL import Image

from .config.shared import CameraShared
from .config.types import JsonType
from .conga_draw import CONGA_SCALE, CongaImageDraw
from .hypfer_handler import HypferMapImageHandler
from .map_data import HyperMapData


class CongaMapImageHandler(HypferMapImageHandler):
    """Image Handler for Conga vacuums.

    Scales pixelSize, canvas size, and all entity coordinates by CONGA_SCALE
    before handing the JSON to the standard Hypfer render pipeline.
    """

    def __init__(self, shared_data: CameraShared) -> None:
        """Initialise the Conga image handler."""
        super().__init__(shared_data)
        self.imd = CongaImageDraw(self)

    @staticmethod
    def _scale_conga_json(m_json: JsonType, scale: int) -> JsonType:
        """Return a deep copy of m_json with coordinates scaled by *scale*.

        Patches:
        - pixelSize → scale
        - size.x / size.y → original × scale
        - every entity's points list → each value × scale
        """
        scaled = copy.deepcopy(m_json)
        scaled["pixelSize"] = scale
        size = scaled.get("size", {})
        scaled["size"] = {
            "x": int(size.get("x", 800)) * scale,
            "y": int(size.get("y", 800)) * scale,
        }
        for entity in scaled.get("entities", []):
            pts = entity.get("points")
            if isinstance(pts, list):
                entity["points"] = [v * scale for v in pts]
        return scaled

    async def async_get_conga_from_json(
        self, m_json: JsonType | None
    ) -> Image.Image | None:
        """Scale Conga JSON to pixel_size=CONGA_SCALE, then render normally."""
        if m_json is None:
            return None
        scaled_json = self._scale_conga_json(m_json, CONGA_SCALE)
        self.json_data = await HyperMapData.async_from_valetudo_json(scaled_json)
        return await super().async_get_image_from_json(m_json=scaled_json)

    def get_vacuum_points(self, rotation_angle: int) -> list[dict[str, int]]:
        """Calculate the calibration points from crop_area.

        The vacuum points must be returned in the order that matches map_points:
        [top-left, top-right, bottom-right, bottom-left] of the DISPLAYED image.

        crop_area is in vacuum coordinates and doesn't rotate, so we need to
        map the correct crop_area corners to the displayed image corners based
        on rotation.
        """
        if not self.crop_area:
            return [
                {"x": 0, "y": 0},
                {"x": 0, "y": 0},
                {"x": 0, "y": 0},
                {"x": 0, "y": 0},
            ]

        vacuum_points = [
            {
                "x": self.crop_area[0] // CONGA_SCALE + self.offset_x,
                "y": self.crop_area[1] // CONGA_SCALE + self.offset_y,
            },
            {
                "x": self.crop_area[2] // CONGA_SCALE - self.offset_x,
                "y": self.crop_area[1] // CONGA_SCALE + self.offset_y,
            },
            {
                "x": self.crop_area[2] // CONGA_SCALE - self.offset_x,
                "y": self.crop_area[3] // CONGA_SCALE - self.offset_y,
            },
            {
                "x": self.crop_area[0] // CONGA_SCALE + self.offset_x,
                "y": self.crop_area[3] // CONGA_SCALE - self.offset_y,
            },
        ]

        match rotation_angle:
            case 90:
                vacuum_points = [
                    vacuum_points[1],
                    vacuum_points[2],
                    vacuum_points[3],
                    vacuum_points[0],
                ]
            case 180:
                vacuum_points = [
                    vacuum_points[2],
                    vacuum_points[3],
                    vacuum_points[0],
                    vacuum_points[1],
                ]
            case 270:
                vacuum_points = [
                    vacuum_points[3],
                    vacuum_points[0],
                    vacuum_points[1],
                    vacuum_points[2],
                ]
            case _:
                pass

        return vacuum_points

