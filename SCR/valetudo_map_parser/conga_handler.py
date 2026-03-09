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

    async def async_get_image_from_json(
        self, m_json: JsonType | None
    ) -> Image.Image | None:
        """Scale Conga JSON to pixel_size=CONGA_SCALE, then render normally."""
        if m_json is None:
            return None
        scaled_json = self._scale_conga_json(m_json, CONGA_SCALE)
        self.json_data = await HyperMapData.async_from_valetudo_json(scaled_json)
        return await super().async_get_image_from_json(m_json=scaled_json)
