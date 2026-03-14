"""
Image Draw Class for Conga Vacuums.
Extends HypferImageDraw to provide a Conga-specific drawing layer.
Pixel data is already normalised to compressedPixels by the time it reaches
this class, so all Hypfer drawing routines apply unchanged.
The canvas is rendered at CONGA_SCALE × the native Conga size so each
boundary pixel becomes a dense CONGA_SCALE×CONGA_SCALE block, producing
filled-looking rooms. Entity coordinates are pre-scaled in the handler.
Version: 0.1.2
"""

from __future__ import annotations

from .hypfer_draw import ImageDraw as HypferImageDraw


# Scale factor: Conga pixelSize=1 vs standard pixelSize=5.
# The handler scales the canvas and all entity coordinates by this factor.
CONGA_SCALE = 5


class CongaImageDraw(HypferImageDraw):
    """Drawing handler for Conga vacuums.

    Inherits all drawing logic from HypferImageDraw unchanged.
    Coordinate scaling is handled entirely in CongaMapImageHandler so
    the draw methods receive already-scaled values.
    """
