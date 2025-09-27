"""
Version: 0.1.10
Status text of the vacuum cleaners.
Class to handle the status text of the vacuum cleaners.
"""

from __future__ import annotations

from ..types import LOGGER, PilPNG
from .translations import translations

LOGGER.propagate = True


class StatusText:
    """
    Status text of the vacuum cleaners.
    """

    def __init__(self, camera_shared):
        self._shared = camera_shared
        self.file_name = self._shared.file_name

    @staticmethod
    async def get_vacuum_status_translation(
        language: str = "en",
    ) -> dict[str, str] | None:
        """
        Get the vacuum status translation.
        @param language: Language code, default 'en'.
        @return: Mapping for the given language or None.
        """
        return translations.get((language or "en").lower())

    async def translate_vacuum_status(self) -> str:
        """Return the translated status with EN fallback and safe default."""
        status = self._shared.vacuum_state or "unknown"
        language = (self._shared.user_language or "en").lower()
        translation = await self.get_vacuum_status_translation(language)
        if not translation:
            translation = translations.get("en", {})
        return translation.get(status, str(status).capitalize())

    async def get_status_text(self, text_img: PilPNG) -> tuple[list[str], int]:
        """
        Compose the image status text.
        :param text_img: Image to draw the text on.
        :return status_text, text_size: List of the status text and the text size.
        """
        status_text = ["If you read me, something really went wrong.."]  # default text
        text_size_coverage = 1.5  # resize factor for the text
        text_size = self._shared.vacuum_status_size  # default text size
        charge_level = "\u03de"  # unicode Koppa symbol
        charging = "\u2211"  # unicode Charging symbol
        vacuum_state = await self.translate_vacuum_status()
        if self._shared.show_vacuum_state:
            status_text = [f"{self.file_name}: {vacuum_state}"]
            language = (self._shared.user_language or "en").lower()
            lang_map = translations.get(language) or translations.get("en", {})
            if not self._shared.vacuum_connection:
                mqtt_disc = lang_map.get(
                    "mqtt_disconnected",
                    translations.get("en", {}).get(
                        "mqtt_disconnected", "Disconnected from MQTT?"
                    ),
                )
                status_text = [f"{self.file_name}: {mqtt_disc}"]
            else:
                if self._shared.current_room:
                    in_room = self._shared.current_room.get("in_room")
                    if in_room:
                        status_text.append(f" ({in_room})")
                if self._shared.vacuum_state == "docked":
                    if self._shared.vacuum_bat_charged():
                        status_text.append(" \u00b7 ")
                        status_text.append(f"{charging}{charge_level} ")
                        status_text.append(f"{self._shared.vacuum_battery}%")
                    else:
                        status_text.append(" \u00b7 ")
                        status_text.append(f"{charge_level} ")
                        ready_txt = lang_map.get(
                            "ready",
                            translations.get("en", {}).get("ready", "Ready."),
                        )
                        status_text.append(ready_txt)
                else:
                    status_text.append(" \u00b7 ")
                    status_text.append(f"{charge_level}")
                    status_text.append(f" {self._shared.vacuum_battery}%")
                if text_size >= 50 and getattr(text_img, "width", None):
                    text_pixels = max(1, sum(len(text) for text in status_text))
                    text_size = int(
                        (text_size_coverage * text_img.width) // text_pixels
                    )
        return status_text, text_size
