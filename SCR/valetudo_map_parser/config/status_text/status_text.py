"""
Version: 0.1.12
Status text of the vacuum cleaners.
Class to handle the status text of the vacuum cleaners.
"""

from __future__ import annotations

from typing import Callable

from ...const import charge_level, charging, dot, text_size_coverage
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
        self._language = (self._shared.user_language or "en").lower()
        self._lang_map = translations.get(self._language) or translations.get("en", {})
        self._compose_functions: list[Callable[[list[str]], list[str]]] = [
            self._current_room,
            self._dock_state,
            self._docked_charged,
            self._docked_ready,
            self._active,
            self._mqtt_disconnected,
        ]  # static ordered sequence of compose functions

    @staticmethod
    async def _get_vacuum_status_translation(
        language: str = "en",
    ) -> dict[str, str] | None:
        """
        Get the vacuum status translation.
        @param language: Language code, default 'en'.
        @return: Mapping for the given language or None.
        """
        return translations.get((language or "en").lower())

    async def _translate_vacuum_status(self) -> str:
        """Return the translated status with EN fallback and safe default."""
        status = self._shared.vacuum_state or "unknown"
        language = (self._shared.user_language or "en").lower()
        translation = await self._get_vacuum_status_translation(language)
        if not translation:
            translation = translations.get("en", {})
        return translation.get(status, str(status).capitalize())

    def _mqtt_disconnected(self, current_state: list[str]) -> list[str]:
        """Return the translated MQTT disconnected status."""
        if not self._shared.vacuum_connection:
            mqtt_disc = (self._lang_map or {}).get(
                "mqtt_disconnected",
                translations.get("en", {}).get(
                    "mqtt_disconnected", "Disconnected from MQTT?"
                ),
            )
            return [f"{self.file_name}: {mqtt_disc}"]
        return current_state

    def _dock_state(self, current_state: list[str]) -> list[str]:
        """Return the dock state if active and not idle."""
        if (
            self._shared.dock_state is not None
            and self._shared.dock_state != "idle"
            and self._shared.vacuum_state == "docked"
        ):
            current_state.append(f" {self._shared.dock_state}")
        return current_state

    def _docked_charged(self, current_state: list[str]) -> list[str]:
        """Return the translated docked and charging status."""
        if self._shared.vacuum_state == "docked" and self._shared.vacuum_bat_charged():
            current_state.append(dot)
            current_state.append(f"{charging}{charge_level} ")
            current_state.append(f"{self._shared.vacuum_battery}%")
        return current_state

    def _docked_ready(self, current_state: list[str]) -> list[str]:
        """Return the translated docked and ready status."""
        if (
            self._shared.vacuum_state == "docked"
            and not self._shared.vacuum_bat_charged()
        ):
            current_state.append(dot)
            current_state.append(f"{charge_level} ")
            ready_txt = (self._lang_map or {}).get(
                "ready",
                translations.get("en", {}).get("ready", "Ready."),
            )
            current_state.append(ready_txt)
        return current_state

    def _current_room(self, current_state: list[str]) -> list[str]:
        """Return the current room information."""
        if self._shared.current_room:
            in_room = self._shared.current_room.get("in_room")
            if in_room and in_room != "Room 31":
                current_state.append(f" ({in_room})")
        return current_state

    def _active(self, current_state: list[str]) -> list[str]:
        """Return the translated active status."""
        if self._shared.vacuum_state != "docked":
            current_state.append(dot)
            current_state.append(f"{charge_level}")
            current_state.append(f" {self._shared.vacuum_battery}%")
        return current_state

    async def get_status_text(self, text_img: PilPNG) -> tuple[list[str], int]:
        """
        Compose the image status text.
        :param text_img: Image to draw the text on.
        :return status_text, text_size: List of the status text and the text size.
        """
        text_size = self._shared.vacuum_status_size  # default text size
        vacuum_state = await self._translate_vacuum_status()
        status_text = [f"{self.file_name}: {vacuum_state}"]
        # Compose Status Text with available data.
        for func in self._compose_functions:
            status_text = func(status_text)
        if text_size >= 50 and getattr(text_img, "width", None):
            text_pixels = max(1, sum(len(text) for text in status_text))
            text_size = int((text_size_coverage * text_img.width) // text_pixels)
        return status_text, text_size
