import datetime
import logging


class TextHUD:
    """Minimal text-based HUD for state and event visibility."""

    def __init__(self, enabled: bool = True, show_timestamps: bool = True):
        self.enabled = enabled
        self.show_timestamps = show_timestamps
        self._logger = logging.getLogger("voice_shell.hud")

    def state(self, from_state: str, to_state: str) -> None:
        self._emit("state", f"{from_state} -> {to_state}")

    def transcript(self, text: str) -> None:
        self._emit("transcript", text)

    def response(self, text: str) -> None:
        self._emit("response", text)

    def action_result(self, text: str) -> None:
        self._emit("action", text)

    def error(self, text: str) -> None:
        self._emit("error", text)

    def _emit(self, label: str, message: str) -> None:
        if not self.enabled:
            return
        if self.show_timestamps:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            self._logger.info("[HUD][%s][%s] %s", stamp, label.upper(), message)
            return
        self._logger.info("[HUD][%s] %s", label.upper(), message)
