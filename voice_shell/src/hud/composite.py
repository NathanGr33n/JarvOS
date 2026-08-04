"""Fan-out HUD that forwards events to multiple backends."""

from __future__ import annotations

from typing import Iterable, List

from .base import HUD


class CompositeHUD:
    """Dispatch the same HUD event to one or more backends."""

    def __init__(self, backends: Iterable[HUD]):
        self.backends: List[HUD] = list(backends)

    def state(self, from_state: str, to_state: str) -> None:
        for backend in self.backends:
            backend.state(from_state, to_state)

    def transcript(self, text: str) -> None:
        for backend in self.backends:
            backend.transcript(text)

    def response(self, text: str) -> None:
        for backend in self.backends:
            backend.response(text)

    def action_result(self, text: str) -> None:
        for backend in self.backends:
            backend.action_result(text)

    def error(self, text: str) -> None:
        for backend in self.backends:
            backend.error(text)

    def close(self) -> None:
        for backend in self.backends:
            closer = getattr(backend, "close", None)
            if callable(closer):
                closer()
