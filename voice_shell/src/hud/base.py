"""Shared HUD protocol used by text and floating implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HUD(Protocol):
    """Minimal event surface consumed by the orchestrator."""

    def state(self, from_state: str, to_state: str) -> None:
        ...

    def transcript(self, text: str) -> None:
        ...

    def response(self, text: str) -> None:
        ...

    def action_result(self, text: str) -> None:
        ...

    def error(self, text: str) -> None:
        ...
