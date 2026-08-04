"""HUD construction helpers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .composite import CompositeHUD
from .text import TextHUD

if TYPE_CHECKING:
    from ..config import HUDConfig

logger = logging.getLogger(__name__)


def create_hud(config: "HUDConfig"):
    """Create a text, floating, or composite HUD from config.

    Floating construction never raises: if GTK/display is unavailable the
    factory falls back to text-only logging and records a warning.
    """
    mode = (getattr(config, "mode", "both") or "both").strip().lower()
    enabled = bool(getattr(config, "enabled", True))
    show_timestamps = bool(getattr(config, "show_timestamps", True))

    text_hud = TextHUD(enabled=enabled and mode in {"text", "both", "all"}, show_timestamps=show_timestamps)
    backends = []
    if mode in {"text", "both", "all"}:
        backends.append(text_hud)

    headless = os.environ.get("JARVOS_HUD_HEADLESS", "").strip() in {"1", "true", "yes"}
    if enabled and (not headless) and mode in {"floating", "wayland", "both", "all"}:
        try:
            from .floating import FloatingHUD

            backends.append(
                FloatingHUD(
                    enabled=True,
                    width=int(getattr(config, "width", 480) or 480),
                    height=int(getattr(config, "height", 140) or 140),
                    anchor=str(getattr(config, "anchor", "top-center") or "top-center"),
                    margin=int(getattr(config, "margin", 24) or 24),
                    opacity=float(getattr(config, "opacity", 0.92) or 0.92),
                )
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Floating HUD unavailable (%s); using text HUD only.", exc)
            if not backends:
                backends.append(TextHUD(enabled=enabled, show_timestamps=show_timestamps))

    if not backends:
        backends.append(TextHUD(enabled=False, show_timestamps=show_timestamps))

    if len(backends) == 1:
        return backends[0]
    return CompositeHUD(backends)
