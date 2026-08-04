"""GTK floating voice bar HUD for Wayland/X11 sessions."""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_STYLES = {
    "IDLE": ("#94a3b8", "#0f172a"),
    "LISTENING": ("#22c55e", "#052e16"),
    "THINKING": ("#f59e0b", "#451a03"),
    "SPEAKING": ("#38bdf8", "#0c4a6e"),
    "ERROR": ("#f87171", "#450a0a"),
}


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class FloatingHUD:
    """Always-on-top floating voice bar.

    Uses GTK4. When gtk4-layer-shell is installed and a Wayland session is
    active, the window is placed as a top-layer overlay. Otherwise it falls
    back to an undecorated always-on-top application window (works on X11 and
    many Wayland compositors).

    All public event methods are thread-safe.
    """

    def __init__(
        self,
        enabled: bool = True,
        width: int = 480,
        height: int = 140,
        anchor: str = "top-center",
        margin: int = 24,
        opacity: float = 0.92,
    ):
        self.enabled = enabled
        self.width = max(280, width)
        self.height = max(100, height)
        self.anchor = anchor
        self.margin = max(0, margin)
        self.opacity = min(1.0, max(0.4, opacity))

        self._ready = threading.Event()
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._app = None
        self._window = None
        self._state_label = None
        self._transcript_label = None
        self._response_label = None
        self._action_label = None
        self._css_provider = None
        self._current_state = "IDLE"
        self._GLib = None
        self._layer_shell = False

        if not enabled:
            return
        self._start_ui_thread()

    # ------------------------------------------------------------------
    # Public HUD API
    # ------------------------------------------------------------------

    def state(self, from_state: str, to_state: str) -> None:
        self._current_state = to_state or from_state or "IDLE"
        self._dispatch(self._apply_state, self._current_state)

    def transcript(self, text: str) -> None:
        self._dispatch(self._set_label, "transcript", _truncate(text))

    def response(self, text: str) -> None:
        self._dispatch(self._set_label, "response", _truncate(text))

    def action_result(self, text: str) -> None:
        self._dispatch(self._set_label, "action", _truncate(text, 120))

    def error(self, text: str) -> None:
        self._current_state = "ERROR"
        self._dispatch(self._apply_state, "ERROR")
        self._dispatch(self._set_label, "action", _truncate(f"Error: {text}", 120))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._GLib is not None and self._app is not None:
            self._GLib.idle_add(self._quit_app)
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # GTK lifecycle
    # ------------------------------------------------------------------

    def _start_ui_thread(self) -> None:
        self._thread = threading.Thread(target=self._run_gtk, name="jarvos-floating-hud", daemon=True)
        self._thread.start()
        # Do not block forever if display is unavailable.
        self._ready.wait(timeout=3.0)

    def _run_gtk(self) -> None:
        try:
            import gi

            gi.require_version("Gtk", "4.0")
            gi.require_version("Gdk", "4.0")
            from gi.repository import Gdk, GLib, Gtk
        except Exception as exc:
            logger.warning("Floating HUD cannot import GTK: %s", exc)
            self._ready.set()
            return

        self._GLib = GLib
        self._Gtk = Gtk
        self._Gdk = Gdk

        try:
            self._app = Gtk.Application(application_id="dev.jarvos.VoiceHUD")
            self._app.connect("activate", self._on_activate)
            # Non-unique so multiple test runs / shells can coexist.
            try:
                flags = Gtk.ApplicationFlags.NON_UNIQUE
                self._app.set_flags(flags)
            except Exception:
                pass
            self._app.run(None)
        except Exception as exc:
            logger.warning("Floating HUD GTK loop failed: %s", exc)
            self._ready.set()

    def _on_activate(self, app) -> None:
        Gtk = self._Gtk
        Gdk = self._Gdk

        window = Gtk.ApplicationWindow(application=app)
        window.set_title("JarvOS Voice")
        window.set_default_size(self.width, self.height)
        window.set_resizable(False)
        try:
            window.set_decorated(False)
        except Exception:
            pass
        try:
            window.set_opacity(self.opacity)
        except Exception:
            pass

        self._layer_shell = self._try_init_layer_shell(window)
        if not self._layer_shell:
            # X11 / generic floating fallback.
            try:
                window.set_hide_on_close(True)
            except Exception:
                pass

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        root.set_margin_top(12)
        root.set_margin_bottom(12)
        root.set_margin_start(14)
        root.set_margin_end(14)
        root.add_css_class("hud-root")

        self._state_label = Gtk.Label(label="IDLE", xalign=0.0)
        self._state_label.add_css_class("hud-state")
        self._transcript_label = Gtk.Label(label="Waiting for wake word…", xalign=0.0)
        self._transcript_label.set_wrap(True)
        self._transcript_label.add_css_class("hud-transcript")
        self._response_label = Gtk.Label(label="", xalign=0.0)
        self._response_label.set_wrap(True)
        self._response_label.add_css_class("hud-response")
        self._action_label = Gtk.Label(label="", xalign=0.0)
        self._action_label.set_wrap(True)
        self._action_label.add_css_class("hud-action")

        root.append(self._state_label)
        root.append(self._transcript_label)
        root.append(self._response_label)
        root.append(self._action_label)
        window.set_child(root)

        self._install_css(window)
        self._apply_state(self._current_state)

        window.present()
        if not self._layer_shell:
            GLib = self._GLib
            GLib.idle_add(self._position_fallback_window, window)

        self._window = window
        self._ready.set()

    def _try_init_layer_shell(self, window) -> bool:
        """Attempt gtk4-layer-shell top overlay placement."""
        try:
            import gi

            # Common typelib names across distro packages.
            for ns, ver in (("Gtk4LayerShell", "1.0"), ("GtkLayerShell", "0.1")):
                try:
                    gi.require_version(ns, ver)
                    break
                except ValueError:
                    continue
            else:
                return False

            try:
                from gi.repository import Gtk4LayerShell as LayerShell
            except ImportError:
                from gi.repository import GtkLayerShell as LayerShell

            if not hasattr(LayerShell, "is_supported") or not LayerShell.is_supported():
                # Some bindings omit is_supported; still try init.
                pass

            LayerShell.init_for_window(window)
            LayerShell.set_layer(window, LayerShell.Layer.TOP)
            LayerShell.set_namespace(window, "jarvos-voice-hud")
            LayerShell.set_exclusive_zone(window, 0)

            anchor = (self.anchor or "top-center").lower().replace("_", "-")
            edges = {
                "top": False,
                "bottom": False,
                "left": False,
                "right": False,
            }
            if "bottom" in anchor:
                edges["bottom"] = True
            else:
                edges["top"] = True
            if "left" in anchor:
                edges["left"] = True
            elif "right" in anchor:
                edges["right"] = True
            # center: top/bottom only

            for edge_name, enabled in edges.items():
                edge = getattr(LayerShell.Edge, edge_name.upper())
                LayerShell.set_anchor(window, edge, enabled)
                if enabled:
                    LayerShell.set_margin(window, edge, self.margin)

            # Keep a stable width on top/bottom centered bars.
            try:
                LayerShell.set_anchor(window, LayerShell.Edge.LEFT, "left" in anchor or "center" in anchor)
                LayerShell.set_anchor(window, LayerShell.Edge.RIGHT, "right" in anchor or "center" in anchor)
                if "center" in anchor and "left" not in anchor and "right" not in anchor:
                    # For true center, only top/bottom anchors + fixed size.
                    LayerShell.set_anchor(window, LayerShell.Edge.LEFT, False)
                    LayerShell.set_anchor(window, LayerShell.Edge.RIGHT, False)
            except Exception:
                pass

            logger.info("Floating HUD using gtk layer-shell overlay")
            return True
        except Exception as exc:
            logger.debug("Layer-shell unavailable: %s", exc)
            return False

    def _position_fallback_window(self, window) -> bool:
        """Best-effort top-center placement for non-layer-shell sessions."""
        try:
            display = window.get_display()
            monitor = display.get_monitors().get_item(0) if hasattr(display, "get_monitors") else None
            if monitor is not None:
                geom = monitor.get_geometry()
                x = geom.x + max(0, (geom.width - self.width) // 2)
                if "bottom" in (self.anchor or ""):
                    y = geom.y + geom.height - self.height - self.margin
                else:
                    y = geom.y + self.margin
                # GTK4 removed set_position; use native surface when possible.
                surface = window.get_surface()
                if surface is not None and hasattr(surface, "set_startup_id"):
                    pass
                # X11-only: GdkX11 helpers may exist.
                try:
                    from gi.repository import GdkX11  # type: ignore

                    if isinstance(surface, GdkX11.X11Surface):
                        import ctypes
                        # Soft fallback: keep default compositor placement.
                except Exception:
                    pass
                logger.debug("Floating HUD fallback geometry target x=%s y=%s", x, y)
        except Exception as exc:
            logger.debug("Fallback positioning skipped: %s", exc)
        # Keep window above others if API exists.
        try:
            window.set_keep_above(True)
        except Exception:
            pass
        return False

    def _install_css(self, window) -> None:
        Gtk = self._Gtk
        Gdk = self._Gdk
        css = """
        .hud-root {
            background-color: rgba(15, 23, 42, 0.94);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        .hud-state {
            font-family: Cantarell, sans-serif;
            font-weight: 700;
            font-size: 13px;
            letter-spacing: 1px;
            color: #e2e8f0;
        }
        .hud-transcript {
            font-family: Cantarell, sans-serif;
            font-size: 14px;
            color: #f8fafc;
        }
        .hud-response {
            font-family: Cantarell, sans-serif;
            font-size: 13px;
            color: #cbd5e1;
        }
        .hud-action {
            font-family: Cantarell, sans-serif;
            font-size: 12px;
            color: #93c5fd;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css.encode("utf-8"))
        display = window.get_display()
        Gtk.StyleContext.add_provider_for_display(
            display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self._css_provider = provider

    def _apply_state(self, state_name: str) -> None:
        if self._state_label is None:
            return
        accent, _bg = _STATE_STYLES.get(state_name.upper(), _STATE_STYLES["IDLE"])
        self._state_label.set_text(state_name.upper())
        # Inline style for accent chip color.
        try:
            self._state_label.set_markup(
                f'<span foreground="{accent}"><b>● {state_name.upper()}</b></span>'
            )
        except Exception:
            self._state_label.set_text(f"● {state_name.upper()}")

    def _set_label(self, kind: str, text: str) -> None:
        label = {
            "transcript": self._transcript_label,
            "response": self._response_label,
            "action": self._action_label,
        }.get(kind)
        if label is not None:
            label.set_text(text)

    def _quit_app(self) -> bool:
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
        if self._app is not None:
            try:
                self._app.quit()
            except Exception:
                pass
        return False

    def _dispatch(self, func, *args) -> None:
        if not self.enabled or self._closed:
            return
        if self._GLib is None:
            return
        self._GLib.idle_add(func, *args)
