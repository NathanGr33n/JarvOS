import logging
from unittest.mock import MagicMock, patch

from voice_shell.src.config import HUDConfig
from voice_shell.src.hud import CompositeHUD, TextHUD, create_hud
from voice_shell.src.hud.floating import FloatingHUD, _truncate


class TestTextHUD:
    """Tests for the text HUD component."""

    def test_hud_enabled_emits_logs(self, caplog):
        """Verify enabled HUD writes structured log messages."""
        hud = TextHUD(enabled=True, show_timestamps=False)
        with caplog.at_level(logging.INFO, logger="voice_shell.hud"):
            hud.state("IDLE", "LISTENING")
            hud.transcript("hello")
        assert "[HUD][STATE] IDLE -> LISTENING" in caplog.text
        assert "[HUD][TRANSCRIPT] hello" in caplog.text

    def test_hud_disabled_no_logs(self, caplog):
        """Verify disabled HUD does not emit logs."""
        hud = TextHUD(enabled=False, show_timestamps=False)
        with caplog.at_level(logging.INFO, logger="voice_shell.hud"):
            hud.response("sample")
        assert caplog.text == ""


class TestCompositeHUD:
    def test_forwards_to_all_backends(self):
        a = MagicMock()
        b = MagicMock()
        hud = CompositeHUD([a, b])
        hud.state("IDLE", "LISTENING")
        hud.transcript("hi")
        hud.response("hello")
        hud.action_result("ok")
        hud.error("boom")
        a.state.assert_called_once_with("IDLE", "LISTENING")
        b.transcript.assert_called_once_with("hi")
        a.response.assert_called_once_with("hello")
        b.action_result.assert_called_once_with("ok")
        a.error.assert_called_once_with("boom")

    def test_close_calls_backend_close(self):
        a = MagicMock()
        b = MagicMock()
        del b.close
        hud = CompositeHUD([a, b])
        hud.close()
        a.close.assert_called_once()


class TestFloatingHelpers:
    def test_truncate_short(self):
        assert _truncate("hello") == "hello"

    def test_truncate_long(self):
        text = "x" * 200
        out = _truncate(text, limit=20)
        assert len(out) == 20
        assert out.endswith("…")

    def test_disabled_floating_does_not_start_thread(self):
        hud = FloatingHUD(enabled=False)
        assert hud._thread is None
        hud.state("IDLE", "LISTENING")
        hud.transcript("hi")
        hud.close()


class TestCreateHUD:
    def test_text_mode(self):
        hud = create_hud(HUDConfig(enabled=True, mode="text"))
        assert isinstance(hud, TextHUD)

    def test_both_mode_with_headless_is_text_only(self, monkeypatch):
        monkeypatch.setenv("JARVOS_HUD_HEADLESS", "1")
        hud = create_hud(HUDConfig(enabled=True, mode="both"))
        assert isinstance(hud, TextHUD)

    def test_both_mode_composes_when_floating_available(self, monkeypatch):
        monkeypatch.delenv("JARVOS_HUD_HEADLESS", raising=False)
        fake = MagicMock()
        with patch.dict("sys.modules", {"voice_shell.src.hud.floating": MagicMock(FloatingHUD=lambda **k: fake)}):
            # Ensure import path uses our fake module
            import importlib
            import voice_shell.src.hud.factory as factory
            importlib.reload(factory)
            hud = factory.create_hud(HUDConfig(enabled=True, mode="both"))
            assert isinstance(hud, CompositeHUD)
            importlib.reload(factory)

    def test_floating_mode_falls_back_to_text(self, monkeypatch):
        monkeypatch.delenv("JARVOS_HUD_HEADLESS", raising=False)
        with patch("voice_shell.src.hud.floating.FloatingHUD", side_effect=RuntimeError("no gtk")):
            hud = create_hud(HUDConfig(enabled=True, mode="floating"))
        assert isinstance(hud, TextHUD)

    def test_disabled_returns_backend(self):
        hud = create_hud(HUDConfig(enabled=False, mode="text"))
        assert isinstance(hud, TextHUD)
        assert hud.enabled is False
