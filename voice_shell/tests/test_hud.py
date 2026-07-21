import logging

from voice_shell.src.hud import TextHUD


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
