import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_shell.src.diagnostics import (
    DiagnosticReport,
    ServiceSnapshot,
    collect_diagnostic_report,
    format_report,
)
from voice_shell.src.health import EngineHealth, HealthReport
from voice_shell import main as main_mod


def _sample_report(ready: bool = True) -> DiagnosticReport:
    engines = HealthReport(
        engines=[
            EngineHealth("stt", ready, True),
            EngineHealth("llm", True, True),
            EngineHealth("tts", True, False, detail=""),
        ]
    )
    return DiagnosticReport(
        generated_at="2026-08-04T00:00:00Z",
        config_path="config.yaml",
        health_enabled=True,
        engines=engines,
        services=[ServiceSnapshot(name="whisper", status="stopped", command=["whisper-server"])],
        endpoints={"stt": "http://127.0.0.1:8081"},
        notes=["example note"],
    )


class TestDiagnosticFormatting:
    def test_text_contains_sections(self):
        text = format_report(_sample_report(ready=True), "text")
        assert "JarvOS Engine Health Status" in text
        assert "Overall:   READY" in text
        assert "[OK ] stt" in text
        assert "Supervised services:" in text
        assert "example note" in text

    def test_text_not_ready(self):
        text = format_report(_sample_report(ready=False), "text")
        assert "NOT READY" in text
        assert "[DOWN]" in text

    def test_json_roundtrip_keys(self):
        payload = json.loads(format_report(_sample_report(), "json"))
        assert payload["ready"] is True
        assert payload["engines"]["ready"] is True
        assert payload["services"][0]["name"] == "whisper"
        assert "stt" in payload["endpoints"]


class TestCollectDiagnosticReport:
    @pytest.mark.asyncio
    async def test_collect_uses_mocked_clients(self, tmp_path: Path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "\n".join(
                [
                    "stt:",
                    "  base_url: http://127.0.0.1:18081",
                    "llm:",
                    "  base_url: http://127.0.0.1:18082",
                    "health:",
                    "  enabled: true",
                    "  require_tts: false",
                    "services:",
                    "  manage_whisper: true",
                    '  whisper_command: ["whisper-server"]',
                ]
            )
        )

        stt = MagicMock()
        stt.health_check = AsyncMock(return_value=True)
        stt.close = AsyncMock()
        llm = MagicMock()
        llm.health_check = AsyncMock(return_value=False)
        llm.close = AsyncMock()
        tts = MagicMock()
        tts.health_check = MagicMock(return_value=True)

        with patch("voice_shell.src.diagnostics.STTClient", return_value=stt), \
             patch("voice_shell.src.diagnostics.LLMClient", return_value=llm), \
             patch("voice_shell.src.diagnostics.TTSClient", return_value=tts):
            report = await collect_diagnostic_report(cfg)

        assert report.config_path == str(cfg)
        assert report.ready is False
        names = {e.name: e.healthy for e in report.engines.engines}
        assert names["stt"] is True
        assert names["llm"] is False
        assert any(s.name == "whisper" for s in report.services)
        stt.close.assert_awaited()
        llm.close.assert_awaited()


class TestStatusCLI:
    def test_status_command_exit_codes(self, tmp_path: Path, capsys):
        ready_report = _sample_report(ready=True)
        down_report = _sample_report(ready=False)

        with patch("voice_shell.main.collect_diagnostic_report", new=AsyncMock(return_value=ready_report)):
            with pytest.raises(SystemExit) as exc:
                main_mod.main(["--config", str(tmp_path / "missing.yaml"), "status"])
            assert exc.value.code == 0
            out = capsys.readouterr().out
            assert "READY" in out

        with patch("voice_shell.main.collect_diagnostic_report", new=AsyncMock(return_value=down_report)):
            with pytest.raises(SystemExit) as exc:
                main_mod.main(["status", "--format", "json"])
            assert exc.value.code == 1
            out = capsys.readouterr().out
            payload = json.loads(out)
            assert payload["ready"] is False

    def test_parse_defaults_to_run(self):
        args = main_mod._parse_args([])
        assert args.command == "run"


class TestModelsCLI:
    def test_models_list(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["models", "list"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "whisper-tiny" in out

    def test_models_status(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["models", "--dir", str(tmp_path), "status"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "missing" in out

    def test_models_download_unknown_name_fails(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["models", "--dir", str(tmp_path), "download", "nonexistent"])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Error" in out
