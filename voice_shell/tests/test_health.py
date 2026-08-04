import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_shell.src.config import Config, HealthConfig
from voice_shell.src.health import EngineHealthGate, HealthReport, build_default_health_gate
from voice_shell.src.engines.tts import TTSClient


class TestHealthReport:
    def test_ready_when_required_healthy(self):
        report = HealthReport(engines=[
            __import__('voice_shell.src.health', fromlist=['EngineHealth']).EngineHealth('stt', True, True),
            __import__('voice_shell.src.health', fromlist=['EngineHealth']).EngineHealth('llm', True, True),
            __import__('voice_shell.src.health', fromlist=['EngineHealth']).EngineHealth('tts', False, False),
        ])
        assert report.ready is True

    def test_not_ready_when_required_down(self):
        from voice_shell.src.health import EngineHealth
        report = HealthReport(engines=[
            EngineHealth('stt', True, True),
            EngineHealth('llm', False, True),
        ])
        assert report.ready is False
        assert [e.name for e in report.unhealthy_required()] == ['llm']
        assert 'llm=down/required' in report.summary()

    def test_to_dict(self):
        from voice_shell.src.health import EngineHealth
        report = HealthReport(engines=[EngineHealth('stt', True, True, detail='')])
        payload = report.to_dict()
        assert payload['ready'] is True
        assert payload['engines'][0]['name'] == 'stt'
        assert payload['engines'][0]['status'] == 'ok'


class TestEngineHealthGate:
    @pytest.mark.asyncio
    async def test_check_once(self):
        gate = EngineHealthGate(
            checks={
                'stt': AsyncMock(return_value=True),
                'llm': AsyncMock(return_value=False),
            },
            required=['stt', 'llm'],
        )
        report = await gate.check_once()
        assert report.ready is False
        assert report.engines[0].healthy is True
        assert report.engines[1].healthy is False

    @pytest.mark.asyncio
    async def test_wait_until_ready_succeeds(self):
        calls = {'n': 0}

        async def flaky():
            calls['n'] += 1
            return calls['n'] >= 2

        gate = EngineHealthGate(
            checks={'stt': flaky, 'llm': AsyncMock(return_value=True)},
            required=['stt', 'llm'],
            timeout=2.0,
            poll_interval=0.05,
        )
        report = await gate.wait_until_ready()
        assert report.ready is True
        assert calls['n'] >= 2

    @pytest.mark.asyncio
    async def test_wait_until_ready_timeout(self):
        gate = EngineHealthGate(
            checks={'stt': AsyncMock(return_value=False)},
            required=['stt'],
            timeout=0.15,
            poll_interval=0.05,
        )
        report = await gate.wait_until_ready()
        assert report.ready is False


class TestBuildDefaultHealthGate:
    @pytest.mark.asyncio
    async def test_uses_client_health_checks(self):
        stt = MagicMock()
        stt.health_check = AsyncMock(return_value=True)
        llm = MagicMock()
        llm.health_check = AsyncMock(return_value=True)
        tts = MagicMock()
        tts.health_check = MagicMock(return_value=True)
        gate = build_default_health_gate(stt, llm, tts, timeout=0.1, poll_interval=0.05)
        report = await gate.wait_until_ready()
        assert report.ready is True
        stt.health_check.assert_awaited()
        llm.health_check.assert_awaited()

    def test_tts_client_health_check_missing_paths(self, tmp_path: Path):
        client = TTSClient(model_path=tmp_path / 'missing.onnx', binary_path=tmp_path / 'no-piper')
        assert client.health_check() is False

    def test_tts_client_health_check_ok(self, tmp_path: Path):
        model = tmp_path / 'voice.onnx'
        model.write_text('x')
        binary = tmp_path / 'piper'
        binary.write_text('#!/bin/sh\n')
        binary.chmod(0o755)
        client = TTSClient(model_path=model, binary_path=binary)
        assert client.health_check() is True


class TestHealthConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.health.enabled is True
        assert cfg.health.block_listening is True
        assert cfg.health.fail_fast is False

    def test_from_yaml(self, tmp_path: Path):
        path = tmp_path / 'c.yaml'
        path.write_text(
            'health:\n  enabled: false\n  require_tts: false\n  startup_timeout: 5\n  fail_fast: true\n'
        )
        cfg = Config.from_yaml(path)
        assert cfg.health.enabled is False
        assert cfg.health.require_tts is False
        assert cfg.health.startup_timeout == 5
        assert cfg.health.fail_fast is True
