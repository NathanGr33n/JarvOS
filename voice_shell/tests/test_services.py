from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.config import Config, ServicesConfig
from voice_shell.src.services import ServiceDefinition, ServiceManager, ServiceStatus


class TestServiceManager:
    def test_from_config_empty_by_default(self):
        mgr = ServiceManager.from_config(Config())
        assert mgr.services == {}

    def test_from_config_registers_commands(self):
        cfg = Config()
        cfg.services = ServicesConfig(
            manage_whisper=True,
            manage_llama=True,
            whisper_command=["whisper-server", "--port", "8081"],
            llama_command=["llama-server", "--port", "8082"],
            whisper_health_url="http://127.0.0.1:8081/",
            llama_health_url="http://127.0.0.1:8082/health",
        )
        mgr = ServiceManager.from_config(cfg)
        assert set(mgr.services) == {"whisper", "llama"}
        assert mgr.services["whisper"].command[0] == "whisper-server"

    def test_register_requires_command(self):
        mgr = ServiceManager()
        with pytest.raises(ValueError):
            mgr.register(ServiceDefinition(name="x", command=[]))

    def test_start_stop_without_health_url(self):
        mgr = ServiceManager(
            [ServiceDefinition(name="sleep", command=["sleep", "30"], ready_timeout=1.0)]
        )
        status = mgr.start("sleep", wait_healthy=True)
        assert status == ServiceStatus.RUNNING
        assert mgr.is_running("sleep")
        assert mgr.stop("sleep") == ServiceStatus.STOPPED
        assert not mgr.is_running("sleep")

    def test_start_missing_binary_fails(self):
        mgr = ServiceManager(
            [ServiceDefinition(name="missing", command=["definitely-not-a-real-bin-xyz"])]
        )
        status = mgr.start("missing", wait_healthy=False)
        assert status == ServiceStatus.FAILED

    def test_check_health_uses_url(self):
        mgr = ServiceManager(
            [
                ServiceDefinition(
                    name="api",
                    command=["sleep", "30"],
                    health_url="http://127.0.0.1:9/health",
                    ready_timeout=0.2,
                    poll_interval=0.05,
                )
            ]
        )
        with patch("voice_shell.src.services.manager.urllib.request.urlopen") as urlopen:
            response = MagicMock()
            response.status = 200
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            urlopen.return_value = response
            status = mgr.start("api", wait_healthy=True)
            assert status == ServiceStatus.RUNNING
            assert mgr.check_health("api") is True
        mgr.stop("api")

    def test_start_all_and_stop_all(self):
        mgr = ServiceManager(
            [
                ServiceDefinition(name="a", command=["sleep", "30"]),
                ServiceDefinition(name="b", command=["sleep", "30"]),
            ]
        )
        statuses = mgr.start_all(wait_healthy=True)
        assert statuses["a"] == ServiceStatus.RUNNING
        assert statuses["b"] == ServiceStatus.RUNNING
        stopped = mgr.stop_all()
        assert stopped["a"] == ServiceStatus.STOPPED
        assert stopped["b"] == ServiceStatus.STOPPED


class TestServicesConfig:
    def test_services_defaults(self):
        cfg = Config()
        assert cfg.services.manage_whisper is False
        assert cfg.services.autostart is False
        assert cfg.services.whisper_command == []

    def test_services_from_yaml(self, tmp_path: Path):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "\n".join(
                [
                    "services:",
                    "  manage_whisper: true",
                    "  autostart: true",
                    "  ready_timeout: 12",
                    "  whisper_command: [\"whisper-server\"]",
                ]
            )
        )
        cfg = Config.from_yaml(path)
        assert cfg.services.manage_whisper is True
        assert cfg.services.autostart is True
        assert cfg.services.ready_timeout == 12
        assert cfg.services.whisper_command == ["whisper-server"]
