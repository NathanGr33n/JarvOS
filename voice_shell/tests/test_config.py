import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from voice_shell.src.config import (
    AudioConfig,
    Config,
    HUDConfig,
    LLMConfig,
    STTConfig,
    TTSConfig,
    WakeWordConfig,
)


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_audio_config(self):
        """Verify default audio settings."""
        cfg = AudioConfig()
        assert cfg.sample_rate_input == 16000
        assert cfg.sample_rate_output == 22050
        assert cfg.chunk_size == 512
        assert cfg.max_recording_duration == 30.0
        assert cfg.input_device is None
        assert cfg.output_device is None

    def test_default_stt_config(self):
        """Verify default STT settings."""
        cfg = STTConfig()
        assert cfg.engine == "whisper.cpp"
        assert cfg.base_url == "http://127.0.0.1:8081"
        assert cfg.language == "en"
        assert cfg.translate is False

    def test_default_llm_config(self):
        """Verify default LLM settings."""
        cfg = LLMConfig()
        assert cfg.engine == "llama.cpp"
        assert cfg.base_url == "http://127.0.0.1:8082"
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.7
        assert cfg.system_prompt is None

    def test_default_tts_config(self):
        """Verify default TTS settings."""
        cfg = TTSConfig()
        assert cfg.engine == "piper"
        assert "en_US-lessac-medium" in cfg.model_path
        assert cfg.speaker_id == 0
        assert cfg.binary_path == "piper"

    def test_default_config(self):
        """Verify default Config creates all sub-configs."""
        cfg = Config()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.wake_word, WakeWordConfig)
        assert isinstance(cfg.stt, STTConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.tts, TTSConfig)
        assert isinstance(cfg.hud, HUDConfig)
        assert cfg.hud.enabled is True
        assert cfg.hud.show_timestamps is True


class TestConfigFromYaml:
    """Tests for loading configuration from YAML files."""

    def test_load_full_config(self, tmp_path: Path):
        """Verify that a complete YAML config overrides defaults correctly."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "audio": {
                "sample_rate_input": 44100,
                "chunk_size": 1024,
                "input_device": 3,
            },
            "stt": {
                "base_url": "http://localhost:9000",
                "language": "es",
            },
            "llm": {
                "max_tokens": 256,
                "temperature": 0.5,
                "system_prompt": "You are a test bot.",
            },
        }
        config_path.write_text(yaml.dump(config_data))

        cfg = Config.from_yaml(config_path)

        assert cfg.audio.sample_rate_input == 44100
        assert cfg.audio.chunk_size == 1024
        assert cfg.audio.input_device == 3
        # Unchanged defaults remain intact
        assert cfg.audio.sample_rate_output == 22050

        assert cfg.stt.base_url == "http://localhost:9000"
        assert cfg.stt.language == "es"
        assert cfg.stt.translate is False  # default

        assert cfg.llm.max_tokens == 256
        assert cfg.llm.temperature == 0.5
        assert cfg.llm.system_prompt == "You are a test bot."

    def test_load_partial_config(self, tmp_path: Path):
        """Verify that missing sections retain defaults."""
        config_path = tmp_path / "config.yaml"
        config_data = {"audio": {"chunk_size": 256}}
        config_path.write_text(yaml.dump(config_data))

        cfg = Config.from_yaml(config_path)
        assert cfg.audio.chunk_size == 256
        assert cfg.audio.sample_rate_input == 16000  # default
        assert cfg.stt.base_url == "http://127.0.0.1:8081"  # default

    def test_load_unknown_keys_ignored(self, tmp_path: Path):
        """Verify that unknown keys in YAML are silently ignored."""
        config_path = tmp_path / "config.yaml"
        config_data = {"audio": {"unknown_key": 123}, "future_section": {"foo": "bar"}}
        config_path.write_text(yaml.dump(config_data))

        cfg = Config.from_yaml(config_path)
        # Should not raise and defaults should be intact
        assert cfg.audio.sample_rate_input == 16000

    def test_load_empty_file(self, tmp_path: Path):
        """Verify that an empty YAML file results in all defaults."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        cfg = Config.from_yaml(config_path)
        assert cfg.audio.sample_rate_input == 16000
        assert cfg.stt.language == "en"

    def test_load_nested_path(self, tmp_path: Path):
        """Verify loading from a nested directory path."""
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        config_path = nested / "voice.yaml"
        config_path.write_text(yaml.dump({"tts": {"speaker_id": 2}}))

        cfg = Config.from_yaml(config_path)
        assert cfg.tts.speaker_id == 2


class TestMemoryConfig:
    def test_memory_defaults(self):
        from voice_shell.src.config import Config
        cfg = Config()
        assert cfg.memory.enabled is True
        assert cfg.memory.db_path == "data/memory.db"
        assert cfg.memory.history_limit == 3

    def test_memory_from_yaml(self, tmp_path):
        from voice_shell.src.config import Config
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "memory:\n  enabled: false\n  db_path: custom.db\n  history_limit: 5\n"
        )
        cfg = Config.from_yaml(path)
        assert cfg.memory.enabled is False
        assert cfg.memory.db_path == "custom.db"
        assert cfg.memory.history_limit == 5


class TestHUDConfig:
    def test_hud_defaults_include_floating_settings(self):
        from voice_shell.src.config import Config
        cfg = Config()
        assert cfg.hud.mode == "both"
        assert cfg.hud.width == 480
        assert cfg.hud.anchor == "top-center"

    def test_hud_from_yaml(self, tmp_path):
        from voice_shell.src.config import Config
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "hud:\n  mode: floating\n  width: 320\n  anchor: bottom-right\n  opacity: 0.8\n"
        )
        cfg = Config.from_yaml(path)
        assert cfg.hud.mode == "floating"
        assert cfg.hud.width == 320
        assert cfg.hud.anchor == "bottom-right"
        assert cfg.hud.opacity == 0.8
