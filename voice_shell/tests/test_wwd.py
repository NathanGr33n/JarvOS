from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.engines.wwd import WakeWordDetector, WakeWordDetectorError


class TestWakeWordDetector:
    """Tests for the WakeWordDetector multi-backend wrapper."""

    def test_init_porcupine_backend(self):
        """Verify initialization with Porcupine backend when model_path is provided."""
        import pvporcupine

        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.ppn"),
            keyword="Hey Nova",
            sensitivity=0.7,
        )
        assert detector._backend_name == "porcupine"
        assert detector.keyword == "Hey Nova"
        pvporcupine.create.assert_called_once_with(
            keyword_paths=["/models/hey_nova.ppn"],
            sensitivities=[0.7],
        )

    def test_init_openwakeword_backend_explicit(self):
        """Verify explicit OpenWakeWord backend selection."""
        import openwakeword

        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.onnx"),
            backend="openwakeword",
            keyword="Hey Nova",
        )
        assert detector._backend_name == "openwakeword"
        openwakeword.Model.assert_called_once_with(
            wakeword_models=["/models/hey_nova.onnx"]
        )

    def test_init_hotkey_fallback_when_no_model(self):
        """Verify hotkey fallback when no model_path is provided."""
        detector = WakeWordDetector(
            backend=None,
            keyword="Hey Nova",
        )
        assert detector._backend_name == "hotkey"

    def test_init_hotkey_explicit(self):
        """Verify explicit hotkey backend selection."""
        detector = WakeWordDetector(
            backend="hotkey",
            keyword="Hey Nova",
        )
        assert detector._backend_name == "hotkey"

    def test_porcupine_process_chunk_detected(self):
        """Verify that process_chunk returns True when Porcupine detects the keyword."""
        import pvporcupine

        pvporcupine.create.return_value.process.return_value = 0

        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.ppn"),
            backend="porcupine",
        )
        # Reset the mock to prepare for a detection
        detector._backend.process.return_value = 0

        # Frame length is 512 samples = 1024 bytes at 16-bit
        frame = b"\x00" * 1024
        assert detector.process_chunk(frame) is True
        detector._backend.process.assert_called_once()

    def test_porcupine_process_chunk_not_detected(self):
        """Verify that process_chunk returns False when Porcupine does not detect."""
        import pvporcupine

        pvporcupine.create.return_value.process.return_value = -1

        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.ppn"),
            backend="porcupine",
        )
        frame = b"\x00" * 1024
        assert detector.process_chunk(frame) is False

    def test_porcupine_process_chunk_too_small(self):
        """Verify that a too-small audio chunk returns False without calling process."""
        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.ppn"),
            backend="porcupine",
        )
        detector._backend.process.reset_mock()
        frame = b"\x00" * 100  # too small
        assert detector.process_chunk(frame) is False
        detector._backend.process.assert_not_called()

    def test_openwakeword_process_chunk_detected(self):
        """Verify that OpenWakeWord returns True when the score exceeds 0.5."""
        import openwakeword

        openwakeword.Model.return_value.predict.return_value = {"model": 0.75}

        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.onnx"),
            backend="openwakeword",
        )
        frame = b"\x00" * 1024
        assert detector.process_chunk(frame) is True

    def test_openwakeword_process_chunk_not_detected(self):
        """Verify that OpenWakeWord returns False when the score is below 0.5."""
        import openwakeword

        openwakeword.Model.return_value.predict.return_value = {"model": 0.25}

        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.onnx"),
            backend="openwakeword",
        )
        frame = b"\x00" * 1024
        assert detector.process_chunk(frame) is False

    def test_hotkey_process_chunk(self):
        """Verify that the hotkey backend returns True only when the flag is set."""
        detector = WakeWordDetector(
            backend="hotkey",
            keyword="Hey Nova",
        )
        assert detector.process_chunk(b"\x00") is False
        detector._hotkey_flag = True
        assert detector.process_chunk(b"\x00") is True
        assert detector._hotkey_flag is False  # flag is reset after read

    def test_start_hotkey(self):
        """Verify that start() starts the keyboard listener if not alive."""
        detector = WakeWordDetector(
            backend="hotkey",
            keyword="Hey Nova",
        )
        # Listener is already created and started once in __init__
        detector.start()
        # Called twice total (once in __init__, once here)
        assert detector._hotkey_listener.start.call_count == 2

    def test_stop_hotkey(self):
        """Verify that stop() stops and clears the keyboard listener."""
        detector = WakeWordDetector(
            backend="hotkey",
            keyword="Hey Nova",
        )
        listener = detector._hotkey_listener
        detector.stop()
        listener.stop.assert_called_once()
        assert detector._hotkey_listener is None

    def test_stop_porcupine(self):
        """Verify that stop() releases the Porcupine handle."""
        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.ppn"),
            backend="porcupine",
        )
        backend = detector._backend
        detector.stop()
        backend.delete.assert_called_once()
        assert detector._backend is None

    def test_repr(self):
        """Verify the repr string contains key fields."""
        detector = WakeWordDetector(
            model_path=Path("/models/hey_nova.ppn"),
            backend="porcupine",
            keyword="Hey Nova",
        )
        r = repr(detector)
        assert "WakeWordDetector" in r
        assert "porcupine" in r
        assert "Hey Nova" in r
