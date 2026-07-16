import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock webrtcvad so tests can run without the native extension being installed.
_mock_vad_module = MagicMock()
_mock_vad = MagicMock()
_mock_vad.is_speech.return_value = False
_mock_vad_module.Vad.return_value = _mock_vad
sys.modules["webrtcvad"] = _mock_vad_module

# Mock sounddevice so tests can run without PortAudio being installed.
_mock_sd_module = MagicMock()

_mock_stream = MagicMock()
_mock_stream.start = MagicMock()
_mock_stream.stop = MagicMock()
_mock_stream.close = MagicMock()

_mock_sd_module.RawInputStream = MagicMock(return_value=_mock_stream)
_mock_sd_module.RawOutputStream = MagicMock(return_value=_mock_stream)
_mock_sd_module.PortAudioError = Exception
_mock_sd_module.CallbackFlags = MagicMock()

sys.modules["sounddevice"] = _mock_sd_module
_mock_sd_module._mock_stream = _mock_stream  # expose for test access

# Mock pynput.keyboard for WakeWordDetector hotkey backend tests.
_mock_pynput = MagicMock()
_mock_key = MagicMock()
_mock_key.ctrl = MagicMock()
_mock_key.shift = MagicMock()
_mock_key.space = MagicMock()
_mock_key.ctrl_l = MagicMock()
_mock_key.ctrl_r = MagicMock()
_mock_key.shift_l = MagicMock()
_mock_key.shift_r = MagicMock()
_mock_pynput.keyboard.Key = _mock_key

_mock_listener = MagicMock()
_mock_listener.is_alive.return_value = False
_mock_pynput.keyboard.Listener = MagicMock(return_value=_mock_listener)

sys.modules["pynput"] = _mock_pynput
sys.modules["pynput.keyboard"] = _mock_pynput.keyboard

# Mock pvporcupine for WakeWordDetector tests.
_mock_porcupine = MagicMock()
_mock_porcupine_handle = MagicMock()
_mock_porcupine_handle.frame_length = 512
_mock_porcupine_handle.process.return_value = 0
_mock_porcupine_handle.delete = MagicMock()
_mock_porcupine.create.return_value = _mock_porcupine_handle
sys.modules["pvporcupine"] = _mock_porcupine

# Mock openwakeword for WakeWordDetector tests.
_mock_oww = MagicMock()
_mock_oww_model = MagicMock()
_mock_oww_model.predict.return_value = {"model": 0.0}
_mock_oww.Model.return_value = _mock_oww_model
sys.modules["openwakeword"] = _mock_oww


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock side_effects and call counts between tests."""
    _mock_stream.reset_mock()
    _mock_vad.reset_mock()
    _mock_vad.is_speech.return_value = False
    _mock_sd_module.RawInputStream.reset_mock()
    _mock_sd_module.RawInputStream.return_value = _mock_stream
    _mock_sd_module.RawInputStream.side_effect = None
    _mock_sd_module.RawOutputStream.reset_mock()
    _mock_sd_module.RawOutputStream.return_value = _mock_stream
    _mock_sd_module.RawOutputStream.side_effect = None
    _mock_porcupine_handle.reset_mock()
    _mock_porcupine_handle.process.return_value = 0
    _mock_porcupine_handle.frame_length = 512
    _mock_oww_model.reset_mock()
    _mock_oww_model.predict.return_value = {"model": 0.0}
    _mock_listener.reset_mock()
    _mock_listener.is_alive.return_value = False
    yield
