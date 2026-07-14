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
_mock_sd_module.PortAudioError = Exception
_mock_sd_module.CallbackFlags = MagicMock()

sys.modules["sounddevice"] = _mock_sd_module
_mock_sd_module._mock_stream = _mock_stream  # expose for test access


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock side_effects and call counts between tests."""
    _mock_stream.reset_mock()
    _mock_vad.reset_mock()
    _mock_vad.is_speech.return_value = False
    _mock_sd_module.RawInputStream.reset_mock()
    _mock_sd_module.RawInputStream.return_value = _mock_stream
    _mock_sd_module.RawInputStream.side_effect = None
    yield
