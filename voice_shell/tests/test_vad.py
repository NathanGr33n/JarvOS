import pytest
from unittest.mock import MagicMock

from voice_shell.src.audio.vad import VoiceActivityDetector


class TestVoiceActivityDetector:
    """Tests for the VoiceActivityDetector wrapper."""

    def test_init_default_params(self):
        """Verify default initialization parameters."""
        vad = VoiceActivityDetector()
        assert vad.sample_rate == 16000
        assert vad.frame_duration_ms == 30
        assert vad.frame_bytes == 960  # 16000 * 0.03 * 2

    def test_init_custom_params(self):
        """Verify custom initialization parameters."""
        vad = VoiceActivityDetector(
            aggressiveness=2,
            sample_rate=8000,
            frame_duration_ms=10,
            end_of_speech_window_ms=500,
        )
        assert vad.sample_rate == 8000
        assert vad.frame_duration_ms == 10
        assert vad.frame_bytes == 160  # 8000 * 0.01 * 2
        assert vad._window_size == 50

    def test_reset_clears_state(self):
        """Verify that reset clears buffers and speech state."""
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960
        vad._window_size = 33
        vad._threshold = 0.5
        vad._buffer = b"x" * 1000
        vad._speech_history = [True, False]
        vad._has_speech = True

        vad.reset()

        assert vad._buffer == b""
        assert len(vad._speech_history) == 0
        assert vad._has_speech is False

    def test_process_no_end_of_speech_without_speech(self):
        """Verify that end-of-speech is never triggered if no speech was detected."""
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960
        vad._window_size = 33
        vad._threshold = 0.5
        vad._buffer = b""
        vad._speech_history = []
        vad._has_speech = False

        silence = b"\x00" * 960
        for _ in range(100):
            result = vad.process(silence)
            assert result is False

    def test_process_end_of_speech_after_speech(self):
        """Verify that end-of-speech is triggered after a period of silence."""
        import collections

        mock_vad = MagicMock()
        # Simulate: first 33 frames are speech, then 33 frames are silence
        call_count = [0]
        def is_speech_side_effect(frame, rate):
            call_count[0] += 1
            return call_count[0] <= 33

        mock_vad.is_speech.side_effect = is_speech_side_effect
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960
        vad._window_size = 33
        vad._threshold = 0.5
        vad._buffer = b""
        vad._speech_history = collections.deque(maxlen=33)
        vad._has_speech = False

        frame = b"\x00" * 960

        # Feed 33 speech frames
        for _ in range(33):
            assert vad.process(frame) is False

        assert vad._has_speech is True

        # Feed silence frames. The deque has maxlen=33. After the first 33
        # speech frames, the deque is all True. Each silence frame drops one
        # True and adds one False. The ratio falls below 0.5 when there are
        # 16 True and 17 False in the window (16/33 ≈ 0.485), which happens
        # on the 17th silence frame (i == 16). Once triggered, the ratio
        # stays below the threshold, so every subsequent call also returns True.
        for i in range(33):
            result = vad.process(frame)
            if i >= 16:
                assert result is True
            else:
                assert result is False

    def test_process_buffers_partial_frames(self):
        """Verify that partial frames are buffered and not processed."""
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960
        vad._window_size = 33
        vad._threshold = 0.5
        vad._buffer = b""
        vad._speech_history = []
        vad._has_speech = False

        # Feed 480 bytes (half a frame) — should not call VAD
        vad.process(b"\x00" * 480)
        assert mock_vad.is_speech.call_count == 0
        assert len(vad._buffer) == 480

        # Feed another 480 bytes — now one full frame
        vad.process(b"\x00" * 480)
        assert mock_vad.is_speech.call_count == 1

    def test_is_speech_valid_frame(self):
        """Verify that a correctly sized frame is passed to webrtcvad."""
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960

        frame = b"\x01" * 960
        result = vad.is_speech(frame)
        assert result is True
        mock_vad.is_speech.assert_called_once_with(frame, 16000)

    def test_is_speech_invalid_frame_size(self):
        """Verify that an incorrectly sized frame raises ValueError."""
        mock_vad = MagicMock()
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960

        with pytest.raises(ValueError, match="Frame size must be exactly 960 bytes"):
            vad.is_speech(b"\x01" * 100)

    def test_has_detected_speech_property(self):
        """Verify the has_detected_speech property reflects internal state."""
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.vad = mock_vad
        vad.sample_rate = 16000
        vad.frame_duration_ms = 30
        vad.frame_bytes = 960
        vad._window_size = 33
        vad._threshold = 0.5
        vad._buffer = b""
        vad._speech_history = []
        vad._has_speech = False

        assert vad.has_detected_speech is False
        vad.process(b"\x00" * 960)
        assert vad.has_detected_speech is True
