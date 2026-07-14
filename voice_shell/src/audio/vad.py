import collections
from typing import Optional

try:
    import webrtcvad
except ImportError as exc:
    raise ImportError(
        "webrtcvad is required for voice activity detection. "
        "Install it with: pip install webrtcvad"
    ) from exc


class VoiceActivityDetector:
    """Wrapper around webrtcvad with buffering and end-of-speech detection.

    Handles the mismatch between arbitrary audio chunk sizes and the fixed
    frame sizes required by webrtcvad (10, 20, or 30 ms at 16 kHz).
    """

    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        end_of_speech_window_ms: int = 1000,
        end_of_speech_threshold: float = 0.5,
    ):
        """Initialize the VAD.

        Args:
            aggressiveness: VAD aggressiveness mode (0-3). Higher is more aggressive
                (filters out more noise). Default 3.
            sample_rate: Audio sample rate in Hz. Must be 8000, 16000, 32000, or 48000.
            frame_duration_ms: Frame duration for VAD processing. Must be 10, 20, or 30.
            end_of_speech_window_ms: Sliding window size for end-of-speech detection.
            end_of_speech_threshold: Ratio of non-speech frames in the window to trigger
                end-of-speech (0.0-1.0).
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_bytes = int(
            sample_rate * (frame_duration_ms / 1000.0) * 2
        )  # 16-bit = 2 bytes per sample

        self._window_size = max(1, int(end_of_speech_window_ms / frame_duration_ms))
        self._threshold = end_of_speech_threshold

        self._buffer: bytes = b""
        self._speech_history: collections.deque = collections.deque(maxlen=self._window_size)
        self._has_speech: bool = False

    def reset(self) -> None:
        """Reset internal buffers and speech state."""
        self._buffer = b""
        self._speech_history.clear()
        self._has_speech = False

    def process(self, audio_bytes: bytes) -> bool:
        """Process audio bytes and detect end-of-speech.

        Internally buffers audio until a complete VAD frame is available,
        then classifies it as speech or non-speech.

        Args:
            audio_bytes: Raw PCM audio chunk (16-bit, mono).

        Returns:
            True if end-of-speech has been detected, False otherwise.
        """
        self._buffer += audio_bytes

        while len(self._buffer) >= self.frame_bytes:
            frame = self._buffer[: self.frame_bytes]
            self._buffer = self._buffer[self.frame_bytes :]

            is_speech = self.vad.is_speech(frame, self.sample_rate)
            self._speech_history.append(is_speech)

            if not self._has_speech and is_speech:
                self._has_speech = True

            if self._has_speech and len(self._speech_history) >= self._window_size:
                speech_ratio = sum(self._speech_history) / len(self._speech_history)
                if speech_ratio < self._threshold:
                    return True

        return False

    @property
    def has_detected_speech(self) -> bool:
        """Return whether any speech has been detected since the last reset."""
        return self._has_speech

    def is_speech(self, frame: bytes) -> bool:
        """Classify a single, complete VAD frame as speech or not.

        Args:
            frame: Audio frame of exactly ``frame_bytes`` length.

        Returns:
            True if the frame contains speech.
        """
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"Frame size must be exactly {self.frame_bytes} bytes, got {len(frame)}"
            )
        return self.vad.is_speech(frame, self.sample_rate)
