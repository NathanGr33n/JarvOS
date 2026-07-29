import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_shell.src.audio.capture import AudioCapture, AudioCaptureError


class TestAudioCapture:
    """Tests for the AudioCapture module."""

    def test_init_default_params(self):
        """Verify default initialization parameters."""
        capture = AudioCapture()
        assert capture.sample_rate == 16000
        assert capture.channels == 1
        assert capture.blocksize == 512
        assert capture.dtype == "int16"
        assert capture.device is None

    def test_init_custom_params(self):
        """Verify custom initialization parameters."""
        capture = AudioCapture(
            sample_rate=44100,
            channels=2,
            blocksize=1024,
            dtype="float32",
            device=3,
        )
        assert capture.sample_rate == 44100
        assert capture.channels == 2
        assert capture.blocksize == 1024
        assert capture.dtype == "float32"
        assert capture.device == 3

    def test_start_creates_and_starts_stream(self):
        """Verify that start() creates a RawInputStream and starts it."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()

        sd.RawInputStream.assert_called_once()
        kwargs = sd.RawInputStream.call_args.kwargs
        assert kwargs["samplerate"] == 16000
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "int16"
        assert kwargs["blocksize"] == 512
        assert kwargs["device"] is None
        assert callable(kwargs["callback"])

        assert capture._running is True
        sd._mock_stream.start.assert_called_once()

    def test_start_raises_on_portaudio_error(self):
        """Verify that PortAudioError is translated to AudioCaptureError."""
        import sounddevice as sd

        sd.RawInputStream.side_effect = sd.PortAudioError("No input device")

        capture = AudioCapture()
        with pytest.raises(AudioCaptureError, match="Failed to open audio input device"):
            capture.start()

    def test_stop_closes_stream(self):
        """Verify that stop() closes and cleans up the stream."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()
        capture.stop()

        sd._mock_stream.stop.assert_called_once()
        sd._mock_stream.close.assert_called_once()
        assert capture._stream is None
        assert capture._running is False

    def test_double_start_is_noop(self):
        """Verify that calling start() twice is safe."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()
        capture.start()

        # Should only create one stream
        assert sd.RawInputStream.call_count == 1

    def test_callback_puts_in_queue(self):
        """Verify that the audio callback puts chunks in the asyncio queue."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        fake_data = np.zeros((512, 1), dtype=np.int16)
        callback(fake_data, 512, {}, None)

        # Drain the queue in the event loop
        chunk = asyncio.run(capture.get_chunk())
        assert chunk == fake_data.tobytes()

    def test_callback_accumulates_in_listening_mode(self):
        """Verify that audio is accumulated when in listening mode."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()
        capture.start_listening()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        fake_data = np.ones((512, 1), dtype=np.int16) * 42
        callback(fake_data, 512, {}, None)

        audio = capture.stop_listening()
        assert audio == fake_data.tobytes()

    def test_callback_idle_buffer_ring(self):
        """Verify that the idle ring buffer overwrites old data."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        large_data = np.ones((512, 1), dtype=np.int16)

        # Fill the buffer with more than 30 seconds worth of data
        for _ in range(2000):
            callback(large_data, 512, {}, None)

        idle_buffer = capture.get_idle_buffer()
        max_expected = 16000 * 30 * 2  # 30 seconds of 16-bit mono
        assert len(idle_buffer) <= max_expected

    def test_start_listening_clears_buffer(self):
        """Verify that start_listening resets the listening buffer."""
        capture = AudioCapture()
        capture.start()
        capture.start_listening()
        capture._listening_buffer.extend(b"old data")

        capture.start_listening()
        assert capture._listening_buffer == bytearray()

    def test_stop_listening_returns_bytes(self):
        """Verify that stop_listening returns accumulated audio and exits listening mode."""
        capture = AudioCapture()
        capture.start()
        capture.start_listening()
        capture._listening_buffer.extend(b"test audio")

        audio = capture.stop_listening()
        assert audio == b"test audio"
        assert capture._is_listening is False

    def test_capture_until_silence_with_vad(self):
        """Verify that capture_until_silence stops when VAD returns True."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        fake_data = np.zeros((512, 1), dtype=np.int16)

        async def _simulate_audio():
            """Simulate audio chunks arriving while capture_until_silence runs."""
            # Wait briefly for capture_until_silence to start listening
            await asyncio.sleep(0.01)
            for _ in range(5):
                callback(fake_data, 512, {}, None)
                await asyncio.sleep(0.001)

        mock_vad = MagicMock()
        mock_vad.process.side_effect = [False, False, True]  # end-of-speech on 3rd chunk
        mock_vad.reset = MagicMock()

        async def _test():
            asyncio.create_task(_simulate_audio())
            audio = await capture.capture_until_silence(mock_vad, max_duration=10.0)
            assert audio == fake_data.tobytes() * 3
            mock_vad.reset.assert_called_once()

        asyncio.run(_test())

    def test_capture_until_silence_timeout(self):
        """Verify that capture_until_silence stops at max_duration."""
        import sounddevice as sd

        capture = AudioCapture()
        capture.start()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        fake_data = np.zeros((512, 1), dtype=np.int16)

        async def _simulate_audio():
            """Feed a slow trickle of audio to avoid blocking the queue forever."""
            await asyncio.sleep(0.01)
            for _ in range(3):
                callback(fake_data, 512, {}, None)
                await asyncio.sleep(0.03)

        mock_vad = MagicMock()
        mock_vad.process.return_value = False
        mock_vad.reset = MagicMock()

        async def _test():
            asyncio.create_task(_simulate_audio())
            start = time.monotonic()
            audio = await capture.capture_until_silence(mock_vad, max_duration=0.05)
            elapsed = time.monotonic() - start

            assert elapsed >= 0.05
            assert elapsed < 0.3
            mock_vad.reset.assert_called_once()

        asyncio.run(_test())

    def test_capture_to_wav(self, tmp_path: Path):
        """Verify that capture_to_wav writes a valid WAV file."""
        import sounddevice as sd
        import wave
        import io

        capture = AudioCapture()
        capture.start()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        fake_data = np.zeros((512, 1), dtype=np.int16)

        async def _simulate_audio():
            await asyncio.sleep(0.01)
            for _ in range(3):
                callback(fake_data, 512, {}, None)
                await asyncio.sleep(0.001)

        mock_vad = MagicMock()
        mock_vad.process.side_effect = [False, False, True]
        mock_vad.reset = MagicMock()

        output = tmp_path / "capture.wav"

        async def _test():
            asyncio.create_task(_simulate_audio())
            result = await capture.capture_to_wav(mock_vad, output, max_duration=10.0)
            assert result == output
            assert output.exists()

            with wave.open(str(output), "rb") as wav:
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
                assert wav.getframerate() == 16000

        asyncio.run(_test())

    def test_context_manager(self):
        """Verify that AudioCapture works as a context manager."""
        import sounddevice as sd

        with AudioCapture() as capture:
            assert capture._running is True
            sd._mock_stream.start.assert_called()

        sd._mock_stream.stop.assert_called()
        sd._mock_stream.close.assert_called()
        assert capture._running is False

    def test_queue_drops_on_full(self):
        """Verify that the callback drops frames when the queue is full."""
        import sounddevice as sd

        capture = AudioCapture()
        capture._queue = asyncio.Queue(maxsize=1)
        capture.start()

        callback = sd.RawInputStream.call_args.kwargs["callback"]
        fake_data = np.zeros((512, 1), dtype=np.int16)

        # Fill the queue
        callback(fake_data, 512, {}, None)

        # Next frame should be dropped without error
        callback(fake_data, 512, {}, None)  # should not raise
