import asyncio
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.audio.playback import AudioPlayback, AudioPlaybackError


class TestAudioPlayback:
    """Tests for the AudioPlayback module."""

    @staticmethod
    def _mock_loop_with_closing_create_task():
        """Build a loop mock that closes passed coroutines from create_task."""
        mock_loop = MagicMock()

        def _create_task(coro):
            coro.close()
            task = MagicMock()
            task.cancel = MagicMock()
            return task

        mock_loop.create_task.side_effect = _create_task
        return mock_loop

    def test_init_default_params(self):
        """Verify default initialization parameters."""
        playback = AudioPlayback()
        assert playback.sample_rate == 22050
        assert playback.channels == 1
        assert playback.blocksize == 2048
        assert playback.dtype == "int16"
        assert playback.device is None

    def test_init_custom_params(self):
        """Verify custom initialization parameters."""
        playback = AudioPlayback(
            sample_rate=44100,
            channels=2,
            blocksize=1024,
            dtype="float32",
            device=3,
        )
        assert playback.sample_rate == 44100
        assert playback.channels == 2
        assert playback.blocksize == 1024
        assert playback.dtype == "float32"
        assert playback.device == 3

    @patch("voice_shell.src.audio.playback.asyncio.get_event_loop")
    def test_start_creates_and_starts_stream(self, mock_get_loop):
        """Verify that start() creates a RawOutputStream and starts it."""
        import sounddevice as sd
        mock_loop = self._mock_loop_with_closing_create_task()
        mock_get_loop.return_value = mock_loop
        playback = AudioPlayback()
        playback.start()

        sd.RawOutputStream.assert_called_once()
        kwargs = sd.RawOutputStream.call_args.kwargs
        assert kwargs["samplerate"] == 22050
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "int16"
        assert kwargs["blocksize"] == 2048
        assert kwargs["device"] is None
        assert callable(kwargs["callback"])

        assert playback._running is True
        sd._mock_stream.start.assert_called_once()
        mock_loop.create_task.assert_called_once()

    def test_start_raises_on_portaudio_error(self):
        """Verify that PortAudioError is translated to AudioPlaybackError."""
        import sounddevice as sd

        sd.RawOutputStream.side_effect = sd.PortAudioError("No output device")

        playback = AudioPlayback()
        with pytest.raises(AudioPlaybackError, match="Failed to open audio output device"):
            playback.start()

    @patch("voice_shell.src.audio.playback.asyncio.get_event_loop")
    def test_stop_closes_stream(self, mock_get_loop):
        """Verify that stop() closes and cleans up the stream."""
        import sounddevice as sd
        mock_loop = self._mock_loop_with_closing_create_task()
        mock_get_loop.return_value = mock_loop
        playback = AudioPlayback()
        playback.start()
        playback.stop()

        sd._mock_stream.stop.assert_called_once()
        sd._mock_stream.close.assert_called_once()
        assert playback._stream is None
        assert playback._running is False

    @patch("voice_shell.src.audio.playback.asyncio.get_event_loop")
    def test_double_start_is_noop(self, mock_get_loop):
        """Verify that calling start() twice is safe."""
        import sounddevice as sd
        mock_loop = self._mock_loop_with_closing_create_task()
        mock_get_loop.return_value = mock_loop
        playback = AudioPlayback()
        playback.start()
        playback.start()

        assert sd.RawOutputStream.call_count == 1

    @patch("voice_shell.src.audio.playback.asyncio.get_event_loop")
    def test_queue_chunk(self, mock_get_loop):
        """Verify that queue_chunk puts audio bytes in the async queue."""
        mock_loop = self._mock_loop_with_closing_create_task()
        mock_get_loop.return_value = mock_loop
        playback = AudioPlayback()
        playback.start()

        fake_audio = b"\x01\x02\x03\x04"
        playback.queue_chunk(fake_audio)

        assert not playback._queue.empty()
        assert playback._queue.get_nowait() == fake_audio

    def test_queue_chunk_drops_on_full(self):
        """Verify that queue_chunk drops chunks when the queue is full."""
        playback = AudioPlayback()
        playback._queue = MagicMock()
        playback._queue.put_nowait.side_effect = asyncio.QueueFull

        playback.queue_chunk(b"\x01\x02")
        playback._queue.put_nowait.assert_called_once()

    @patch("voice_shell.src.audio.playback.asyncio.get_event_loop")
    def test_context_manager(self, mock_get_loop):
        """Verify that AudioPlayback works as a context manager."""
        import sounddevice as sd
        mock_loop = self._mock_loop_with_closing_create_task()
        mock_get_loop.return_value = mock_loop
        with AudioPlayback() as playback:
            assert playback._running is True
            sd._mock_stream.start.assert_called()

        sd._mock_stream.stop.assert_called()
        sd._mock_stream.close.assert_called()
        assert playback._running is False

    def test_repr(self):
        """Verify the repr string contains the class name."""
        playback = AudioPlayback()
        r = repr(playback)
        assert "AudioPlayback" in r
