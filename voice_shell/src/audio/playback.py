import asyncio
import logging
import queue
from typing import Optional

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError(
        "sounddevice is required for audio playback. "
        "Install it with: pip install sounddevice"
    ) from exc

logger = logging.getLogger(__name__)


class AudioPlaybackError(Exception):
    """Raised when audio playback fails or the device is unavailable."""


class AudioPlayback:
    """Async audio playback that consumes raw PCM chunks from a queue.

    Uses ``sounddevice.RawOutputStream`` with a callback that reads from a
    thread-safe ``queue.Queue``. An async bridge task moves chunks from an
    ``asyncio.Queue`` (fed by the orchestrator) into the thread-safe queue.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        channels: int = 1,
        blocksize: int = 2048,
        dtype: str = "int16",
        device: Optional[int] = None,
    ):
        """Initialize the audio playback.

        Args:
            sample_rate: Sampling rate in Hz (default 22050 for Piper output).
            channels: Number of audio channels (default 1 for mono).
            blocksize: Number of frames per callback (default 2048).
            dtype: NumPy data type string (default 'int16').
            device: sounddevice device index, or None for the default output device.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self.device = device

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._thread_queue: Optional[queue.Queue] = None  # initialized in start()
        self._pending = memoryview(b"")
        self._stream: Optional[sd.RawOutputStream] = None
        self._running: bool = False
        self._bridge_task: Optional[asyncio.Task] = None

    def _callback(self, outdata, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """sounddevice output callback (runs in a separate thread)."""
        if status:
            logger.debug("[AudioPlayback] status: %s", status)

        size = len(outdata)
        offset = 0
        while offset < size:
            if not self._pending:
                try:
                    self._pending = memoryview(self._thread_queue.get_nowait())
                except queue.Empty:
                    break
                if not self._pending:
                    continue

            count = min(len(self._pending), size - offset)
            outdata[offset : offset + count] = self._pending[:count]
            # Keep the tail ahead of newer chunks without copying it.
            self._pending = self._pending[count:]
            offset += count

        if offset < size:
            outdata[offset:] = b"\x00" * (size - offset)

    def start(self) -> None:
        """Start the audio output stream and the async bridge task."""
        if self._running:
            return

        self._thread_queue = queue.Queue()
        self._pending = memoryview(b"")
        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.blocksize,
                device=self.device,
                callback=self._callback,
            )
        except sd.PortAudioError as exc:
            raise AudioPlaybackError(
                f"Failed to open audio output device: {exc}"
            ) from exc

        self._stream.start()
        self._running = True
        self._bridge_task = asyncio.get_event_loop().create_task(self._bridge_loop())

    def stop(self) -> None:
        """Stop and close the audio output stream."""
        self._running = False
        if self._bridge_task is not None:
            self._bridge_task.cancel()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._pending = memoryview(b"")

    async def _bridge_loop(self) -> None:
        """Bridge from asyncio.Queue to thread-safe queue.Queue."""
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            self._thread_queue.put(chunk)

    def queue_chunk(self, audio_bytes: bytes) -> None:
        """Queue a raw PCM chunk for playback.

        Args:
            audio_bytes: Raw PCM audio data (16-bit, mono, matching sample_rate).

        Raises:
            asyncio.QueueFull: If the queue is at capacity.
        """
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.warning("Audio playback queue full; dropping chunk")

    async def wait_for_empty(self) -> None:
        """Wait until all queued audio chunks have been consumed by the playback stream."""
        while (
            self._pending
            or not self._queue.empty()
            or (self._thread_queue is not None and not self._thread_queue.empty())
        ):
            await asyncio.sleep(0.05)

    def __enter__(self) -> "AudioPlayback":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"running={self._running}"
            f")"
        )
