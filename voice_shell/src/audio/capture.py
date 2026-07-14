import asyncio
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError(
        "sounddevice is required for audio capture. "
        "Install it with: pip install sounddevice"
    ) from exc

from ..utils.wav_writer import write_wav_from_buffer


class AudioCaptureError(Exception):
    """Raised when audio capture fails or the device is unavailable."""


class AudioCapture:
    """Continuous audio capture with async queue-based chunk retrieval.

    Manages a sounddevice input stream in a background thread and bridges
    audio chunks to the asyncio event loop via an asyncio.Queue.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 512,
        dtype: str = "int16",
        device: Optional[int] = None,
    ):
        """Initialize the audio capture.

        Args:
            sample_rate: Sampling rate in Hz (default 16000 for wake-word + STT).
            channels: Number of audio channels (default 1 for mono).
            blocksize: Number of frames per callback (default 512, ~32 ms at 16 kHz).
            dtype: NumPy data type string (default 'int16').
            device: sounddevice device index, or None for the default input device.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self.device = device

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._stream: Optional[sd.RawInputStream] = None
        self._running: bool = False

        # Ring buffer for pre-listening audio (e.g., wake-word context).
        # Stores up to 30 seconds of raw audio frames as bytes.
        self._idle_buffer: deque = deque(
            maxlen=int(sample_rate * 30 * np.dtype(dtype).itemsize)
        )

        # Listening accumulation buffer
        self._listening_buffer: bytearray = bytearray()
        self._is_listening: bool = False

    def _callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """sounddevice input callback (runs in a separate thread)."""
        if status:
            # Log non-critical statuses but keep capturing.
            print(f"[AudioCapture] status: {status}")

        audio_bytes = indata.tobytes()

        if self._is_listening:
            self._listening_buffer.extend(audio_bytes)
        else:
            self._idle_buffer.extend(audio_bytes)

        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            pass  # Drop frames if the consumer is too slow.

    def start(self) -> None:
        """Start the audio input stream."""
        if self._running:
            return

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.blocksize,
                device=self.device,
                callback=self._callback,
            )
        except sd.PortAudioError as exc:
            raise AudioCaptureError(
                f"Failed to open audio input device: {exc}"
            ) from exc

        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def get_chunk(self) -> bytes:
        """Retrieve a single audio chunk from the async queue.

        Returns:
            Raw PCM audio bytes (size = blocksize * channels * sample_width).
        """
        return await self._queue.get()

    def start_listening(self) -> None:
        """Switch to listening mode: audio chunks are accumulated."""
        self._is_listening = True
        self._listening_buffer = bytearray()

    def stop_listening(self) -> bytes:
        """Stop listening and return the accumulated audio buffer.

        Returns:
            All audio bytes captured during the listening session.
        """
        self._is_listening = False
        return bytes(self._listening_buffer)

    async def capture_until_silence(
        self,
        vad,
        max_duration: float = 30.0,
    ) -> bytes:
        """Capture audio until the VAD detects end-of-speech or a timeout.

        This method automatically enters listening mode, feeds incoming
        audio chunks to the VAD, and returns the complete audio buffer
        when speech ends or the maximum duration is reached.

        Uses a short ``get_chunk`` timeout so that the max_duration check
        is still evaluated even when the audio queue is empty (e.g., during
        tests or after the microphone is unplugged).

        Args:
            vad: A voice activity detector with ``process(audio_bytes) -> bool``
                and ``reset()`` methods.
            max_duration: Maximum recording duration in seconds.

        Returns:
            Raw PCM audio bytes captured during the session.
        """
        self.start_listening()
        vad.reset()

        loop = asyncio.get_running_loop()
        start_time = loop.time()

        try:
            while True:
                elapsed = loop.time() - start_time
                if elapsed >= max_duration:
                    break

                try:
                    chunk = await asyncio.wait_for(self.get_chunk(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if vad.process(chunk):
                    break
        finally:
            audio = self.stop_listening()

        return audio

    async def capture_to_wav(
        self,
        vad,
        output_path: Path,
        max_duration: float = 30.0,
    ) -> Path:
        """Capture audio until silence and write it to a WAV file.

        Args:
            vad: A voice activity detector with ``process(audio_bytes) -> bool``
                and ``reset()`` methods.
            output_path: Destination path for the WAV file.
            max_duration: Maximum recording duration in seconds.

        Returns:
            The output path.
        """
        audio = await self.capture_until_silence(vad, max_duration)
        write_wav_from_buffer(
            audio,
            output_path=output_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        return output_path

    def get_idle_buffer(self) -> bytes:
        """Return the current contents of the idle ring buffer.

        Useful for including pre-wake-word audio context in a recording.

        Returns:
            Raw PCM audio bytes (up to 30 seconds).
        """
        return bytes(self._idle_buffer)

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
