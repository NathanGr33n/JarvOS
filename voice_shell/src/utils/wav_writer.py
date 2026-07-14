import wave
import io
from pathlib import Path
from typing import Optional


def write_wav_from_buffer(
    audio_bytes: bytes,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Write raw PCM audio bytes to a WAV file or return WAV bytes.

    Args:
        audio_bytes: Raw PCM audio data (little-endian signed integers).
        output_path: If provided, writes WAV file to this path. If None, returns WAV bytes.
        sample_rate: Sampling rate in Hz (default 16000).
        channels: Number of channels (default 1 for mono).
        sample_width: Bytes per sample (default 2 for 16-bit).

    Returns:
        WAV file bytes if output_path is None, otherwise empty bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    wav_bytes = buffer.getvalue()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        return b""

    return wav_bytes
