import pytest
import struct
import wave
import io
from pathlib import Path

from voice_shell.src.utils.wav_writer import write_wav_from_buffer


class TestWavWriter:
    """Tests for the wav_writer utility."""

    def test_write_wav_to_bytes(self):
        """Verify that PCM bytes are correctly wrapped in a WAV container."""
        audio = b"\x01\x02" * 1000  # 2000 bytes = 1000 samples @ 16-bit mono
        wav_bytes = write_wav_from_buffer(audio, sample_rate=16000, channels=1, sample_width=2)

        assert len(wav_bytes) > len(audio)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() == 1000

    def test_write_wav_to_file(self, tmp_path: Path):
        """Verify that WAV bytes are written to disk correctly."""
        audio = b"\x03\x04" * 500
        output = tmp_path / "test.wav"
        result = write_wav_from_buffer(audio, output_path=output)

        assert result == b""
        assert output.exists()

        with wave.open(str(output), "rb") as wav:
            assert wav.getnframes() == 500
            assert wav.getsampwidth() == 2

    def test_write_wav_different_sample_rate(self):
        """Verify that non-default sample rates are preserved in the header."""
        audio = b"\x00\x00" * 200
        wav_bytes = write_wav_from_buffer(audio, sample_rate=44100, channels=2, sample_width=2)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getframerate() == 44100
            assert wav.getnchannels() == 2

    def test_write_wav_empty_audio(self):
        """Verify that empty audio produces a valid, empty WAV file."""
        wav_bytes = write_wav_from_buffer(b"", sample_rate=16000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getnframes() == 0

    def test_write_wav_creates_parent_directories(self, tmp_path: Path):
        """Verify that missing parent directories are created automatically."""
        audio = b"\x05\x06" * 100
        output = tmp_path / "nested" / "dir" / "test.wav"
        write_wav_from_buffer(audio, output_path=output)
        assert output.exists()
