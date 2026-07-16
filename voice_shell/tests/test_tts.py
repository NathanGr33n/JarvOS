import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.engines.tts import TTSClient, TTSClientError


class TestTTSClient:
    """Tests for the TTSClient Piper wrapper."""

    def test_init(self):
        """Verify that initialization stores model path and speaker ID."""
        client = TTSClient(model_path=Path("/models/voice.onnx"), speaker_id=2)
        assert client.model_path == Path("/models/voice.onnx")
        assert client.speaker_id == 2
        assert client.binary_path == Path("piper")

    def test_synthesize_success(self):
        """Verify that synthesize returns audio bytes on a successful subprocess run."""
        client = TTSClient(model_path=Path("/models/voice.onnx"))

        fake_audio = b"\x01\x02\x03\x04"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=fake_audio, stderr=b"", returncode=0
            )
            result = client.synthesize("Hello world")

        assert result == fake_audio
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert kwargs["input"] == b"Hello world"
        assert kwargs["check"] is True

    def test_synthesize_empty_text(self):
        """Verify that synthesize returns empty bytes for empty/whitespace text."""
        client = TTSClient(model_path=Path("/models/voice.onnx"))
        assert client.synthesize("") == b""
        assert client.synthesize("   ") == b""

    def test_synthesize_failure_raises(self):
        """Verify that a failed subprocess run raises TTSClientError."""
        client = TTSClient(model_path=Path("/models/voice.onnx"))

        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["piper"],
                stderr=b"model not found",
            )
            with pytest.raises(TTSClientError, match="model not found"):
                client.synthesize("Hello")

    def test_synthesize_empty_output_raises(self):
        """Verify that empty stdout from Piper raises TTSClientError."""
        client = TTSClient(model_path=Path("/models/voice.onnx"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"", stderr=b"", returncode=0
            )
            with pytest.raises(TTSClientError, match="empty audio"):
                client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_stream_synthesize_yields_audio(self):
        """Verify that stream_synthesize yields audio per sentence."""
        client = TTSClient(model_path=Path("/models/voice.onnx"))

        with patch.object(client, "synthesize") as mock_synth:
            mock_synth.side_effect = [b"audio1", b"audio2", b"audio3"]

            text_queue = asyncio.Queue()
            audio_queue = await client.stream_synthesize(text_queue)

            async def _feed():
                await text_queue.put("Hello world.")
                await text_queue.put(" How are you?")
                await text_queue.put(" Goodbye!")
                await text_queue.put(None)

            asyncio.create_task(_feed())

            chunks = []
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                chunks.append(chunk)

            assert chunks == [b"audio1", b"audio2", b"audio3"]

    @pytest.mark.asyncio
    async def test_stream_synthesize_handles_partial_sentences(self):
        """Verify that tokens are buffered until a sentence-ending char is seen."""
        client = TTSClient(model_path=Path("/models/voice.onnx"))

        with patch.object(client, "synthesize") as mock_synth:
            mock_synth.return_value = b"audio"

            text_queue = asyncio.Queue()
            audio_queue = await client.stream_synthesize(text_queue)

            async def _feed():
                # Feed one character at a time
                for ch in "Hi there!":
                    await text_queue.put(ch)
                await text_queue.put(None)

            asyncio.create_task(_feed())

            chunks = []
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0] == b"audio"
            mock_synth.assert_called_once_with("Hi there!")

    def test_repr(self):
        """Verify the repr string contains key fields."""
        client = TTSClient(model_path=Path("/models/voice.onnx"), speaker_id=1)
        r = repr(client)
        assert "TTSClient" in r
        assert "voice.onnx" in r
        assert "speaker=1" in r
