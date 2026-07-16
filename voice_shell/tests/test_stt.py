import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from voice_shell.src.engines.stt import STTClient, STTClientError


class AsyncContextManagerMock:
    """A reusable async context manager mock that returns a response object."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class FakeResponse:
    """Fake aiohttp response for unit testing."""

    def __init__(self, status, text=None, json_data=None, raise_on_text=None):
        self.status = status
        self._text = text
        self._json = json_data
        self._raise_on_text = raise_on_text

    async def text(self):
        if self._raise_on_text:
            raise self._raise_on_text
        return self._text or ""

    async def json(self):
        if self._json is not None:
            return self._json
        raise aiohttp.ContentTypeError(
            MagicMock(), MagicMock()
        )


class TestSTTClient:
    """Tests for the STTClient whisper.cpp wrapper."""

    def test_init(self):
        """Verify that initialization stores the base URL and parameters."""
        client = STTClient(
            base_url="http://localhost:9000",
            language="es",
            translate=True,
        )
        assert client.base_url == "http://localhost:9000"
        assert client.language == "es"
        assert client.translate is True

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Verify health_check returns True when the server responds 200."""
        client = STTClient()
        fake_resp = FakeResponse(status=200)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value = AsyncContextManagerMock(fake_resp)
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Verify health_check returns False when the server is unreachable."""
        client = STTClient()

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Connection refused")
            result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_transcribe_file_success(self, tmp_path: Path):
        """Verify transcribe_file returns the text from a successful inference."""
        client = STTClient()
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"fake_wav_data")

        fake_resp = FakeResponse(status=200, json_data={"text": "  hello world  "})

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value = AsyncContextManagerMock(fake_resp)
            result = await client.transcribe_file(wav_path)

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self):
        """Verify transcribe_file raises STTClientError when the file is missing."""
        client = STTClient()
        with pytest.raises(STTClientError, match="WAV file not found"):
            await client.transcribe_file(Path("/nonexistent/audio.wav"))

    @pytest.mark.asyncio
    async def test_transcribe_file_server_error(self, tmp_path: Path):
        """Verify transcribe_file raises STTClientError on a non-200 response."""
        client = STTClient()
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"fake_wav_data")

        fake_resp = FakeResponse(status=500, text="Internal Server Error")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value = AsyncContextManagerMock(fake_resp)
            with pytest.raises(STTClientError, match="whisper-server returned 500"):
                await client.transcribe_file(wav_path)

    @pytest.mark.asyncio
    async def test_close(self):
        """Verify that close() shuts down the aiohttp session."""
        client = STTClient()
        await client._get_session()
        assert client._session is not None
        await client.close()
        assert client._session is None or client._session.closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Verify that STTClient works as an async context manager."""
        async with STTClient() as client:
            # Session is lazily created, so assert the object itself is valid
            assert isinstance(client, STTClient)
        assert client._session is None or client._session.closed

    def test_repr(self):
        """Verify the repr string contains key fields."""
        client = STTClient(base_url="http://localhost:9000", language="fr")
        r = repr(client)
        assert "STTClient" in r
        assert "localhost:9000" in r
        assert "fr" in r
