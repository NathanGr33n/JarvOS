import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from voice_shell.src.engines.llm import LLMClient, LLMClientError


class AsyncContextManagerMock:
    """A reusable async context manager mock that returns a response object."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class FakeSSEResponse:
    """Fake aiohttp response that yields Server-Sent Event lines."""

    def __init__(self, status, lines):
        self.status = status
        self._lines = lines

    async def text(self):
        return "\n".join(self._lines)

    @property
    def content(self):
        """Return an async iterable that yields bytes lines."""
        queue = asyncio.Queue()
        for line in self._lines:
            queue.put_nowait(line.encode("utf-8"))
        return _QueueAsyncIterator(queue)


class _QueueAsyncIterator:
    """Async iterator wrapper around an asyncio.Queue."""

    def __init__(self, queue):
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._queue.empty():
            raise StopAsyncIteration
        return await self._queue.get()


class TestLLMClient:
    """Tests for the LLMClient llama.cpp wrapper."""

    def test_init(self):
        """Verify that initialization stores the base URL and parameters."""
        client = LLMClient(
            base_url="http://localhost:9000",
            max_tokens=256,
            temperature=0.5,
        )
        assert client.base_url == "http://localhost:9000"
        assert client.max_tokens == 256
        assert client.temperature == 0.5

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Verify health_check returns True when the server responds 200."""
        client = LLMClient()
        fake_resp = FakeSSEResponse(status=200, lines=[])

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value = AsyncContextManagerMock(fake_resp)
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Verify health_check returns False when the server is unreachable."""
        client = LLMClient()

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Connection refused")
            result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_streams_tokens(self):
        """Verify that generate yields tokens from an SSE stream."""
        client = LLMClient()
        lines = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}",
            "data: [DONE]",
        ]
        fake_resp = FakeSSEResponse(status=200, lines=lines)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value = AsyncContextManagerMock(fake_resp)
            tokens = []
            async for token in client.generate("Say hello"):
                tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Verify that the system prompt is included in the request payload."""
        client = LLMClient()
        lines = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Yes\"}}]}",
            "data: [DONE]",
        ]
        fake_resp = FakeSSEResponse(status=200, lines=lines)

        captured_payload = None

        def _capture_post(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return AsyncContextManagerMock(fake_resp)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = _capture_post
            async for _ in client.generate("question", system_prompt="You are a test bot."):
                pass

        assert captured_payload is not None
        messages = captured_payload["messages"]
        assert messages[0] == {"role": "system", "content": "You are a test bot."}
        assert messages[1] == {"role": "user", "content": "question"}

    @pytest.mark.asyncio
    async def test_generate_server_error(self):
        """Verify that generate raises LLMClientError on a non-200 response."""
        client = LLMClient()
        fake_resp = FakeSSEResponse(status=500, lines=["Internal Server Error"])
        fake_resp.text = AsyncMock(return_value="Internal Server Error")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value = AsyncContextManagerMock(fake_resp)
            with pytest.raises(LLMClientError, match="llama-server returned 500"):
                async for _ in client.generate("question"):
                    pass

    @pytest.mark.asyncio
    async def test_generate_full(self):
        """Verify that generate_full returns the concatenated response string."""
        client = LLMClient()
        lines = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"The answer\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\" is 42.\"}}]}",
            "data: [DONE]",
        ]
        fake_resp = FakeSSEResponse(status=200, lines=lines)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value = AsyncContextManagerMock(fake_resp)
            result = await client.generate_full("What is the answer?")

        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_close(self):
        """Verify that close() shuts down the aiohttp session."""
        client = LLMClient()
        await client._get_session()
        assert client._session is not None
        await client.close()
        assert client._session is None or client._session.closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Verify that LLMClient works as an async context manager."""
        async with LLMClient() as client:
            # Session is lazily created, so assert the object itself is valid
            assert isinstance(client, LLMClient)
        assert client._session is None or client._session.closed

    def test_repr(self):
        """Verify the repr string contains key fields."""
        client = LLMClient(base_url="http://localhost:9000", max_tokens=128, temperature=0.9)
        r = repr(client)
        assert "LLMClient" in r
        assert "localhost:9000" in r
        assert "128" in r
        assert "0.9" in r