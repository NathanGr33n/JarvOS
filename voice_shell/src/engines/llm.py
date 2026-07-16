import asyncio
import json
from typing import AsyncIterator, Optional

import aiohttp


class LLMClientError(Exception):
    """Raised when the LLM engine fails to generate a response."""


class LLMClient:
    """Async HTTP client for the ``llama.cpp`` server (``llama-server``).

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint with
    Server-Sent Events (SSE) streaming to yield tokens as they are generated.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        """Initialize the LLM client.

        Args:
            base_url: Base URL of the running ``llama-server`` instance.
            max_tokens: Maximum number of tokens to generate per request.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            timeout: HTTP request timeout in seconds (including streaming).
        """
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a cached aiohttp session, creating one if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def health_check(self) -> bool:
        """Check whether the llama-server is reachable.

        Returns:
            ``True`` if the server responds with HTTP 200.
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """Stream LLM tokens via the OpenAI-compatible chat completions endpoint.

        Args:
            prompt: The user message / prompt to send.
            system_prompt: Optional system-level instructions (e.g., persona, rules).
            stop_sequences: Optional list of strings that stop generation.

        Yields:
            Individual text tokens as they are generated.

        Raises:
            LLMClientError: If the HTTP request fails or the server returns an error.
        """
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if stop_sequences is not None:
            payload["stop"] = stop_sequences

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise LLMClientError(
                        f"llama-server returned {resp.status}: {body}"
                    )

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue

                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except aiohttp.ClientError as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc

    async def generate_full(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """Non-streaming wrapper that returns the complete response as a string.

        Useful for simple use cases where streaming is not required.

        Args:
            prompt: The user message / prompt to send.
            system_prompt: Optional system-level instructions.
            stop_sequences: Optional list of strings that stop generation.

        Returns:
            The full concatenated response text.
        """
        parts = []
        async for token in self.generate(prompt, system_prompt, stop_sequences):
            parts.append(token)
        return "".join(parts)

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"url={self.base_url}, "
            f"max_tokens={self.max_tokens}, "
            f"temp={self.temperature}"
            f")"
        )
