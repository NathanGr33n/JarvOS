import asyncio
from pathlib import Path
from typing import Optional

import aiohttp


class STTClientError(Exception):
    """Raised when the speech-to-text engine fails to transcribe audio."""


class STTClient:
    """Async HTTP client for the ``whisper.cpp`` server (``whisper-server``).

    Communicates with the local whisper-server via its ``/inference`` endpoint
    to transcribe WAV audio files into text.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        language: str = "en",
        translate: bool = False,
        timeout: float = 30.0,
    ):
        """Initialize the STT client.

        Args:
            base_url: Base URL of the running ``whisper-server`` instance.
            language: Language code for transcription (e.g., ``en``, ``auto``).
            translate: Whether to translate non-English speech to English.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.translate = translate
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a cached aiohttp session, creating one if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def health_check(self) -> bool:
        """Check whether the whisper-server is reachable.

        Returns:
            ``True`` if the server responds with HTTP 200.
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def transcribe_file(self, wav_path: Path) -> str:
        """Send a WAV file to whisper-server and return the transcript.

        Args:
            wav_path: Path to a mono, 16-bit PCM WAV file (16 kHz recommended).

        Returns:
            The transcribed text (with leading/trailing whitespace stripped).

        Raises:
            STTClientError: If the HTTP request fails or the server returns an error.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise STTClientError(f"WAV file not found: {wav_path}")

        try:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                wav_path.open("rb"),
                filename=wav_path.name,
                content_type="audio/wav",
            )
            # whisper-server inference parameters
            data.add_field("language", self.language)
            data.add_field("translate", "true" if self.translate else "false")

            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/inference",
                data=data,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise STTClientError(
                        f"whisper-server returned {resp.status}: {body}"
                    )

                result = await resp.json()
                # whisper-server returns {"text": "..."} or similar
                text = result.get("text", "")
                return text.strip()
        except aiohttp.ClientError as exc:
            raise STTClientError(f"STT request failed: {exc}") from exc

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "STTClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"url={self.base_url}, lang={self.language}"
            f")"
        )
