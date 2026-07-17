import asyncio
import subprocess
from pathlib import Path
from typing import Optional


class TTSClientError(Exception):
    """Raised when the TTS engine fails to synthesize audio."""


class TTSClient:
    """Local neural text-to-speech client using Piper (ONNX-based, fast).

    Synthesizes text into raw PCM audio bytes (16-bit, mono, 22050 Hz).
    Supports sentence-by-sentence streaming so that playback can start
    before the LLM has finished generating the entire response.
    """

    def __init__(
        self,
        model_path: Path,
        speaker_id: int = 0,
        binary_path: Path = Path("piper"),
    ):
        """Initialize the TTS client.

        Args:
            model_path: Path to the Piper ONNX voice model file.
            speaker_id: Speaker ID for multi-speaker models (default 0).
            binary_path: Path to the ``piper`` executable.
        """
        self.model_path = Path(model_path)
        self.speaker_id = speaker_id
        self.binary_path = Path(binary_path)

    def synthesize(self, text: str) -> bytes:
        """Synthesize a single block of text into raw PCM audio bytes.

        Args:
            text: The text to speak.

        Returns:
            Raw 16-bit PCM audio bytes (mono, 22050 Hz).

        Raises:
            TTSClientError: If the ``piper`` binary fails or returns no audio.
        """
        if not text.strip():
            return b""

        cmd = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--speaker",
            str(self.speaker_id),
            "--output_file",
            "-",  # write to stdout
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TTSClientError(
                f"Piper synthesis failed: {exc.stderr.decode('utf-8', errors='replace')}"
            ) from exc

        audio = proc.stdout
        if not audio:
            raise TTSClientError("Piper returned empty audio output.")

        return audio

    async def stream_synthesize(
        self,
        text_stream: asyncio.Queue,
        sentence_end_chars: str = ".!?",
    ) -> asyncio.Queue:
        """Sentence-by-sentence streaming synthesis.

        Reads text tokens from ``text_stream``, buffers them into sentences
        (delimited by punctuation), and yields synthesized audio chunks.

        This is a consumer-producer pattern: the caller feeds text into
        ``text_stream`` and consumes audio from the returned queue.

        Args:
            text_stream: An ``asyncio.Queue`` that yields text tokens.
            sentence_end_chars: Characters that mark the end of a sentence.

        Returns:
            An ``asyncio.Queue`` that yields raw PCM audio bytes per sentence.
            When the text stream is exhausted, a ``None`` sentinel is placed
            in the queue to signal completion.
        """
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        buffer = ""

        async def _producer():
            nonlocal buffer
            try:
                while True:
                    try:
                        token = await asyncio.wait_for(text_stream.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # If nothing arrives for 5 seconds, flush whatever
                        # is in the buffer and continue waiting.
                        if buffer.strip():
                            audio = self.synthesize(buffer.strip())
                            await audio_queue.put(audio)
                            buffer = ""
                        continue

                    if token is None:
                        break

                    buffer += token

                    # Check for complete sentences
                    while any(ch in buffer for ch in sentence_end_chars):
                        # Find the earliest sentence-ending character
                        split_idx = min(
                            (
                                buffer.index(ch)
                                for ch in sentence_end_chars
                                if ch in buffer
                            ),
                            default=-1,
                        )
                        if split_idx == -1:
                            break

                        sentence = buffer[: split_idx + 1].strip()
                        buffer = buffer[split_idx + 1 :]

                        if sentence:
                            audio = self.synthesize(sentence)
                            await audio_queue.put(audio)

                # Flush remaining buffer
                if buffer.strip():
                    audio = self.synthesize(buffer.strip())
                    await audio_queue.put(audio)
            except Exception as exc:
                await audio_queue.put(exc)
            finally:
                await audio_queue.put(None)  # sentinel

        asyncio.create_task(_producer())
        return audio_queue

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_path}, "
            f"speaker={self.speaker_id}, "
            f"binary={self.binary_path}"
            f")"
        )
