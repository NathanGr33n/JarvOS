import asyncio
import logging
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from .audio.capture import AudioCapture
from .audio.playback import AudioPlayback
from .audio.vad import VoiceActivityDetector
from .config import Config
from .engines.llm import LLMClient
from .engines.stt import STTClient
from .engines.tts import TTSClient
from .engines.wwd import WakeWordDetector
from .utils.wav_writer import write_wav_from_buffer
from .actions.executor import ActionExecutor
from .actions.registry import ActionRegistry

logger = logging.getLogger(__name__)


class State(Enum):
    """Voice shell operational states."""
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class Orchestrator:
    """Central coordinator that drives the voice shell pipeline.

    Manages a state machine across four phases:
    - **IDLE**: Always-on wake word detection (low CPU).
    - **LISTENING**: Capture audio with VAD; stop on silence or timeout.
    - **THINKING**: Send transcript to STT, then to LLM. Wait for first token.
    - **SPEAKING**: Stream LLM tokens to TTS and play audio; execute actions concurrently.

    All state transitions are async-safe. Exceptions in any phase are caught,
    logged, and the orchestrator returns to IDLE so the loop can continue.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are the voice interface of a local Linux operating system. "
        "Your name is Nova.\n\n"
        "Rules:\n"
        "1. Be concise. Speak naturally but briefly. Users are listening, not reading.\n"
        "2. If the user asks you to perform an action, start your response with the action in brackets.\n"
        "3. Supported actions:\n"
        "   - [EXEC:shell:ls]           (list directory)\n"
        "   - [EXEC:shell:cat <path>]   (read file contents)\n"
        "   - [EXEC:shell:cd <path>]    (change directory)\n"
        "   - [EXEC:app:firefox]        (launch application)\n"
        "   - [EXEC:time]               (read current time)\n"
        "   - [EXEC:date]               (read current date)\n"
        "4. If you cannot perform an action, say so and explain why.\n"
        "5. Never confirm destructive actions. For this proof-of-concept, only read/list actions are permitted.\n"
    )

    def __init__(self, config: Optional[Config] = None):
        """Initialize the orchestrator and all sub-components from a config.

        Args:
            config: A ``Config`` instance. If ``None``, default settings are used.
        """
        self.config = config or Config()
        self.state = State.IDLE
        self.running = False

        # Audio pipeline
        self.audio_capture = AudioCapture(
            sample_rate=self.config.audio.sample_rate_input,
            blocksize=self.config.audio.chunk_size,
            device=self.config.audio.input_device,
        )
        self.audio_playback = AudioPlayback(
            sample_rate=self.config.audio.sample_rate_output,
            device=self.config.audio.output_device,
        )
        self.vad = VoiceActivityDetector(
            aggressiveness=3,
            sample_rate=self.config.audio.sample_rate_input,
        )
        self.wwd = WakeWordDetector(
            model_path=Path(self.config.wake_word.model_path) if self.config.wake_word.model_path else None,
            keyword=self.config.wake_word.keyword,
            sensitivity=self.config.wake_word.sensitivity,
            backend=self.config.wake_word.engine,
        )

        # AI engines
        self.stt = STTClient(
            base_url=self.config.stt.base_url,
            language=self.config.stt.language,
            translate=self.config.stt.translate,
        )
        self.llm = LLMClient(
            base_url=self.config.llm.base_url,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
        )
        self.tts = TTSClient(
            model_path=Path(self.config.tts.model_path),
            speaker_id=self.config.tts.speaker_id,
            binary_path=Path(self.config.tts.binary_path),
        )

        # Action layer
        self.executor = ActionExecutor(
            registry=ActionRegistry(
                allowed_shell_commands=self.config.actions.allowed_shell_commands,
                allowed_apps=self.config.actions.allowed_apps,
            )
        )
        self.system_prompt = self.config.llm.system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Conversation history (last 3 turns for context)
        self._history: list[tuple[str, str]] = []
        self._last_transcript: str = ""
        self._last_response: str = ""

    async def run(self) -> None:
        """Run the main voice shell loop until ``stop()`` is called."""
        self.running = True
        self.audio_capture.start()
        self.audio_playback.start()
        self.wwd.start()

        logger.info("Voice shell started. State: %s", self.state.name)

        try:
            while self.running:
                try:
                    if self.state == State.IDLE:
                        await self._idle_phase()
                    elif self.state == State.LISTENING:
                        await self._listening_phase()
                    elif self.state in (State.THINKING, State.SPEAKING):
                        await self._thinking_and_speaking_phase()
                except Exception as exc:
                    logger.exception("Error in %s phase: %s", self.state.name, exc)
                    await self._transition_to(State.IDLE)
        finally:
            await self._shutdown()

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self.running = False

    async def _shutdown(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down voice shell...")
        self.audio_capture.stop()
        self.audio_playback.stop()
        self.wwd.stop()
        await self.stt.close()
        await self.llm.close()

    async def _transition_to(self, new_state: State) -> None:
        """Log and switch to a new operational state."""
        if self.state != new_state:
            logger.info("State: %s → %s", self.state.name, new_state.name)
        self.state = new_state

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _idle_phase(self) -> None:
        """Wait for the wake word by inspecting audio chunks from the capture queue."""
        chunk = await self.audio_capture.get_chunk()
        if self.wwd.process_chunk(chunk):
            logger.info("Wake word detected")
            await self._transition_to(State.LISTENING)

    async def _listening_phase(self) -> None:
        """Capture audio until VAD detects end-of-speech, then write to a WAV file."""
        audio = await self.audio_capture.capture_until_silence(
            self.vad,
            max_duration=self.config.audio.max_recording_duration,
        )

        if not audio:
            logger.warning("No audio captured; returning to IDLE")
            await self._transition_to(State.IDLE)
            return

        # Write to a temporary WAV file for the STT engine
        wav_path = Path(tempfile.gettempdir()) / "voice_shell_capture.wav"
        write_wav_from_buffer(
            audio,
            output_path=wav_path,
            sample_rate=self.config.audio.sample_rate_input,
            channels=1,
        )
        logger.info("Captured %d bytes of audio → %s", len(audio), wav_path)

        # Transcribe
        try:
            self._last_transcript = await self.stt.transcribe_file(wav_path)
        except Exception as exc:
            logger.error("STT failed: %s", exc)
            self._last_transcript = ""
            await self._transition_to(State.IDLE)
            return

        if not self._last_transcript.strip():
            logger.info("Empty transcript; returning to IDLE")
            await self._transition_to(State.IDLE)
            return

        logger.info("Transcript: %s", self._last_transcript)
        await self._transition_to(State.THINKING)

    async def _thinking_and_speaking_phase(self) -> None:
        """Send transcript to LLM, stream TTS audio, and execute actions concurrently."""
        prompt = self._build_prompt(self._last_transcript)

        # Prepare TTS streaming
        text_queue: asyncio.Queue = asyncio.Queue()
        audio_queue = await self.tts.stream_synthesize(text_queue)

        # Start consuming audio chunks from TTS and feeding them to playback
        playback_consumer = asyncio.create_task(self._consume_tts_audio(audio_queue))

        # Stream LLM tokens and feed them to the TTS text queue
        llm_response = ""
        try:
            async for token in self.llm.generate(prompt, system_prompt=self.system_prompt):
                llm_response += token
                await text_queue.put(token)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            await text_queue.put(None)
            await playback_consumer
            await self._transition_to(State.IDLE)
            return

        # Signal end of text stream to TTS
        await text_queue.put(None)

        # Wait for all audio to finish playing
        await playback_consumer
        await self.audio_playback.wait_for_empty()

        # Store the full response (with action tags) for history
        self._last_response = llm_response

        # Execute actions from the response
        result = self.executor.parse_and_execute(llm_response)
        if result.cleaned_response:
            logger.info("Response: %s", result.cleaned_response)

        # If there were action results, speak them too
        if result.action_result:
            logger.info("Action result: %s", result.action_result)
            try:
                action_audio = self.tts.synthesize(f"Action result: {result.action_result}")
                self.audio_playback.queue_chunk(action_audio)
                await self.audio_playback.wait_for_empty()
            except Exception as exc:
                logger.error("TTS synthesis of action result failed: %s", exc)

        # Update conversation history
        self._history.append((self._last_transcript, result.cleaned_response))
        while len(self._history) > 3:
            self._history.pop(0)

        await self._transition_to(State.IDLE)

    async def _consume_tts_audio(self, audio_queue: asyncio.Queue) -> None:
        """Consume raw PCM chunks from the TTS audio queue and queue them for playback."""
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                logger.error("TTS stream error: %s", chunk)
                break
            self.audio_playback.queue_chunk(chunk)

    def _build_prompt(self, transcript: str) -> str:
        """Build the LLM prompt including system context and recent history."""
        parts = []
        for user_msg, assistant_msg in self._history:
            parts.append(f"User: {user_msg}")
            parts.append(f"Assistant: {assistant_msg}")
        parts.append(f"User: {transcript}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"Orchestrator(state={self.state.name}, running={self.running})"
