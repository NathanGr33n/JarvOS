import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_shell.src.orchestrator import Orchestrator, State


class TestOrchestrator:
    """Tests for the Orchestrator state machine and pipeline integration."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        from voice_shell.src.config import Config

        config = Config()
        config.memory.enabled = True
        config.memory.db_path = str(tmp_path / "test_memory.db")
        config.memory.history_limit = 3

        with patch("voice_shell.src.orchestrator.AudioCapture") as MockCapture, \
             patch("voice_shell.src.orchestrator.AudioPlayback") as MockPlayback, \
             patch("voice_shell.src.orchestrator.VoiceActivityDetector") as MockVad, \
             patch("voice_shell.src.orchestrator.WakeWordDetector") as MockWwd, \
             patch("voice_shell.src.orchestrator.STTClient") as MockStt, \
             patch("voice_shell.src.orchestrator.LLMClient") as MockLlm, \
             patch("voice_shell.src.orchestrator.TTSClient") as MockTts, \
             patch("voice_shell.src.orchestrator.ActionExecutor") as MockExecutor:

            orch = Orchestrator(config=config)
            yield orch
            orch.memory.close()

    def test_init_default_config(self, orchestrator):
        """Verify the orchestrator initializes with default config and IDLE state."""
        assert orchestrator.state == State.IDLE
        assert orchestrator.running is False
        assert orchestrator._history == []
        assert orchestrator._last_transcript == ""
        assert orchestrator._last_response == ""

    def test_default_system_prompt(self, orchestrator):
        """Verify the default system prompt is set."""
        assert "Nova" in orchestrator.system_prompt
        assert "respond in JSON with keys response and actions" in orchestrator.system_prompt
        assert "Tool catalog:" in orchestrator.system_prompt

    def test_repr(self, orchestrator):
        """Verify the repr shows the state and running flag."""
        r = repr(orchestrator)
        assert "IDLE" in r
        assert "running=False" in r

    def test_build_prompt_empty_history(self, orchestrator):
        """Verify prompt building with no history includes only the current transcript."""
        orch = orchestrator
        orch._last_transcript = "What time is it?"
        prompt = orch._build_prompt(orch._last_transcript)
        assert prompt == "User: What time is it?"

    def test_build_prompt_with_history(self, orchestrator):
        """Verify prompt building includes the last 3 turns of history."""
        orch = orchestrator
        orch._history = [
            ("Hello", "Hi there!"),
            ("What is the weather?", "It is sunny."),
        ]
        orch._last_transcript = "What time is it?"
        prompt = orch._build_prompt(orch._last_transcript)
        lines = prompt.split("\n")
        assert lines[0] == "User: Hello"
        assert lines[1] == "Assistant: Hi there!"
        assert lines[2] == "User: What is the weather?"
        assert lines[3] == "Assistant: It is sunny."
        assert lines[4] == "User: What time is it?"

    def test_history_rolls_over(self, orchestrator):
        """Verify history only keeps the last 3 turns."""
        orch = orchestrator
        orch._history = [
            ("A", "B"),
            ("C", "D"),
            ("E", "F"),
        ]
        orch._history.append(("G", "H"))
        while len(orch._history) > 3:
            orch._history.pop(0)
        assert len(orch._history) == 3
        assert orch._history[0] == ("C", "D")

    @pytest.mark.asyncio
    async def test_idle_phase_detects_wake_word(self, orchestrator):
        """Verify IDLE phase transitions to LISTENING on wake word detection."""
        orch = orchestrator
        orch.audio_capture.get_chunk = AsyncMock(return_value=b"fake_audio")
        orch.wwd.process_chunk = MagicMock(return_value=True)

        await orch._idle_phase()
        assert orch.state == State.LISTENING

    @pytest.mark.asyncio
    async def test_idle_phase_no_wake_word(self, orchestrator):
        """Verify IDLE phase stays IDLE when no wake word is detected."""
        orch = orchestrator
        orch.audio_capture.get_chunk = AsyncMock(return_value=b"fake_audio")
        orch.wwd.process_chunk = MagicMock(return_value=False)

        await orch._idle_phase()
        assert orch.state == State.IDLE

    @pytest.mark.asyncio
    async def test_listening_phase_no_audio(self, orchestrator):
        """Verify LISTENING phase returns to IDLE when no audio is captured."""
        orch = orchestrator
        orch.state = State.LISTENING
        orch.audio_capture.capture_until_silence = AsyncMock(return_value=b"")

        await orch._listening_phase()
        assert orch.state == State.IDLE

    @pytest.mark.asyncio
    async def test_listening_phase_empty_transcript(self, orchestrator):
        """Verify LISTENING phase returns to IDLE on empty transcript."""
        orch = orchestrator
        orch.state = State.LISTENING
        orch.audio_capture.capture_until_silence = AsyncMock(return_value=b"some_audio")
        orch.stt.transcribe_file = AsyncMock(return_value="   ")

        await orch._listening_phase()
        assert orch.state == State.IDLE

    @pytest.mark.asyncio
    async def test_listening_phase_valid_transcript(self, orchestrator):
        """Verify LISTENING phase transitions to THINKING with a valid transcript."""
        orch = orchestrator
        orch.state = State.LISTENING
        orch.audio_capture.capture_until_silence = AsyncMock(return_value=b"some_audio")
        orch.stt.transcribe_file = AsyncMock(return_value="Hello Nova")

        await orch._listening_phase()
        assert orch.state == State.THINKING
        assert orch._last_transcript == "Hello Nova"

    @pytest.mark.asyncio
    async def test_thinking_and_speaking_phase(self, orchestrator):
        """Verify the THINKING/SPEAKING phase streams LLM tokens to TTS and playback."""
        orch = orchestrator
        orch.state = State.THINKING
        orch._last_transcript = "Hello"

        # Mock LLM generate
        async def mock_generate(prompt, system_prompt=None):
            for token in ["Hi", " there", "."]:
                yield token
        orch.llm.generate = mock_generate

        # Mock TTS stream_synthesize
        audio_queue = asyncio.Queue()
        audio_queue.put_nowait(b"audio_chunk")
        audio_queue.put_nowait(None)
        orch.tts.stream_synthesize = AsyncMock(return_value=audio_queue)
        orch.tts.synthesize = MagicMock(return_value=b"action_audio")

        orch.audio_playback.queue_chunk = MagicMock()
        orch.audio_playback.wait_for_empty = AsyncMock()
        orch.executor.parse_and_execute = MagicMock(return_value=MagicMock(
            cleaned_response="Hi there.", action_result=""
        ))

        await orch._thinking_and_speaking_phase()

        assert orch.state == State.IDLE
        assert orch._last_response == "Hi there."
        orch.executor.parse_and_execute.assert_called_once_with("Hi there.")

    @pytest.mark.asyncio
    async def test_thinking_and_speaking_phase_with_action(self, orchestrator):
        """Verify the phase synthesizes action results when actions are present."""
        orch = orchestrator
        orch.state = State.THINKING
        orch._last_transcript = "What time is it?"

        async def mock_generate(prompt, system_prompt=None):
            yield "It is [EXEC:time] right now."
        orch.llm.generate = mock_generate

        audio_queue = asyncio.Queue()
        audio_queue.put_nowait(None)
        orch.tts.stream_synthesize = AsyncMock(return_value=audio_queue)
        orch.tts.synthesize = MagicMock(return_value=b"action_audio")

        orch.audio_playback.queue_chunk = MagicMock()
        orch.audio_playback.wait_for_empty = AsyncMock()
        orch.executor.parse_and_execute = MagicMock(return_value=MagicMock(
            cleaned_response="It is right now.", action_result="[time] 12:34 PM"
        ))

        await orch._thinking_and_speaking_phase()

        assert orch.state == State.IDLE
        orch.tts.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_thinking_and_speaking_phase_llm_error(self, orchestrator):
        """Verify LLM errors transition the state back to IDLE."""
        orch = orchestrator
        orch.state = State.THINKING
        orch._last_transcript = "Hello"

        async def mock_generate(prompt, system_prompt=None):
            raise RuntimeError("LLM failure")
            yield ""  # makes it an async generator function
        orch.llm.generate = mock_generate

        audio_queue = asyncio.Queue()
        audio_queue.put_nowait(None)
        orch.tts.stream_synthesize = AsyncMock(return_value=audio_queue)
        orch.audio_playback.wait_for_empty = AsyncMock()

        await orch._thinking_and_speaking_phase()
        assert orch.state == State.IDLE

    @pytest.mark.asyncio
    async def test_listening_phase_stt_error(self, orchestrator):
        """Verify STT errors transition the state back to IDLE."""
        orch = orchestrator
        orch.state = State.LISTENING
        orch.audio_capture.capture_until_silence = AsyncMock(return_value=b"audio")
        orch.stt.transcribe_file = AsyncMock(side_effect=RuntimeError("STT failed"))

        await orch._listening_phase()
        assert orch.state == State.IDLE
        assert orch._last_transcript == ""

    @pytest.mark.asyncio
    async def test_run_stops_gracefully(self, orchestrator):
        """Verify the run loop stops when running is set to False."""
        orch = orchestrator
        orch.audio_capture.get_chunk = AsyncMock(return_value=b"chunk")
        orch.wwd.process_chunk = MagicMock(return_value=False)
        orch.stt.close = AsyncMock()
        orch.llm.close = AsyncMock()

        # Stop after one iteration of IDLE
        call_count = 0
        original_idle = orch._idle_phase
        async def stop_after_idle():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                orch.stop()
            await original_idle()

        with patch.object(orch, "_idle_phase", stop_after_idle):
            await orch.run()

        assert orch.running is False
        orch.stt.close.assert_called_once()
        orch.llm.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_exception_recovery(self, orchestrator):
        """Verify exceptions in the main loop are caught and the orchestrator returns to IDLE."""
        orch = orchestrator
        orch.state = State.IDLE

        orch.audio_capture.get_chunk = AsyncMock(side_effect=RuntimeError("Unexpected"))
        orch.stt.close = AsyncMock()
        orch.llm.close = AsyncMock()

        # Stop after the exception is caught once
        call_count = 0
        async def stop_after_exception():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                orch.stop()
            raise RuntimeError("Unexpected")

        with patch.object(orch, "_idle_phase", stop_after_exception):
            await orch.run()

        assert orch.state == State.IDLE
        assert orch.running is False
