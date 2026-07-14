# Technical Specification: Voice Shell Proof-of-Concept (Phase 1)

**Document Version**: 1.0
**Date**: 2026-07-14
**Status**: Draft
**Scope**: Python-based proof-of-concept demonstrating a voice-first, local-only interface layer on top of a standard Linux desktop.

---

## 1. Objectives

1. Validate end-to-end latency of the local voice pipeline (speech → STT → LLM → TTS → audio) on target hardware.
2. Verify that quantized, local models can reliably interpret natural language OS commands (e.g., "list files," "open the browser," "what is the time?").
3. Establish a modular codebase structure that allows Phase 2 components to be swapped in without architectural rewrites.
4. Demonstrate hands-free operation: wake word → speech input → verbal response → action execution.

**Out of Scope**: Custom OS kernel/display server, function-calling/tool-use framework (JSON schema routing), persistent memory/RAG, sandboxing, multi-user profiles, proactive agents, visual HUD.

---

## 2. System Architecture

The PoC is a single-host Python application orchestrating four external local engines via subprocesses and HTTP APIs. It runs as a user-level service on an existing Linux desktop (Zorin OS / Debian / Arch).

```
+----------------------------------------------------------------------------------+
|                           Voice Shell (Python 3.11+)                             |
|  +---------------+  +-------------+  +-------------+  +-------------------------+  |
|  | Wake Word     |  | STT Client  |  | LLM Client  |  | Action Executor       |  |
|  | Listener      |  | (whisper.cpp|  | (llama.cpp  |  | (subprocess / regex)  |  |
|  | (pvporcupine /|  |  server API)|  |  server API) |  |                       |  |
|  |  keyboard)    |  |             |  |             |  |                       |  |
|  +-------+-------+  +------+------+  +------+------+  +-----------+-----------+  |
|          |                 |                 |                     |              |
|          v                 v                 v                     v              |
|  +-------+---------------------------------+----------------------------------+  |
|  |                      Orchestrator (asyncio)                                   |  |
|  |  - State machine: IDLE → LISTENING → THINKING → SPEAKING → IDLE             |  |
|  |  - Audio I/O management (PyAudio / sounddevice)                             |  |
|  |  - Streaming text routing (STT chunks → LLM prompt → TTS chunks)            |  |
|  +-------------------------------------+---------------------------------------+  |
|                                        |                                          |
|  +-------------------------------------+---------------------------------------+  |
|  |                         TTS Client (Piper / Coqui)                            |  |
|  +----------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------+
           |                                |                               |
           v                                v                               v
   +---------------+               +-------------------+           +------------------+
   |   Microphone  |               |  Speaker (Audio)  |           |  System Shell    |
   |  (USB / PCIe) |               |  (USB / HDMI / 3.5mm|           |  (bash calls)    |
   +---------------+               +-------------------+           +------------------+
```

### 2.1 State Machine

```
[IDLE] ---(wake word / hotkey)--> [LISTENING]
[LISTENING] ---(VAD silence timeout / max duration)--> [THINKING]
[THINKING] ---(LLM first token received)--> [SPEAKING]
[SPEAKING] ---(TTS finished + action executed)--> [IDLE]
```

- **IDLE**: Wake word engine is active, STT/LLM/TTS engines are idle or in low-power standby. Audio capture is active but buffered only for wake word detection.
- **LISTENING**: Full audio buffer is captured. STT engine streams partial transcripts. Voice Activity Detection (VAD) determines end-of-speech.
- **THINKING**: Audio capture stops. Full transcript is finalized. Prompt is constructed and sent to LLM. System waits for the first token.
- **SPEAKING**: TTS engine streams audio as LLM tokens arrive. The Action Executor parses the completed LLM response for commands and executes them concurrently with TTS.

---

## 3. Component Specifications

### 3.1 Wake Word Detection (WWD)

**Role**: Transition the system from IDLE to LISTENING.

**Implementation Options**:
| Option | Library / Tool | Model Size | Notes |
|--------|--------------|------------|-------|
| **Primary** | `pvporcupine` (Picovine) | < 1MB | Commercial but free tier for personal use. Extremely fast, low CPU. |
| **Fallback** | `openwakeword` | ~1MB | Fully open-source, ONNX-based. Slightly higher CPU usage. |
| **Dev Fallback** | Global hotkey (`pynput` or `keyboard` library) | N/A | For development in noisy environments or if microphone is unavailable. |

**Interface**:
```python
class WakeWordDetector:
    def __init__(self, model_path: str, sensitivity: float = 0.5):
        ...
    
    def process_chunk(self, audio_chunk: bytes) -> bool:
        """Returns True if wake word detected in chunk."""
        ...
    
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

**Audio Requirements**:
- Format: 16-bit PCM, mono, 16kHz.
- Buffer size: 512 frames (~32ms) per `process_chunk` call.
- Must not block the audio I/O thread.

### 3.2 Speech-to-Text (STT)

**Role**: Convert user speech into text during the LISTENING state.

**Implementation**: `whisper.cpp` HTTP server (`whisper-server` binary compiled from upstream).

**Server Configuration**:
```bash
./whisper-server \
  -m models/ggml-tiny.bin \
  --host 127.0.0.1 --port 8081 \
  -pc 1 \
  --convert \
  --no-prints
```

**Model Selection by Hardware**:
| Hardware | Model | Size | Expected RTF (Real-Time Factor) |
|----------|-------|------|--------------------------------|
| Raspberry Pi 5 | `ggml-tiny.bin` | 39M | ~0.3x (faster than real-time) |
| Budget Laptop | `ggml-base.bin` | 74M | ~0.2x |
| Mid Desktop | `ggml-small.bin` | 244M | ~0.5x |

**Client Interface**:
```python
class STTClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8081"):
        ...
    
    async def transcribe_file(self, wav_path: Path) -> str:
        """Send a complete WAV file to whisper-server and return full transcript."""
        ...
    
    async def health_check(self) -> bool:
        ...
```

**Streaming Strategy (PoC)**:
For Phase 1, the PoC will use **file-based, near-real-time** transcription rather than true streaming:
1. Audio is captured in a ring buffer during LISTENING.
2. VAD detects end-of-speech.
3. The ring buffer is written to a temporary mono 16kHz WAV file.
4. The WAV file is sent to `whisper-server` via HTTP POST (`/inference`).
5. The returned text is passed to the LLM.

*Rationale*: True streaming STT requires chunked HTTP or WebSocket support, which adds complexity. File-based transcription is simpler and still achieves acceptable latency for short commands (< 3 seconds). A true streaming STT implementation is planned for Phase 2.

**Voice Activity Detection (VAD)**:
- **Library**: `webrtcvad` (Python wrapper of Google's WebRTC VAD).
- **Mode**: Aggressive mode (3) to quickly detect end-of-speech and reduce latency.
- **Logic**: Collect 30ms frames. If > 50% of frames in a 1-second window are non-speech, trigger end-of-speech.
- **Timeout**: Maximum listening duration is 30 seconds to prevent runaway capture.

### 3.3 LLM Core

**Role**: Interpret the user's transcribed text and generate a natural language response and/or an action command.

**Implementation**: `llama.cpp` HTTP server (`llama-server` binary compiled from upstream).

**Server Configuration**:
```bash
./llama-server \
  -m models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8082 \
  -c 4096 \
  --no-avx512 \
  --threads 4 \
  --n-predict 512
```

**Model Selection by Hardware**:
| Hardware | Model | Quantization | RAM | Context |
|----------|-------|-------------|-----|---------|
| Raspberry Pi 5 | `Qwen2.5-1.5B-Instruct` | Q4_K_M | ~1.2GB | 4096 |
| Budget Laptop | `Qwen2.5-7B-Instruct` | Q4_K_M | ~4.5GB | 4096 |
| Budget Laptop | `Llama-3.2-3B-Instruct` | Q4_K_M | ~2.5GB | 4096 |
| Mid Desktop | `Qwen2.5-14B-Instruct` | Q4_K_M | ~8.5GB | 8192 |

**Client Interface**:
```python
class LLMClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8082"):
        ...
    
    async def generate(self, prompt: str, system_prompt: str | None = None) -> AsyncIterator[str]:
        """Yields tokens as they are generated by llama-server (SSE streaming)."""
        ...
    
    async def health_check(self) -> bool:
        ...
```

**System Prompt**:
```
You are the voice interface of a local Linux operating system. Your name is "Nova".
You respond to natural language commands and can execute simple system actions.

Rules:
1. Be concise. Speak naturally but briefly. Users are listening, not reading.
2. If the user asks you to perform an action, start your response with the action in brackets.
3. Supported actions:
   - [EXEC:shell:ls]           (list directory)
   - [EXEC:shell:cd <path>]    (change directory)
   - [EXEC:shell:cat <path>]   (read file contents)
   - [EXEC:app:firefox]        (launch application)
   - [EXEC:time]               (read current time)
   - [EXEC:date]               (read current date)
4. If you cannot perform an action, say so and explain why.
5. Never confirm destructive actions. For this proof-of-concept, only read/list actions are permitted.

Example:
User: What files are in my home directory?
Assistant: [EXEC:shell:ls ~] Your home directory contains Documents, Downloads, and Projects.
```

**Prompt Formatting**:
Use the model's native chat template (e.g., Qwen's `<|im_start|>system...<|im_end|>`). The `llama-server` API accepts `messages` in OpenAI-compatible format and handles templating internally if `--chat-template` is passed, or the client can pre-format the prompt.

**Token Generation Limits**:
- Max tokens: 512 (enough for a concise response + action tag).
- Temperature: 0.7 (balance between creativity and determinism for commands).
- Stop sequences: `[END]`, `<|im_end|>` (model-specific).

### 3.4 Text-to-Speech (TTS)

**Role**: Convert the LLM's text response into audible speech.

**Implementation**: `piper` (local neural TTS).

**Model Selection**:
- **Voice**: `en_US-lessac-medium` (high quality, ~50MB) or `en_US-lessac-low` (lower quality, ~20MB, faster).
- **Hardware**: On a Pi, the `low` model is recommended to keep CPU available for the LLM. On a budget laptop, `medium` is acceptable.

**Execution**:
```bash
echo "Hello, how can I help you?" | piper \
  --model en_US-lessac-medium.onnx \
  --output_file - | \
  aplay -r 22050 -f S16_LE -t raw -c 1
```

**Client Interface**:
```python
class TTSClient:
    def __init__(self, model_path: Path, speaker_id: int = 0):
        ...
    
    def synthesize(self, text: str) -> bytes:
        """Returns raw PCM audio bytes (16-bit, mono, 22050Hz)."""
        ...
    
    async def stream_synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Yields audio chunks as text sentences arrive."""
        ...
```

**Streaming Strategy (PoC)**:
1. The LLM client yields tokens via SSE.
2. A `SentenceBuffer` accumulates tokens until a sentence-ending punctuation (`.`, `?`, `!`) is found.
3. The completed sentence is sent to `piper` for synthesis.
4. `piper` outputs raw PCM audio, which is immediately queued to the audio output stream.
5. This allows TTS to start speaking while the LLM is still generating the rest of the response.

**Audio Output**:
- Library: `sounddevice` or `pyaudio`.
- Format: 16-bit PCM, mono, 22050Hz (Piper's default).
- Buffer: 2048 frames to prevent underruns.

### 3.5 Action Executor

**Role**: Parse the LLM response for action tags and execute the corresponding system command.

**Implementation**: Simple regex parser over the LLM output string.

**Supported Actions (PoC)**:
| Tag | Regex | Action | Safety |
|-----|-------|--------|--------|
| `[EXEC:shell:ls]` | `\[EXEC:shell:ls\]` | `subprocess.run(['ls', '-la'], capture_output=True, text=True)` | Read-only. Safe. |
| `[EXEC:shell:cd <path>]` | `\[EXEC:shell:cd (.+?)\]` | `os.chdir(path)` | Validates path exists. Read-only. |
| `[EXEC:shell:cat <path>]` | `\[EXEC:shell:cat (.+?)\]` | `subprocess.run(['cat', path], ...)` | Validates path exists. Read-only. |
| `[EXEC:app:<name>]` | `\[EXEC:app:(\w+)\]` | `subprocess.Popen([name])` | Whitelist of safe apps (e.g., firefox, code, nautilus). |
| `[EXEC:time]` | `\[EXEC:time\]` | `datetime.datetime.now().strftime('%I:%M %p')` | Safe. |
| `[EXEC:date]` | `\[EXEC:date\]` | `datetime.datetime.now().strftime('%A, %B %d')` | Safe. |

**Interface**:
```python
class ActionExecutor:
    def __init__(self, allowed_commands: list[str]):
        ...
    
    def parse_and_execute(self, llm_response: str) -> tuple[str, str]:
        """
        Returns (cleaned_response, action_result).
        cleaned_response is the LLM response with action tags stripped.
        action_result is the stdout of the executed command or an error message.
        """
        ...
```

**Safety Rules**:
1. All actions are **read-only or launch-only**. No delete, write, or format actions.
2. Shell commands are **not** executed via `shell=True`. They use a strict subprocess call with a whitelist.
3. App launching uses a whitelist of known desktop applications.
4. All file paths are validated with `os.path.exists()` and `os.path.isabs()` checks before access.
5. The executor logs all actions to a local file (`~/.voice_shell/actions.log`).

---

## 4. Audio I/O Subsystem

### 4.1 Audio Input (Capture)

**Library**: `sounddevice` (Python wrapper around PortAudio). Preferred over PyAudio for better Linux support and async-friendly callback API.

**Configuration**:
```python
import sounddevice as sd
import numpy as np

stream = sd.RawInputStream(
    samplerate=16000,
    channels=1,
    dtype='int16',
    blocksize=512,  # 32ms chunks
    callback=audio_callback
)
```

**Buffering Strategy**:
- A `collections.deque` (maxlen ~ 16000 * 30) stores the last 30 seconds of audio as 16-bit integers.
- During IDLE, the buffer is continuously overwritten. The WWD receives the latest 32ms chunk.
- During LISTENING, the buffer is frozen and appended to until VAD detects silence or timeout.
- At end of LISTENING, the buffer is converted to a WAV file via `wave` module and sent to STT.

### 4.2 Audio Output (Playback)

**Configuration**:
```python
stream = sd.RawOutputStream(
    samplerate=22050,
    channels=1,
    dtype='int16',
    blocksize=2048
)
```

**Queue Management**:
- A `queue.Queue` buffers synthesized PCM audio chunks.
- A separate playback thread dequeues chunks and writes them to the `RawOutputStream`.
- If the queue is empty, the stream is paused to prevent audio glitches.
- Echo cancellation (basic): The system is muted during TTS playback to prevent the microphone from hearing the speaker. This is a simple software mute in the audio capture callback, not true acoustic echo cancellation.

---

## 5. Orchestrator & State Management

The Orchestrator is the central Python `asyncio` event loop that manages the state machine and coordinates all components.

```python
class VoiceShellOrchestrator:
    def __init__(self, config: Config):
        self.state = State.IDLE
        self.wwd = WakeWordDetector(config.wwd_model)
        self.stt = STTClient(config.stt_url)
        self.llm = LLMClient(config.llm_url)
        self.tts = TTSClient(config.tts_model)
        self.executor = ActionExecutor(config.allowed_commands)
        self.audio_capture = AudioCapture()
        self.audio_playback = AudioPlayback()
    
    async def run(self):
        await self.start_engines()
        asyncio.create_task(self._audio_capture_loop())
        asyncio.create_task(self._playback_loop())
        
        while self.running:
            if self.state == State.IDLE:
                await self._idle_loop()
            elif self.state == State.LISTENING:
                await self._listening_loop()
            elif self.state == State.THINKING:
                await self._thinking_loop()
            elif self.state == State.SPEAKING:
                await self._speaking_loop()
    
    async def _idle_loop(self):
        chunk = await self.audio_capture.get_chunk()
        if self.wwd.process_chunk(chunk):
            self.transition_to(State.LISTENING)
    
    async def _listening_loop(self):
        transcript = await self.audio_capture.capture_until_silence(self.stt)
        self.last_transcript = transcript
        self.transition_to(State.THINKING)
    
    async def _thinking_loop(self):
        prompt = self._build_prompt(self.last_transcript)
        self.llm_response = ""
        self.sentence_buffer = ""
        
        async for token in self.llm.generate(prompt):
            self.llm_response += token
            self.sentence_buffer += token
            
            if self._is_sentence_end(self.sentence_buffer):
                sentence = self._extract_sentence(self.sentence_buffer)
                self.audio_playback.queue_tts(sentence)
                self.sentence_buffer = ""
        
        # Execute action after full response is received
        result = self.executor.parse_and_execute(self.llm_response)
        if result.action_result:
            self.audio_playback.queue_tts(f"Action result: {result.action_result}")
        
        self.transition_to(State.SPEAKING)
    
    async def _speaking_loop(self):
        await self.audio_playback.wait_for_empty_queue()
        self.transition_to(State.IDLE)
```

**Transition Rules**:
- `IDLE → LISTENING`: Wake word detected or hotkey pressed.
- `LISTENING → THINKING`: VAD silence detected or 30-second max duration reached.
- `THINKING → SPEAKING`: First LLM token is received (visual feedback changes to "Speaking").
- `SPEAKING → IDLE`: TTS playback queue is empty and action execution is complete.
- **Emergency Reset**: Any state can transition to `IDLE` if a critical error occurs (e.g., engine crash, audio device failure). An error tone is played.

---

## 6. Configuration & File Structure

```
voice_shell/
├── main.py                  # Entry point, argument parsing, orchestrator init
├── config.yaml              # User-editable configuration
├── requirements.txt         # Python dependencies
├── models/                  # Downloaded model files (not in git)
│   ├── porcupine/
│   │   └── wake_word.ppn
│   ├── whisper/
│   │   └── ggml-tiny.bin
│   ├── llm/
│   │   └── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
│   └── piper/
│       └── en_US-lessac-medium.onnx
├── bin/                     # Compiled external binaries (not in git)
│   ├── whisper-server
│   ├── llama-server
│   └── piper
├── src/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── config.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py       # sounddevice input
│   │   ├── playback.py      # sounddevice output + queue
│   │   └── vad.py           # webrtcvad wrapper
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── wwd.py           # porcupine / openwakeword
│   │   ├── stt.py           # whisper.cpp HTTP client
│   │   ├── llm.py           # llama.cpp HTTP client
│   │   └── tts.py           # piper subprocess wrapper
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── executor.py      # regex parser + subprocess runner
│   │   └── registry.py      # action whitelist & definitions
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── wav_writer.py    # in-memory buffer → WAV file
└── tests/
    ├── test_audio.py
    ├── test_stt.py
    ├── test_llm.py
    ├── test_tts.py
    └── test_actions.py
```

**Configuration Schema (`config.yaml`)**:
```yaml
audio:
  input_device: null          # null = default
  output_device: null
  sample_rate_input: 16000
  sample_rate_output: 22050
  chunk_size: 512
  max_recording_duration: 30  # seconds

wake_word:
  engine: "porcupine"         # porcupine | openwakeword | hotkey
  model_path: "models/porcupine/wake_word.ppn"
  keyword: "Hey Nova"
  sensitivity: 0.5

stt:
  engine: "whisper.cpp"
  binary_path: "bin/whisper-server"
  model_path: "models/whisper/ggml-tiny.bin"
  host: "127.0.0.1"
  port: 8081
  language: "en"
  translate: false

llm:
  engine: "llama.cpp"
  binary_path: "bin/llama-server"
  model_path: "models/llm/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
  host: "127.0.0.1"
  port: 8082
  context_size: 4096
  max_tokens: 512
  temperature: 0.7
  threads: 4
  gpu_layers: 0               # 0 = CPU only
  system_prompt: "..."

tts:
  engine: "piper"
  binary_path: "bin/piper"
  model_path: "models/piper/en_US-lessac-medium.onnx"
  speaker_id: 0
  volume: 0.8

actions:
  allowed_shell_commands: ["ls", "cat", "pwd", "date"]
  allowed_apps: ["firefox", "nautilus", "code", "terminal"]
  require_confirmation: false  # PoC: disabled for read-only actions

logging:
  level: "INFO"
  file: "~/.voice_shell/voice_shell.log"
```

---

## 7. External Dependencies & Build Instructions

### 7.1 Python Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**`requirements.txt`**:
```
sounddevice>=0.4.6
numpy>=1.24.0
pyaudio>=0.2.13          # fallback if sounddevice fails
webrtcvad>=2.0.10
requests>=2.31.0
aiohttp>=3.9.0
pyyaml>=6.0.1
pvporcupine>=3.0.0       # optional, requires license key
pynput>=1.7.6             # hotkey fallback
```

### 7.2 Compiling External Binaries

**whisper.cpp**:
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make -j$(nproc)  # or cmake -B build && cmake --build build --config Release
# binary: ./build/bin/whisper-server or ./server
```

**llama.cpp**:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j$(nproc) LLAMA_NO_AVX512=1  # CPU-only build
# binary: ./build/bin/llama-server or ./server
```

**Piper**:
```bash
# Download pre-built binary from GitHub releases (recommended)
# https://github.com/rhasspy/piper/releases
# Or build from source (requires onnxruntime)
```

### 7.3 Downloading Models

**Whisper**:
```bash
# Download from huggingface or whisper.cpp scripts
bash whisper.cpp/models/download-ggml-model.sh tiny
```

**LLM (GGUF)**:
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir ./models/llm/
```

**Piper Voice**:
```bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## 8. Performance & Latency Targets

### 8.1 Target Hardware: Raspberry Pi 5 (8GB)

| Pipeline Segment | Target Latency | Budget |
|------------------|----------------|--------|
| Wake Word Detection | < 50ms | CPU |
| Audio Capture + VAD | 1-2s (silence detection) | N/A |
| STT (`tiny` model) | 0.5-1s | CPU |
| LLM TTT (time-to-first-token) | 1-2s | CPU |
| LLM Full Response (100 tokens) | 2-4s | CPU |
| TTS First Audio Chunk | 0.2s | CPU |
| TTS Full Playback | 1-2s | CPU |
| **Total End-to-End** | **5-10s** | **CPU** |

### 8.2 Target Hardware: Budget Laptop (x86, 8GB RAM)

| Pipeline Segment | Target Latency | Budget |
|------------------|----------------|--------|
| Wake Word Detection | < 50ms | CPU |
| Audio Capture + VAD | 1-2s | N/A |
| STT (`base` model) | 0.3-0.5s | CPU |
| LLM TTT (3B model) | 0.5-1s | CPU |
| LLM Full Response (100 tokens) | 1-2s | CPU |
| TTS First Audio Chunk | 0.2s | CPU |
| TTS Full Playback | 1-2s | CPU |
| **Total End-to-End** | **3-6s** | **CPU** |

### 8.3 Optimization Strategies

1. **Model Warm-Up**: Both `llama-server` and `whisper-server` must be started at boot and kept loaded in RAM. Cold-start loading a GGUF model adds 5-10 seconds of latency.
2. **Memory Mapping**: `llama-server` uses `mmap` by default. Ensure the model file is on a fast SSD or eMMC (not a slow SD card on a Pi).
3. **CPU Affinity**: Pin the voice shell process to specific cores and the LLM server to others to reduce context switching.
4. **Quantization**: Always use Q4_K_M quantization for the LLM. Q8_0 is better quality but 2x the RAM and slower on CPU.
5. **Context Pruning**: Keep the conversation history short (last 3 turns + system prompt). Longer contexts increase inference time linearly.

---

## 9. Error Handling & Resilience

| Error Scenario | Detection | Recovery Action |
|---------------|-------------|-----------------|
| Engine binary not found | Startup check | Log fatal error, print setup instructions, exit. |
| Model file missing | Startup check | Log fatal error, print download command, exit. |
| Audio device unavailable | Runtime (PyAudio error) | Retry with default device. If fails, fallback to keyboard-only mode. |
| STT server timeout | HTTP timeout (10s) | Log error, play error tone, return "I didn't catch that." to user. |
| LLM server timeout | HTTP timeout (30s) | Log error, play error tone, return "I'm having trouble thinking." |
| LLM generates gibberish | Response validation (no valid chars) | Discard response, retry once with lower temperature. If fails, apologize. |
| Action execution failure | Subprocess exception | Log error, return stderr to user as spoken text. |
| Engine crash (OOM/segfault) | Process exit code | Auto-restart engine (max 3 restarts). If persistent, shut down gracefully. |
| TTS audio underrun | sounddevice callback | Increase playback buffer size. If persistent, skip TTS and print text. |

**Error Tone**: A short, non-annoying WAV file (e.g., a soft chime) played to indicate an error occurred without requiring the user to look at logs.

---

## 10. Testing & Validation Plan

### 10.1 Unit Tests

| Component | Test Case | Expected Result |
|-----------|-----------|-----------------|
| Audio Capture | Record 5s of audio, save to WAV. | WAV file is 16kHz, mono, 16-bit, correct duration. |
| VAD | Feed 5s speech + 2s silence. | Detects end-of-speech at ~5-6s. |
| WWD | Feed 10s audio with wake word at 3s. | Detects wake word exactly once. |
| STT Client | Send a known WAV file to mock server. | Returns expected transcript text. |
| LLM Client | Send prompt to mock server. | Yields expected tokens in order. |
| TTS Client | Synthesize "Hello world." | Returns non-empty PCM bytes. |
| Action Executor | Parse `[EXEC:shell:ls]` from string. | Executes `ls`, returns stdout. |
| Action Executor | Parse `[EXEC:shell:rm -rf /]` | Rejects (not in whitelist), returns error. |

### 10.2 Integration Tests

| Test Case | Hardware | Expected Result |
|-----------|----------|-----------------|
| End-to-end: "What time is it?" | Pi 5 | Response spoken within 10s. |
| End-to-end: "List my files." | Budget Laptop | `ls` output spoken within 6s. |
| End-to-end: "Open Firefox." | Budget Laptop | Firefox launches within 8s. |
| Noise resilience: Play music at 50% volume, give command. | Budget Laptop | STT accuracy > 70% (basic noise handling). |
| Long command: Dictate a 50-word sentence. | Budget Laptop | Full transcript accurate, no truncation. |

### 10.3 Manual QA Checklist

- [ ] System starts without errors on a fresh boot.
- [ ] Wake word triggers reliably from 1-2 meters away.
- [ ] System correctly ignores audio when TTS is playing (no self-triggering).
- [ ] Action tags are stripped from the spoken response.
- [ ] Graceful fallback to keyboard hotkey if microphone is unplugged.
- [ ] CPU usage is < 50% during idle (wake word only).
- [ ] CPU usage is acceptable during full pipeline (< 80% on all cores).
- [ ] No audio glitches (popping, crackling) during TTS playback.
- [ ] Logs are written and readable.

---

## 11. Security & Safety Considerations (Phase 1)

While full sandboxing is a Phase 2/3 concern, the PoC must not create a dangerous local backdoor.

1. **No Remote Exposure**: The `llama-server` and `whisper-server` HTTP APIs bind to `127.0.0.1` **only**. Never `0.0.0.0`.
2. **No Shell Injection**: The Action Executor uses a strict whitelist and `subprocess.run(..., shell=False)` with a fixed argument list. No user input is ever passed to a shell interpreter.
3. **No Network Calls from Actions**: The action whitelist contains no `curl`, `wget`, or `ping` commands.
4. **Read-Only Default**: Write/delete actions are excluded. The PoC cannot delete files or modify system settings.
5. **Logging**: All actions and LLM outputs are logged to a local, unprivileged file. Logs are not uploaded.
6. **Model Provenance**: Model files must be downloaded from official Hugging Face repositories or verified with SHA-256 checksums to prevent supply-chain attacks.

---

## 12. Future Integration Points (Phase 2+ Prep)

To ensure the PoC codebase is forward-compatible, the following abstractions must be respected:

1. **Engine Interface**: The `STTClient`, `LLMClient`, and `TTSClient` classes must expose a generic interface so that future engines (e.g., `faster-whisper`, `vLLM`, `melotts`) can be swapped in without modifying the Orchestrator.
2. **Action Registry**: The `ActionExecutor` must read from a `registry.py` dictionary. Phase 2 will replace the regex parser with a JSON schema validator and tool dispatcher.
3. **Config Schema**: The `config.yaml` must be versioned. Phase 2 will add fields for `vector_db`, `sandbox`, and `memory`.
4. **Audio Streaming**: The audio capture/playback modules must use `asyncio` and `queue`-based patterns so that true streaming STT and WebSocket-based TTS can be integrated later.
5. **State Machine**: The `Orchestrator` state machine is the foundation for Phase 3's custom compositor. Future states (e.g., `PROACTIVE`, `WAITING_CONFIRMATION`) will be added.

---

## 13. Appendix

### A. Example Session Log

```
[2026-07-14 10:00:15] STATE: IDLE → LISTENING (wake word: "Hey Nova")
[2026-07-14 10:00:17] STATE: LISTENING → THINKING (VAD silence detected, transcript: "what files are in my downloads folder")
[2026-07-14 10:00:18] LLM: [EXEC:shell:ls ~/Downloads] You have three files: report.pdf, notes.txt, and image.png.
[2026-07-14 10:00:18] ACTION: EXEC shell:ls ~/Downloads → stdout: "report.pdf\nnotes.txt\nimage.png\n"
[2026-07-14 10:00:18] STATE: THINKING → SPEAKING
[2026-07-14 10:00:19] TTS: "You have three files: report.pdf, notes.txt, and image.png."
[2026-07-14 10:00:21] STATE: SPEAKING → IDLE (playback complete)
```

### B. Glossary

| Term | Definition |
|------|------------|
| **GGUF** | Georgi Gerganov Universal Format. A file format for LLM weights optimized for `llama.cpp`. |
| **Q4_K_M** | A quantization method in GGUF that uses 4-bit weights with mixed K-quantization. Good balance of size and quality. |
| **RTF** | Real-Time Factor. A measure of STT speed. RTF=0.5 means 1 second of audio is processed in 0.5 seconds. |
| **VAD** | Voice Activity Detection. Algorithm that distinguishes speech from silence/noise. |
| **WWD** | Wake Word Detection. Always-on listening for a specific keyword. |
| **SSE** | Server-Sent Events. An HTTP protocol for streaming text from server to client. Used by `llama-server`. |
| **PCM** | Pulse-Code Modulation. Uncompressed raw audio format. |
| **mmap** | Memory-mapped file I/O. Allows `llama.cpp` to load models from disk without fully reading them into RAM. |
