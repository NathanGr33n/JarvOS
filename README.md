# JarvOS

**A local-first, voice-native operating system that turns your computer into a conversational assistant.**

JarvOS is an AI-first OS where your voice is the primary interface. Speak naturally to launch apps, manage files, write code, browse the web, and control your system. All inference, speech processing, and data storage happens on-device — no cloud dependencies, no API keys, and no privacy compromises.

Think of it as your own personal Jarvis: always listening, always local, always yours.

---

## Vision

JarvOS eliminates the traditional GUI and CLI as primary interfaces. Instead, you converse with your computer. The system responds verbally and visually, making it fully operable hands-free and eyes-free while retaining a lightweight visual layer for complex tasks.

The core philosophy is **local-first autonomy**:
- All AI inference runs on-device using quantized, open-source models
- Speech-to-text and text-to-speech are fully local
- No network calls for core functionality
- Complete user privacy — your data never leaves your machine

---

## Why JarvOS?

Existing voice assistants (Siri, Alexa, Google Assistant) are cloud-based, proprietary, and privacy-invasive. Local alternatives like Mycroft or OpenVoiceOS still often rely on cloud STT/LLM backends and are focused on smart-home skills rather than full OS control.

JarvOS is different:
- **True local-only**: No cloud, no APIs, no internet required for core operation
- **Voice-native OS design**: Not a chatbot bolted onto a desktop, but an OS built around conversational interaction
- **Minimal hardware footprint**: Runs on a Raspberry Pi 5 or a budget laptop
- **Full system control**: Launch apps, manage files, write code, configure settings — all by voice

---

## Repository Layout

```
JarvOS/
├── voice_shell/           # Phase 1–2 Python voice pipeline + Action Core
│   ├── main.py            # CLI entrypoint (run | status | models)
│   ├── config.yaml        # Default configuration
│   ├── requirements.txt   # Runtime Python dependencies
│   ├── src/               # Engines, actions, HUD, memory, services, app store
│   └── tests/             # Unit tests (pytest)
├── os_environment/        # Phase 3 systemd units, scripts, optional dwl compositor
├── ResearchDesign/        # Architecture and phase design docs
├── pytest.ini             # Pytest config (run from repo root)
└── README.md
```

---

## Architecture Overview

JarvOS is built as a modular pipeline of specialized, lightweight models coordinated by a central orchestrator:

```
Microphone → Wake Word → STT → LLM Core → TTS → Speaker
                                      ↓
                               Action Executor
                                      ↓
                               System / Apps
```

### Key Components

| Layer | Technology | Role |
|-------|------------|------|
| **Wake Word** | Porcupine / OpenWakeWord / hotkey fallback | Always-on listening for "Hey Nova" (default config uses `Ctrl+Shift+Space`) |
| **STT** | whisper.cpp server | Converts speech to text over local HTTP |
| **LLM Core** | llama.cpp server + Qwen2.5/Llama 3.2 | Interprets intent, generates responses, issues structured actions |
| **TTS** | Piper | Fast neural text-to-speech |
| **Action Layer** | Python orchestrator + JSON tool schemas | Executes safe OS tools with optional confirmation gates |
| **Memory** | SQLite | Conversation history and key/value facts |
| **HUD** | Text + GTK4 floating overlay | Status, transcripts, and visual feedback (`hud.mode: text|floating|both`) |
| **Display** | Wayland + optional dwl compositor (Phase 3) | Layer-shell overlay for the floating HUD |
| **Base OS** | Linux (Debian/Arch/Alpine) | Stripped-down, audio-first, lightweight |

---

## Target Hardware

| Profile | Minimum Specs | Expected LLM Model | End-to-End Latency |
|---------|---------------|-------------------|--------------------|
| **Raspberry Pi 5** | 8GB RAM | Qwen2.5-1.5B (Q4) | 5–10s |
| **Budget Laptop** | x86, 8GB RAM, iGPU | Qwen2.5-7B or Llama 3.2-3B (Q4) | 3–6s |
| **Mid-Range Desktop** | x86, 16GB RAM, dGPU | Qwen2.5-14B or Mistral-Nemo-12B (Q4) | Near real-time |

---

## Project Status

This project is in **early implementation**. Design docs live in [`ResearchDesign/`](ResearchDesign/), and the working codebase is under [`voice_shell/`](voice_shell/) and [`os_environment/`](os_environment/):

- [`AI_First_Voice_OS.md`](ResearchDesign/AI_First_Voice_OS.md) — High-level architecture, vision, tech stack, and roadmap
- [`Voice_Shell_PoC_Technical_Spec.md`](ResearchDesign/Voice_Shell_PoC_Technical_Spec.md) — Detailed technical specification for Phase 1
- [`Phase3_Custom_OS_Environment.md`](ResearchDesign/Phase3_Custom_OS_Environment.md) — Phase 3 service architecture and session bootstrap design
- [`voice_shell/`](voice_shell/) — Phase 1 pipeline, Phase 2 Action Core (tool schemas, safe OS tools, confirmation gates, SQLite memory, structured actions, text/floating HUD), and a local model app store (`models list|status|download`)
- [`os_environment/`](os_environment/) — Phase 3 foundation: systemd unit templates, install/start scripts, service supervision helpers, and an opt-in dwl-based compositor (`os_environment/compositor/`)

### Implementation Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1: Voice Shell** | Python PoC on a standard Linux desktop. Wake word → STT → LLM → TTS → action execution. | **Implemented (active hardening)** |
| **Phase 2: Action Core** | Structured function-calling with JSON tool schemas, persistent memory, and a text-based HUD. | **In Progress (schemas, tools, confirmation, memory landed)** |
| **Phase 3: Custom OS** | Boot into a minimal Wayland environment with a floating voice HUD and voice-controlled app launcher. | **In Progress (services + HUD + health gates + opt-in dwl compositor + model app store)** |
| **Phase 4: Hardware Tuning** | Optimize for Raspberry Pi 5, model swapping, and hardware-accelerated inference. | **Planned** |
| **Phase 5: Advanced Agents** | Proactive suggestions, multi-user voice profiles, local email/calendar, and advanced coding assistant. | **Planned** |

---

## Core Principles

1. **Voice as Primary Input** — Keyboard and mouse are secondary, fallback interfaces.
2. **Local-Only Inference** — All LLM, STT, and TTS run on-device. No cloud.
3. **Minimal Hardware Footprint** — Must run on a Raspberry Pi 5 or budget hardware.
4. **Privacy by Design** — User data never leaves the device.
5. **Graceful Degradation** — Falls back to simpler models or text input when under load.

---

## Quick Start (Voice Shell)

> The voice shell is a user-level Python application that runs on any existing Linux desktop. It does not require installing a custom OS.

### Prerequisites

- Linux (Zorin OS, Debian, Arch, or Ubuntu)
- Python 3.11+
- A working microphone and speakers
- Local whisper.cpp and llama.cpp HTTP servers (or configure `services.*` to supervise them)
- Piper binary on `PATH` (or set `tts.binary_path` in config)
- Disk space for models (varies by profile; start with ~2GB)

### 1. Clone the Repository

```bash
git clone https://github.com/NathanGr33n/JarvOS.git
cd JarvOS
```

### 2. Create a Virtual Environment and Install Dependencies

Run all commands from the **repository root** so `voice_shell` imports resolve:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r voice_shell/requirements.txt

# Recommended for the test suite
pip install pytest pytest-asyncio
```

Optional floating HUD (system packages):

```bash
# Debian/Ubuntu/Zorin
sudo apt install python3-gi gir1.2-gtk-4.0
# Optional Wayland layer-shell overlay support
# sudo apt install gir1.2-gtk4layershell-1.0   # package name may vary
```

### 3. Install External Engines

```bash
# whisper.cpp (build the server binary you will point config at)
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp && cmake -B build && cmake --build build -j"$(nproc)"

# llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && cmake -B build && cmake --build build -j"$(nproc)"

# Piper — download a pre-built binary for your platform
# https://github.com/rhasspy/piper/releases
```

Start the engines bound to localhost only (example ports match `voice_shell/config.yaml`):

```bash
# STT — adjust binary/model paths for your machine
./whisper.cpp/build/bin/whisper-server \
  -m /path/to/ggml-tiny.bin \
  --host 127.0.0.1 --port 8081

# LLM
./llama.cpp/build/bin/llama-server \
  -m /path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8082
```

Alternatively, install the user systemd units under [`os_environment/`](os_environment/systemd/README.md) and set `JARVOS_WHISPER_BIN`, `JARVOS_WHISPER_MODEL`, `JARVOS_LLAMA_BIN`, `JARVOS_LLAMA_MODEL`, and `JARVOS_CONFIG`.

### 4. Download Models

Preferred: use the built-in local model app store (writes under `./models/`, which is gitignored):

```bash
# From the repository root, with the venv active
python -m voice_shell.main models list
python -m voice_shell.main models download whisper-tiny
python -m voice_shell.main models download piper-lessac-medium
python -m voice_shell.main models status
```

Catalog entries currently include STT/TTS artifacts such as `whisper-tiny`, `whisper-base`, and `piper-lessac-medium`. LLM GGUF weights are still fetched manually, for example:

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir ./models/llm/
```

Point `tts.model_path` and your llama/whisper server flags at the downloaded files. Piper also needs the matching `.onnx.json` next to the voice model when required by your Piper build.

### 5. Configure

Edit [`voice_shell/config.yaml`](voice_shell/config.yaml) (or copy it and pass `--config`):

- `stt.base_url` / `llm.base_url` — local engine endpoints (default `127.0.0.1:8081` / `:8082`)
- `tts.binary_path` / `tts.model_path` — Piper binary and voice model
- `wake_word.engine` — default is `"hotkey"` (`Ctrl+Shift+Space`); set a model path for Porcupine/OpenWakeWord
- `hud.mode` — `text`, `floating`, or `both`
- `memory.db_path` — default `data/memory.db` (gitignored)
- `actions.*` — command/app allowlists and confirmation gates
- `health.*` — startup readiness gates for STT/LLM/TTS

### 6. Run

Always invoke from the **repository root**:

```bash
source .venv/bin/activate

# Health gates + supervised service snapshot (no full voice loop)
python -m voice_shell.main --config voice_shell/config.yaml status
python -m voice_shell.main --config voice_shell/config.yaml status --format json

# Start the orchestrator
python -m voice_shell.main --config voice_shell/config.yaml run
# or simply:
python -m voice_shell.main --config voice_shell/config.yaml
```

With the default hotkey wake backend, press **Ctrl+Shift+Space**, then speak. With a wake-word model configured, say **"Hey Nova"**.

Examples: *"What files are in my home directory?"* or *"What time is it?"*

### 7. Tests

```bash
source .venv/bin/activate
pip install pytest pytest-asyncio   # if not already installed
pytest
```

Tests run headless by default (`JARVOS_HUD_HEADLESS=1` is set in `voice_shell/tests/conftest.py`).

For engine setup details and the full Phase 1 design, see the [Phase 1 Technical Specification](ResearchDesign/Voice_Shell_PoC_Technical_Spec.md). For systemd stack install and the optional compositor, see [`os_environment/systemd/README.md`](os_environment/systemd/README.md).

---

## Optional: systemd stack and compositor

```bash
./os_environment/scripts/install_user_units.sh
systemctl --user daemon-reload
systemctl --user enable --now jarvos.target
systemctl --user status jarvos.target
```

The opt-in dwl compositor is **not** part of `jarvos.target` (it can change your graphical session). Build and enable separately only if you want a layer-shell session for the floating HUD:

```bash
./os_environment/compositor/build.sh
systemctl --user enable --now jarvos-compositor.target
```

---

## Tech Stack Summary

| Layer | Technology |
|-------|------------|
| **Kernel** | Linux (Mainline) |
| **Base OS** | Debian / Arch / Alpine |
| **Audio** | PipeWire + ALSA |
| **Display** | Wayland + optional dwl compositor (Phase 3) |
| **Wake Word** | OpenWakeWord / Porcupine / hotkey fallback |
| **STT** | whisper.cpp |
| **LLM Engine** | llama.cpp |
| **LLM Models** | Qwen2.5 / Phi-3 / Llama 3.2 (GGUF Q4) |
| **TTS** | Piper |
| **Persistent Memory** | SQLite — conversation history + key/value facts (implemented) |
| **Vector/Semantic Search** | sqlite-vec (planned, not yet implemented) |
| **Languages** | Python / Rust / C++ |
| **Sandboxing** | Bubblewrap / Firejail (planned, not yet implemented) |

---

## Security & Privacy

- **No Remote Exposure**: All internal APIs bind to `127.0.0.1` only.
- **No Shell Injection**: Action execution uses strict allowlists and `subprocess` without shell interpretation.
- **Read-Only Default**: Destructive file actions are not exposed. Application launches and higher-risk tools can require explicit confirmation (`actions.require_confirmation`).
- **Model downloads**: The app store only fetches URLs from a curated catalog and pins SHA-256 after first download (TOFU).
- **Encrypted Storage**: User data and model weights are intended for an encrypted partition (LUKS) in future phases.
- **No Telemetry**: No data collection, no model training on user data, no cloud logging.

---

## Contributing

JarvOS is in early implementation. Contributions, feedback, and ideas are welcome. Please open an issue or discussion to share thoughts on architecture, model choices, or hardware targets before submitting large code changes.

Suggested workflow:
1. Keep changes scoped and testable (`pytest` from the repo root).
2. Do not commit models, virtualenvs, `data/`, or compositor build trees (see `.gitignore`).
3. Prefer feature branches for larger work, verify functionality/security, then merge back.

---

## License

[License TBD — to be determined]

---

*Built for privacy, designed for voice, and made to run on anything from a Raspberry Pi to a gaming rig.*
