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
| **Wake Word** | Porcupine / OpenWakeWord | Always-on listening for "Hey Nova" |
| **STT** | whisper.cpp (tiny/base) | Converts speech to text in real-time |
| **LLM Core** | llama.cpp + Qwen2.5/Llama 3.2 | Interprets intent, generates responses, issues commands |
| **TTS** | Piper | Fast neural text-to-speech |
| **Action Layer** | Python orchestrator + regex parser | Executes file system, app, and system commands |
| **Display** | Wayland + custom compositor (Phase 3) | Minimal HUD for status, transcripts, and visual feedback |
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

This project is now in **early implementation**. Core design docs remain in [`ResearchDesign/`](ResearchDesign/), and a working codebase exists in [`voice_shell/`](voice_shell/):

- [`AI_First_Voice_OS.md`](ResearchDesign/AI_First_Voice_OS.md) — High-level architecture, vision, tech stack, and roadmap
- [`Voice_Shell_PoC_Technical_Spec.md`](ResearchDesign/Voice_Shell_PoC_Technical_Spec.md) — Detailed technical specification for Phase 1 (Voice Shell proof-of-concept)
- [`Phase3_Custom_OS_Environment.md`](ResearchDesign/Phase3_Custom_OS_Environment.md) — Phase 3 service architecture and session bootstrap design
- [`voice_shell/`](voice_shell/) — Implemented Phase 1 pipeline plus Phase 2 Action Core (tool schemas, expanded safe OS tools, confirmation gates, SQLite memory, structured actions, text HUD)
- [`os_environment/`](os_environment/) — Phase 3 foundation: systemd unit templates, install/start scripts, and service supervision helpers

### Implementation Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1: Voice Shell** | Python PoC on a standard Linux desktop. Wake word → STT → LLM → TTS → action execution. | **Implemented (active hardening)** |
| **Phase 2: Action Core** | Structured function-calling with JSON tool schemas, persistent memory, and a text-based HUD. | **In Progress (schemas, tools, confirmation, memory landed)** |
| **Phase 3: Custom OS** | Boot into a minimal Wayland environment with a floating voice HUD and voice-controlled app launcher. | **In Progress (service foundation + units)** |
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

## Quick Start (Phase 1 PoC)

> Phase 1 is a user-level Python application that runs on any existing Linux desktop. It does not require installing a custom OS.

### Prerequisites

- Linux (Zorin OS, Debian, Arch, or Ubuntu)
- Python 3.11+
- A working microphone and speakers
- ~2GB free disk space for models

### 1. Clone the Repository

```bash
git clone https://github.com/NathanGr33n/JarvOS.git
cd JarvOS
```

### 2. Install External Engines

```bash
# whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp && make -j$(nproc)

# llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j$(nproc) LLAMA_NO_AVX512=1

# Piper (download pre-built binary)
# https://github.com/rhasspy/piper/releases
```

### 3. Download Models

```bash
pip install huggingface-hub

# Whisper tiny model
bash whisper.cpp/models/download-ggml-model.sh tiny

# LLM (Qwen2.5-1.5B)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir ./models/llm/

# Piper voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### 4. Install Python Dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Configure & Run

Copy and edit `config.yaml` to point to your engine binaries and model paths, then:

```bash
python main.py
```

Say **"Hey Nova"** and ask a question like *"What files are in my home directory?"* or *"What time is it?"*

For full configuration details, engine setup, and build instructions, see the [Phase 1 Technical Specification](ResearchDesign/Voice_Shell_PoC_Technical_Spec.md).

---

## Tech Stack Summary

| Layer | Technology |
|-------|------------|
| **Kernel** | Linux (Mainline) |
| **Base OS** | Debian / Arch / Alpine |
| **Audio** | PipeWire + ALSA |
| **Display** | Wayland + Custom Compositor (Phase 3) |
| **Wake Word** | OpenWakeWord / Porcupine |
| **STT** | whisper.cpp |
| **LLM Engine** | llama.cpp |
| **LLM Models** | Qwen2.5 / Phi-3 / Llama 3.2 (GGUF Q4) |
| **TTS** | Piper |
| **Vector DB** | sqlite-vec (Phase 2+) |
| **Languages** | Python / Rust / C++ |
| **Sandboxing** | Bubblewrap / Firejail (Phase 2+) |

---

## Security & Privacy

- **No Remote Exposure**: All internal APIs bind to `127.0.0.1` only.
- **No Shell Injection**: Action execution uses strict whitelists and `subprocess` without shell interpretation.
- **Read-Only Default**: Phase 1 cannot delete or modify files. Destructive actions require explicit confirmation (Phase 2+).
- **Encrypted Storage**: User data and model weights are stored on an encrypted partition (LUKS) in future phases.
- **No Telemetry**: No data collection, no model training on user data, no cloud logging.

---

## Contributing

JarvOS is in early research and design. Contributions, feedback, and ideas are welcome. Please open an issue or discussion to share thoughts on architecture, model choices, or hardware targets before submitting code.

---

## License

[License TBD — to be determined]

---

*Built for privacy, designed for voice, and made to run on anything from a Raspberry Pi to a gaming rig.*
