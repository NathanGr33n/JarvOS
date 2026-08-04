# Technical Specification: Custom OS Environment (Phase 3)

**Document Version**: 1.0
**Date**: 2026-08-04
**Status**: Draft / In Progress
**Scope**: Service architecture, session bootstrap, and foundation for a voice-primary Linux environment built on the existing `voice_shell` Action Core.

---

## 1. Objectives

1. Run the voice pipeline and local engines as managed long-lived services rather than ad-hoc terminal processes.
2. Provide a reproducible desktop-session bootstrap path that starts STT, LLM, TTS-ready voice shell, and a minimal HUD.
3. Establish packaging and unit layout that can later host a Wayland compositor and floating voice HUD.
4. Keep Phase 3 incremental: ship service foundation first; compositor/app-store later.

**Out of Scope (this slice)**: Full custom compositor implementation, model app store UI, LUKS/encrypted root, multi-user voice profiles.

---

## 2. Architecture

```
+--------------------------- Linux Session -----------------------------+
|  systemd --user (or ServiceManager fallback)                         |
|    +-- jarvos-whisper.service   (whisper.cpp server :8081)           |
|    +-- jarvos-llama.service     (llama.cpp server  :8082)            |
|    +-- jarvos-voice-shell.service (Python orchestrator + HUD)        |
|                                                                      |
|  Optional later:                                                     |
|    +-- jarvos-compositor.service (wlroots-based HUD shell)           |
+----------------------------------------------------------------------+
```

### 2.1 Design principles

- **Local-only binds**: engine HTTP APIs stay on `127.0.0.1`.
- **Ordered startup**: STT/LLM healthy before voice shell becomes interactive.
- **Graceful degradation**: if systemd user units are unavailable, `ServiceManager` can supervise subprocesses for development.
- **Config-driven paths**: binaries/models come from env vars or `config.yaml`, never hard-coded absolute user paths in units.

---

## 3. Service inventory

| Unit | Role | Default endpoint |
|------|------|------------------|
| `jarvos-whisper.service` | whisper.cpp HTTP server | `127.0.0.1:8081` |
| `jarvos-llama.service` | llama-server HTTP API | `127.0.0.1:8082` |
| `jarvos-voice-shell.service` | Phase 1/2 orchestrator | N/A (local audio) |

Shared target: `jarvos.target` pulls the stack for a session.

---

## 4. Session bootstrap

1. User logs into a graphical or tty session.
2. `jarvos.target` starts (enabled via user systemd or login script).
3. Whisper and llama units start and pass health checks.
4. Voice shell starts, loads memory/config, enters IDLE wake-word loop.
5. Text HUD (and later floating HUD) reflects state transitions.

Development fallback:

```bash
./os_environment/scripts/start_stack.sh
```

This uses the in-process/service-manager path without requiring installed units.

---

## 5. Wayland HUD roadmap (later Phase 3)

- Minimal wlroots/dwl-based compositor or layer-shell overlay on an existing compositor.
- Floating voice bar: state chip, last transcript, last response, action toast.
- Global hotkey remains available as wake fallback.

Not implemented in the foundation slice.

---

## 6. Repository layout

```
os_environment/
  systemd/                 # unit templates
  scripts/                 # install + start helpers
voice_shell/src/services/  # Python service manager + health checks
ResearchDesign/
  Phase3_Custom_OS_Environment.md
```

---

## 7. Acceptance criteria (foundation slice)

- Unit templates exist and document required environment variables.
- Install/start scripts can render units or launch the stack in dev mode.
- `ServiceManager` can start/stop/health-check configured engine processes.
- Config exposes service supervision settings.
- Tests cover manager lifecycle without requiring real model binaries.
- README marks Phase 3 as in progress (foundation).

---

## 8. Security notes

- No engine ports on non-loopback interfaces.
- Service manager never invokes a shell for command assembly.
- Future sandboxing (bubblewrap/firejail) wraps optional code-exec tools, not the core engines initially.
