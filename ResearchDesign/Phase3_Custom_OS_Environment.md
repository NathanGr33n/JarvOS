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

- **Engine health gates**: STT/LLM/TTS readiness is verified at startup and again before accepting speech after a wake word.

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
| `jarvos-compositor.service` | Optional dwl compositor session | N/A (local display) |

Shared target: `jarvos.target` pulls the engine + voice-shell stack for a
session. `jarvos-compositor.target` is a **separate, opt-in** target (see
Section 5) since enabling it changes the graphical session.

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

## 5. Floating Wayland HUD

### 5.1 Current implementation
- `voice_shell/src/hud/floating.py` — GTK4 floating voice bar
- Optional `gtk4-layer-shell` top overlay when available
- Fallback: undecorated always-on-top window (X11 / generic Wayland)
- Modes via `hud.mode`: `text`, `floating`, `both` (default)
- Events: state chip, transcript, response, action/error toast
- Thread-safe updates via GLib idle dispatch so the async orchestrator can emit freely

### 5.2 Config
```yaml
hud:
  enabled: true
  mode: both
  width: 480
  height: 140
  anchor: top-center
  margin: 24
  opacity: 0.92
```

### 5.3 Compositor integration (opt-in)
- `os_environment/compositor/build.sh` clones and builds stock `dwl` v0.8
  (wlroots-based, already supports `wlr-layer-shell`, so the existing
  floating HUD renders without any compositor source changes).
- `os_environment/compositor/session.sh` is dwl's `-s` startup command; it
  launches the voice shell as the session's app.
- `jarvos-compositor.service` + `jarvos-compositor.target` wire this into
  systemd, deliberately kept out of the default `jarvos.target` since
  enabling them changes the user's graphical session. The default unit
  nests dwl inside an existing session for safe local testing; standalone
  TTY/KMS operation on dedicated hardware is documented as an alternative
  in `os_environment/systemd/README.md`.

### 5.4 Later roadmap
- Richer visual action feedback and animation
- Global hotkey remains available as wake fallback
- A curated "app store" UI on top of the `voice_shell.src.appstore` module
  (currently CLI-only via `main.py models`)

---

## 6. Repository layout

```
os_environment/
  systemd/                 # unit templates (engines, voice shell, compositor)
  scripts/                 # install + start helpers
  compositor/              # opt-in dwl build script + session startup command
voice_shell/src/services/  # Python service manager + health checks
voice_shell/src/appstore/  # local model catalog + TOFU-checksummed downloader
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
- Floating HUD module ships with text/floating/both modes and unit tests.
- Orchestrator health gates block listening when required engines are down.
- Opt-in compositor build script and systemd units exist and are documented
  (kept out of the default `jarvos.target`).
- A local model app store CLI (`main.py models`) can list, check status, and
  download catalog entries with checksum pinning, covered by unit tests.

---

## 8. Security notes

- No engine ports on non-loopback interfaces.
- Service manager never invokes a shell for command assembly.
- Future sandboxing (bubblewrap/firejail) wraps optional code-exec tools, not the core engines initially.
