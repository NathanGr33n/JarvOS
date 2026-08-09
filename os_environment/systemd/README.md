# JarvOS systemd units

These are **templates**. Adjust `WorkingDirectory`, binary paths, and model paths for your machine before enabling.

## Install (user systemd)

```bash
./os_environment/scripts/install_user_units.sh
systemctl --user daemon-reload
systemctl --user enable --now jarvos.target
systemctl --user status jarvos.target
```

## Required environment / paths

| Variable | Purpose |
|----------|---------|
| `JARVOS_WHISPER_BIN` | whisper-server binary |
| `JARVOS_WHISPER_MODEL` | ggml model path |
| `JARVOS_LLAMA_BIN` | llama-server binary |
| `JARVOS_LLAMA_MODEL` | GGUF model path |
| `JARVOS_CONFIG` | voice_shell config.yaml |

Engines must listen on `127.0.0.1` only.

## Optional: compositor (opt-in)

`jarvos-compositor.service` + `jarvos-compositor.target` are **not** part of
`jarvos.target` and are never enabled automatically, since enabling them
changes your graphical session. They wrap [dwl](https://codeberg.org/dwl/dwl)
(wlroots-based, already supports `wlr-layer-shell`, which is what
`voice_shell/src/hud/floating.py` needs to render its overlay). No source
patches are applied; stock dwl is used as-is.

```bash
./os_environment/compositor/build.sh   # clones + builds dwl v0.8; prints missing
                                        # -dev packages instead of installing them
./os_environment/scripts/install_user_units.sh
systemctl --user daemon-reload
systemctl --user enable --now jarvos-compositor.target
```

The default unit nests dwl inside your existing graphical session
(`After=graphical-session.target`) for safe local testing — it runs as an
ordinary window, not a display-server takeover. For standalone TTY/KMS
operation on dedicated hardware (e.g. a Raspberry Pi with no other desktop),
remove the `After=graphical-session.target` line and enable the unit from a
VT instead; see dwl's own README for VT/seat requirements (`seatd` or
`elogind`/`systemd-logind`).

`dwl`'s `-s` startup command is `os_environment/compositor/session.sh`, which
launches the voice shell (`voice_shell/main.py`) as the session's app; the
floating HUD then renders as a layer-shell overlay automatically.
