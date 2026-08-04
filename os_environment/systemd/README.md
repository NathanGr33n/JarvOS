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
