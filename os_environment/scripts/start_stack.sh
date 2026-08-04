#!/usr/bin/env bash
# Development helper: validate config and list service definitions.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/voice_shell"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 - <<"PY"
from pathlib import Path
import sys
sys.path.insert(0, str(Path(".").resolve().parent))
from voice_shell.src.config import Config
from voice_shell.src.services import ServiceManager

cfg_path = Path("config.yaml")
cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()
mgr = ServiceManager.from_config(cfg)
print("Service definitions:")
for name, svc in mgr.services.items():
    print(f"  - {name}: {" ".join(svc.command)} (health={svc.health_url or "n/a"})")
print()
print("Dev note: start whisper/llama externally or via systemd, then:")
print("  python3 main.py --config config.yaml")
PY
