#!/usr/bin/env bash
# dwl "-s" startup command: launches the JarvOS voice shell inside the
# compositor session. voice_shell/src/hud/floating.py auto-detects
# wlr-layer-shell support at runtime and renders its overlay accordingly;
# no compositor-side changes are needed for the HUD to appear.
#
# dwl writes status information to this script's stdin; since we don't
# consume it, stdin is explicitly closed to avoid blocking dwl (see
# os_environment/systemd/README.md and the dwl README's "-s" section).

set -u
exec <&-

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${JARVOS_CONFIG:-${REPO_ROOT}/voice_shell/config.yaml}"

cd "${REPO_ROOT}/voice_shell" || exit 1
exec python3 main.py --config "${CONFIG}"
