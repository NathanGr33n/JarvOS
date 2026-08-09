#!/usr/bin/env bash
# Build the optional JarvOS compositor: stock dwl (wlroots-based, layer-shell
# capable) pinned to release v0.8. This is a dev/opt-in helper; nothing here
# is invoked automatically by the rest of JarvOS.
#
# dwl already supports wlr-layer-shell, which is what
# voice_shell/src/hud/floating.py needs to render its overlay, so no
# source patches are required or applied here.
#
# This script never uses sudo. If required system packages are missing, it
# prints what to install and exits instead of attempting to install them.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/os_environment/compositor/build"
DWL_REF="v0.8"
DWL_URL="https://codeberg.org/dwl/dwl.git"

missing=()
for cmd in git make cc pkg-config; do
  command -v "${cmd}" >/dev/null 2>&1 || missing+=("${cmd}")
done
if ! pkg-config --exists wlroots-0.19 2>/dev/null; then
  missing+=("wlroots-0.19 (pkg-config module; install libwlroots-dev / wlroots)")
fi
for mod in wayland-server xkbcommon libinput; do
  pkg-config --exists "${mod}" 2>/dev/null || missing+=("${mod} (pkg-config module)")
done

if [ "${#missing[@]}" -ne 0 ]; then
  echo "Missing build prerequisites for the JarvOS compositor:"
  for item in "${missing[@]}"; do
    echo "  - ${item}"
  done
  echo
  echo "Install the -dev packages for your distro, e.g. on Debian/Ubuntu:"
  echo "  sudo apt install build-essential pkg-config libwlroots-dev \\"
  echo "    wayland-protocols libxkbcommon-dev libinput-dev"
  echo "Then re-run this script."
  exit 1
fi

mkdir -p "${BUILD_DIR}"
if [ -d "${BUILD_DIR}/dwl/.git" ]; then
  echo "Updating existing dwl checkout at ${BUILD_DIR}/dwl"
  git -C "${BUILD_DIR}/dwl" fetch --tags origin
else
  echo "Cloning dwl (${DWL_REF}) into ${BUILD_DIR}/dwl"
  git clone "${DWL_URL}" "${BUILD_DIR}/dwl"
fi
git -C "${BUILD_DIR}/dwl" checkout "${DWL_REF}"

echo "Building dwl with stock upstream config.def.h (layer-shell support included by default)..."
make -C "${BUILD_DIR}/dwl"

echo
echo "Build complete: ${BUILD_DIR}/dwl/dwl"
echo "See os_environment/systemd/jarvos-compositor.service for how to run it,"
echo "and os_environment/compositor/session.sh for the voice-shell startup command."
