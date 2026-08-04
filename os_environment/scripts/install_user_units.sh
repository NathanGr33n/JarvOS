#!/usr/bin/env bash
# Install JarvOS user systemd unit templates into ~/.config/systemd/user

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="${REPO_ROOT}/os_environment/systemd"
UNIT_DST="${HOME}/.config/systemd/user"

mkdir -p "${UNIT_DST}"

for unit in jarvos.target jarvos-whisper.service jarvos-llama.service jarvos-voice-shell.service; do
  src="${UNIT_SRC}/${unit}"
  dst="${UNIT_DST}/${unit}"
  sed "s|%h/Git/JarvOS|${REPO_ROOT}|g" "${src}" > "${dst}"
  echo "Installed ${dst}"
done

echo
echo "Next:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now jarvos.target"
