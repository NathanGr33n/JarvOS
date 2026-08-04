#!/usr/bin/env bash
# Stop JarvOS user target if installed.

set -u
if systemctl --user status jarvos.target >/dev/null 2>&1; then
  systemctl --user stop jarvos.target
  echo "Stopped jarvos.target"
else
  echo "jarvos.target is not active (or user systemd unavailable)."
fi
