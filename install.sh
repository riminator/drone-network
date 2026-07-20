#!/usr/bin/env bash
# install.sh — macOS / Linux convenience launcher
# Usage:
#   chmod +x install.sh && ./install.sh
#   ./install.sh --check          # verify only
#   ./install.sh --skip-torch     # skip PyTorch
set -e

# Prefer python3, fall back to python
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "[ERROR] Python not found. Install from https://python.org"
  exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"
$PYTHON install.py "$@"
