#!/bin/bash
# Double-click this file in Finder to start the app.
cd "$(dirname "$0")" || exit 1

echo "Starting clonecut…"
echo

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  ! ffmpeg is not installed."
  echo "    Install it with:  brew install ffmpeg"
  echo "    Then double-click this file again."
  echo
fi

if ! command -v uv >/dev/null 2>&1; then
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [ -x "$candidate" ] && { export PATH="$(dirname "$candidate"):$PATH"; break; }
  done
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "  ! uv is not installed. Install it with:"
  echo "      curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "    then close this window and double-click again."
  echo
  read -r -p "Press Return to close."
  exit 1
fi

uv run app.py
echo
read -r -p "clonecut has stopped. Press Return to close this window."
