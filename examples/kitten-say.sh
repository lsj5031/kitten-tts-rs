#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
# - KITTEN_TTS_BIN: path to kitten-tts binary (default: kitten-tts from PATH)
# - AUDIOCPP_LIB: path to the audio.cpp C API library

KITTEN_TTS_BIN="${KITTEN_TTS_BIN:-kitten-tts}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $(basename "$0") <text...>"
  exit 1
fi

cmd=(
  "$KITTEN_TTS_BIN"
  play
  --text
  "$*"
)

exec "${cmd[@]}"
