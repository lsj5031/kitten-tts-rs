#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
# - KITTEN_TTS_BIN: path to kitten-tts binary (default: kitten-tts from PATH)
# - ORT_LIB: explicit ONNX Runtime dylib path
# - CUDA_LIB_DIR: CUDA runtime library directory for preload
# - CUDNN_LIB_DIR: cuDNN runtime library directory for preload

KITTEN_TTS_BIN="${KITTEN_TTS_BIN:-kitten-tts}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $(basename "$0") <text...>"
  exit 1
fi

cmd=(
  "$KITTEN_TTS_BIN"
  play
  --phonemizer
  espeak-ng
  --text
  "$*"
)

if [ -n "${ORT_LIB:-}" ]; then
  cmd+=(--ort-lib "$ORT_LIB")
fi
if [ -n "${CUDA_LIB_DIR:-}" ]; then
  cmd+=(--cuda-lib-dir "$CUDA_LIB_DIR")
fi
if [ -n "${CUDNN_LIB_DIR:-}" ]; then
  cmd+=(--cudnn-lib-dir "$CUDNN_LIB_DIR")
fi

exec "${cmd[@]}"
