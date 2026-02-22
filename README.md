# kitten-tts-rs

Fast Rust CLI for KittenTTS ONNX models.

## Why This CLI

- Pure Rust inference with ONNX Runtime
- First-run model download and local cache
- Multi-model support with presets or custom Hugging Face repo
- WAV output, raw stream output, and direct speaker playback
- Voice alias support (`Leo`, `Bella`, etc.)
- Style controls for tone/prosody variation

## Defaults

- Default model: `KittenML/kitten-tts-nano-0.8-fp32`
- Default voice: random canonical voice (unless `--voice` is set)
- Default style: random style row (unless `--style-index` is set)
- Sample rate: `24000`
- Output: `output.wav` for `synthesize`
- Playback gain (`play`): `2.5x` with anti-clipping limiter enabled
- GPU runtime: CUDA execution provider is required; startup fails if CUDA is unavailable
- ONNX Runtime is loaded dynamically from:
  - `--ort-lib /path/to/libonnxruntime.so` (preferred), or
  - `ORT_DYLIB_PATH=/path/to/libonnxruntime.so`, or
  - default loader path (`libonnxruntime.so`)
- Phonemizer mode: `espeak-ng` (default)
  - `auto` tries `espeak-ng`, then `espeak`
- Randomness control: `--seed <u64>` makes voice/style selection reproducible

## CUDA / ORT Setup

This CLI is GPU-only. You must provide a CUDA-enabled ONNX Runtime shared library and matching CUDA/cuDNN runtime libs.

Example:

```bash
kitten-tts play \
  --text "hello" \
  --ort-lib /path/to/libonnxruntime.so \
  --cuda-lib-dir /path/to/cuda/lib64 \
  --cudnn-lib-dir /path/to/cudnn/lib
```

## Install

Build locally:

```bash
cargo build --release
```

Optional local install:

```bash
mkdir -p ~/.local/bin
install -m 755 target/release/kitten-tts ~/.local/bin/kitten-tts
```

## Quick Start

Examples below assume `libonnxruntime` is discoverable via `--ort-lib` or `ORT_DYLIB_PATH`.

List models:

```bash
kitten-tts models list
```

Fetch model cache:

```bash
kitten-tts model fetch --model nano-0.8-fp32
```

List voices:

```bash
kitten-tts voices
```

Play directly on speakers:

```bash
kitten-tts play \
  --text "Hello from kitten-tts-rs" \
  --phonemizer espeak-ng
```

Optional helper script (repo example):

```bash
chmod +x examples/kitten-say.sh
ORT_LIB=/path/to/libonnxruntime.so \
CUDA_LIB_DIR=/path/to/cuda/lib \
CUDNN_LIB_DIR=/path/to/cudnn/lib \
examples/kitten-say.sh "hello world"
```

Optional make target:

```bash
make say \
  SAY_TEXT="hello world" \
  ORT_LIB=/path/to/libonnxruntime.so \
  CUDA_LIB_DIR=/path/to/cuda/lib \
  CUDNN_LIB_DIR=/path/to/cudnn/lib
```

## Usage

Synthesize WAV:

```bash
kitten-tts synthesize \
  --text "Hello from Rust KittenTTS." \
  --output hello.wav
```

Read text from file:

```bash
kitten-tts synthesize \
  --text-file README.md \
  --output readme.wav
```

Stream raw PCM (`f32le`, mono, 24kHz):

```bash
kitten-tts stream \
  --text "Streaming test sentence." \
  --phonemizer espeak-ng \
  --trim-tail 0 > stream.f32
```

Play streamed output with `ffplay`:

```bash
kitten-tts stream \
  --text "Live stream playback." \
  --phonemizer espeak-ng \
  --trim-tail 0 \
| ffplay -autoexit -nodisp -loglevel error -f f32le -ar 24000 -ch_layout mono -
```

## Voice and Tone Control

Pick voice:

```bash
kitten-tts play --text "Voice test" --voice Bella --phonemizer espeak-ng
```

Tune style/tone via style embedding index:

```bash
kitten-tts play \
  --text "Style test" \
  --voice Bella \
  --style-index 320 \
  --phonemizer espeak-ng
```

Notes:

- Omit `--voice` to pick a random canonical voice.
- Omit `--style-index` to pick a random style row.
- `--seed` makes random voice/style choices reproducible.
- `--gain` adjusts `play` loudness (`2.5` default).
- `--allow-clipping` disables limiter and applies requested gain directly.
- `--voice` changes speaker identity.
- `--style-index` changes tone/prosody flavor within that voice.
- `--speed` changes speaking rate.

## Model Selection

Preset choices:

- `nano-0.8-fp32` (default)
- `nano-0.8-int8`
- `micro-0.8`
- `mini-0.8`

Use preset:

```bash
kitten-tts synthesize --model mini-0.8 --text "Model test" --output mini.wav
```

Use custom repo:

```bash
kitten-tts synthesize \
  --repo-id KittenML/kitten-tts-nano-0.8-int8 \
  --text "Custom repo test" \
  --output custom.wav
```

`--repo-id` overrides `--model`.
