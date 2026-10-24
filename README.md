# kitten-tts-rs

Fast Rust CLI for KittenTTS ONNX models.

## Why This CLI

- Pure Rust inference with ONNX Runtime, plus an optional native audio.cpp engine
- First-run model download and local cache
- Multi-model support with presets or custom Hugging Face repo
- WAV output, raw stream output, and direct speaker playback
- Voice alias support (`Leo`, `Bella`, etc.)
- Style controls for tone/prosody variation

## Defaults

- Default model: `KittenML/kitten-tts-nano-0.8-fp32`
- Default engine: `auto` (native only when it can serve the request, else ONNX Runtime)
- Default voice: random canonical voice (unless `--voice` is set)
- Default style: random style row (unless `--style-index` is set)
- Sample rate: `24000`
- Output: `output.wav` for `synthesize`
- Playback gain (`play`): `2.5x` with anti-clipping limiter enabled
- ONNX path runtime: CUDA execution provider is required; startup fails if CUDA is unavailable
- CUDA session defaults: arena strategy `same-as-requested`, conv algo `heuristic`, graph opt level 1, memory pattern off, parallel execution off
- VRAM profiles (`--vram-profile`): `minimal` / `balanced` / `performance` — curated presets that override individual session flags
- ONNX Runtime is loaded dynamically from:
  - `--ort-lib /path/to/libonnxruntime.so` (preferred), or
  - `ORT_DYLIB_PATH=/path/to/libonnxruntime.so`, or
  - default loader path (`libonnxruntime.so`)
- Phonemizer mode: `espeak-ng` (default)
  - `auto` tries `espeak-ng`, then `espeak`
- Randomness control: `--seed <u64>` makes voice/style selection reproducible

## CUDA / ORT Setup

The ONNX Runtime path is GPU-only. You must provide a CUDA-enabled ONNX Runtime shared library and matching CUDA/cuDNN runtime libs. The optional native engine does not use ONNX Runtime.

Example:

```bash
kitten-tts play \
  --text "hello" \
  --ort-lib /path/to/libonnxruntime.so \
  --cuda-lib-dir /path/to/cuda/lib64 \
  --cudnn-lib-dir /path/to/cudnn/lib
```

## Install

### Pre-built Releases

Linux/macOS/Windows releases are available on the [GitHub Releases](https://github.com/lsj5031/kitten-tts-rs/releases) page.

Linux releases include bundled CUDA 12.2 runtime libraries (libcublasLt.so.12, libcublas.so.12, libcudart.so.12, libcufft.so.11, libcurand.so.11, libnvrtc.so.12) alongside the ONNX Runtime GPU build, so no separate CUDA installation is needed when using the binary.

### Build from Source

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

## Native Engine (optional)

Synthesis can also run through the [audio.cpp](https://github.com/0xShug0/audio.cpp)
C API instead of ONNX Runtime. It is opt-in and leaves the ONNX defaults alone.
The evaluation, decision and measurements live in
[docs/native-backend-evaluation.md](docs/native-backend-evaluation.md).

Requirements:

- audio.cpp C API library v0.9.2 or newer (`--native-lib` or `AUDIOCPP_LIB`)
- `libespeak-ng` for the KittenTTS 0.8 frontend (`--native-espeak-lib` and
  `--native-espeak-data`, or auto-detected under Homebrew/system paths)
- KittenTTS Mini 0.8 GGUF, downloaded on first use (~288 MiB) into the cache, or
  passed with `--native-model /path/to.gguf`. To download it ahead of time:
  `kitten-tts native fetch` (add `--cache-dir` or `--force` as needed).

```bash
kitten-tts synthesize \
  --text "Hello from the native engine" \
  --engine native \
  --model mini-0.8 \
  --native-lib /path/to/libaudiocpp.dylib \
  --native-threads 8 \
  --native-espeak-lib /opt/homebrew/lib/libespeak-ng.dylib \
  --native-espeak-data /opt/homebrew/share/espeak-ng-data \
  --output native.wav
```

`--engine auto` is the default. It selects the native engine only when it can
serve the request: `--model mini-0.8`, an explicit `--native-model`, or
`--native-family kitten_tts2`. The default ONNX preset `nano-0.8-fp32` keeps
using ONNX Runtime. `--engine native` always uses the native engine.

Native notes:

- `--native-backend auto` (the default) tries accelerators before `cpu`:
  `metal` on macOS arm64, and `cuda`, `vulkan`, `hip` on Linux/Windows (session
  creation falls through the list). If the accelerated run returns silent or
  non-finite audio, the CLI prints a notice and retries on `cpu` before failing.
  Force one backend with `--native-backend cpu|metal|cuda|vulkan|hip`.
- Measured on Apple M5 with 8 threads and the Mini 0.8 FP32 GGUF: Metal 0.79 s
  vs CPU 1.71 s for 6.59 s of audio (RTF 0.12 vs 0.26). Metal is intermittently
  unusable on this model: two observed runs among about a dozen returned silent
  or non-finite samples, and `auto` then retried on CPU and completed. Expect the
  retry (~1.5 s extra here) occasionally; forcing `--native-backend cpu`
  avoids it.
- For `kitten_tts2`, `auto` also selects Metal on macOS (2.92 s vs 4.9 s for
  3.14 s of audio at 8 threads). Backends do not produce identical samples: the
  same seed gave 75,360 samples on Metal vs 68,640 on CPU, so determinism holds
  per backend.
- `--style-index`, `--vram-profile`, and `--repo-id` are ONNX-only and rejected
  on the native path.
- `--trim-tail` and `--phonemizer` are ONNX-only. The native 0.8 frontend uses
  its own eSpeak-ng, and the port trims its own output.
- `--speed` maps to the native speaking rate for `kitten_tts` (verified against
  upstream); `kitten_tts2` ignores it and prints a notice. `--seed` maps to the
  native request seed. Streaming is per text chunk; these ports have no native
  streaming session.
- Native `kitten_tts` voices: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`,
  `Kiki`, `Leo`. Omit `--voice` for a random one. For `kitten_tts2`, pass a preset
  from the package (`Bruno`, `German`, ...) or omit `--voice` for the engine
  default.

### KittenTTS 2 (opt-in)

KittenTTS 2 runs only through the native engine, with a local GGUF, and requires
acknowledging the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md)
(commercial registration and revenue/funding limits):

```bash
kitten-tts synthesize \
  --text "Hello from KittenTTS 2" \
  --engine native \
  --native-family kitten_tts2 \
  --native-model /path/to/kitten-tts2.gguf \
  --accept-kitten2-license \
  --native-lib /path/to/libaudiocpp.dylib \
  --output v2.wav
```

v2 weights are not downloaded automatically and are not bundled. Validated here
with the 2026-10-21 Q8 GGUF (3.34 GB) on macOS arm64 CPU and Metal: English and
German presets, streaming, multi-chunk text and seeded determinism all work.
Metal is faster (2.92 s vs 4.9 s for 3.14 s of audio at 8 threads), and the same
seed produces different sample counts across backends, so treat determinism as
per-backend. Metal on v2 has not shown the invalid-output behaviour seen on
0.8; `auto` still validates and retries on CPU. Not validated: CUDA, cloning, the remaining language presets, or
listening quality. The port ignores `--speed`; the CLI prints a notice when it is
set.

### Native model download

`native fetch` downloads the native model without synthesizing, so the weights
can be prepared ahead of time or on another machine:

```bash
kitten-tts native fetch                          # into the platform cache
kitten-tts native fetch --cache-dir ./models     # into a specific directory
kitten-tts native fetch --force                  # re-download even if cached
```

It prints the resolved path, byte size and source URL, and needs no ONNX Runtime.
`native fetch --family kitten_tts2` exits with an explanation instead of
downloading, because v2 weights require a manual download and license
acknowledgement. `kitten-tts models list` also prints the native download hint.

### Native probe

Inspect a native library without synthesizing:

```bash
kitten-tts native probe --native-lib /path/to/libaudiocpp.dylib
```

The probe prints the audio.cpp build version, ABI version, registered model
families and task vocabulary. It exits non-zero when the ABI major does not match,
when a required symbol is missing, or when `--family` names an absent family. The
library path can also come from `AUDIOCPP_LIB`, then falls back to the platform
default name (`libaudiocpp.so`, `libaudiocpp.dylib`, `audiocpp.dll`).

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

## VRAM / Session Tuning

Fine-grained control over ONNX Runtime session parameters for tuning GPU performance vs. VRAM usage.

### Quick Presets

Use `--vram-profile` to apply curated presets (overrides individual flags):

```bash
# Lowest VRAM usage
kitten-tts play --text "hello" --vram-profile minimal

# Default balanced behavior (same as omitting the flag)
kitten-tts play --text "hello" --vram-profile balanced

# Maximum performance (uses more VRAM)
kitten-tts play --text "hello" --vram-profile performance
```

Profile details:

| Setting | `minimal` | `balanced` | `performance` |
|---|---|---|---|
| Arena strategy | same-as-requested | *(individual flag)* | next-power-of-two |
| Conv algo search | heuristic | *(individual flag)* | exhaustive |
| Graph optimization | Level 1 | *(individual flag)* | Level 3 |
| Memory pattern | off | *(individual flag)* | on |
| Parallel execution | off | *(individual flag)* | on |

### Individual Flags

When `--vram-profile` is not set (or set to `balanced`), individual flags pass through:

```bash
kitten-tts play \
  --text "hello" \
  --arena-strategy next-power-of-two \
  --conv-algo exhaustive \
  --graph-opt 3 \
  --memory-pattern \
  --parallel-execution
```

Available flags:

- `--arena-strategy <same-as-requested|next-power-of-two>` — Memory arena extend strategy
- `--conv-algo <heuristic|default|exhaustive>` — cuDNN convolution algorithm search
- `--graph-opt <1|3>` — ONNX graph optimization level (1 = basic, 3 = full)
- `--memory-pattern` — Enable memory pattern planning (pre-allocate buffers)
- `--parallel-execution` — Enable parallel execution (concurrent kernels)

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
