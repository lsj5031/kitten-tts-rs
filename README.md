# kitten-tts-rs

`kitten-tts` is a Rust command-line tool for local KittenTTS text-to-speech. It writes WAV files (`synthesize`), plays audio on your speakers (`play`), or streams raw 24 kHz mono PCM (`stream`) from `--text`, `--text-file`, or piped `stdin`. Model weights are downloaded from Hugging Face on first use and cached locally.

It supports two inference engines (`--engine auto` by default):

- **Native `audio.cpp` engine (`native`)**: Runs GGUF models in-process through the [`audio.cpp`](https://github.com/0xShug0/audio.cpp) C API (`libaudiocpp`). Used by default on macOS with KittenTTS Mini 0.8, and available on any platform for Mini 0.8 or KittenTTS 2 without ONNX Runtime or an NVIDIA GPU.
- **ONNX Runtime engine (`onnx`)**: Runs KittenTTS 0.8 ONNX models (`nano-0.8-fp32` by default, `nano-0.8-int8`, `micro-0.8`, `mini-0.8`, or a custom `--repo-id`). **Requires an NVIDIA GPU with CUDA** (no CPU fallback) and supports fine-grained style embedding selection (`--style-index`).

## Choose Your Engine & Prerequisites

| Platform / Hardware | Recommended Engine | Required Dependencies |
|---|---|---|
| **macOS (Apple Silicon)** | **Native (`audio.cpp`)** *(selected by default)* | 1. `audio.cpp` C API library v0.9.2+ (`libaudiocpp.dylib`, ABI major 0) via `AUDIOCPP_LIB` or `--native-lib`<br>2. `espeak-ng` shared library + data (`brew install espeak-ng`) for KittenTTS 0.8<br>3. Optional for `play`: `ffplay` (`brew install ffmpeg`) or `pw-play` |
| **Linux / Windows with NVIDIA GPU** | **ONNX Runtime (CUDA)** *(selected by default)* or **Native (`audio.cpp`)** | **ONNX path**:<br>1. NVIDIA GPU driver + CUDA-enabled `libonnxruntime` + CUDA 12 & cuDNN 9 runtime libraries (bundled in Linux release archives)<br>2. `espeak-ng` (or `espeak`) binary on `PATH`<br>3. Optional for `play`: `ffplay` or `pw-play` |
| **Linux / Windows without CUDA** | **Native (`audio.cpp`)** *(pass `--engine native` or `--model mini-0.8`)* | 1. `audio.cpp` C API library v0.9.2+ (`libaudiocpp.so` / `audiocpp.dll`)<br>2. `libespeak-ng` + `espeak-ng-data` for KittenTTS 0.8<br>3. Optional for `play`: `ffplay` or `pw-play` |

**Verification status**: The native `audio.cpp` engine has been verified end-to-end on macOS arm64 (CPU and Metal) with KittenTTS Mini 0.8 and KittenTTS 2 Q8. Native Linux, Windows, and GPU backends (`cuda`, `vulkan`, `hip`) are wired to `audio.cpp` but have not been tested in this repository yet. See [`docs/native-backend-evaluation.md`](docs/native-backend-evaluation.md) for measurements and parity details.

## Install

### Build from Source

Building from `main` is recommended, especially on macOS or when using the native `audio.cpp` engine (tagged releases up to `v0.0.4` predate native engine support):

Install a current Rust toolchain with [rustup](https://rustup.rs/) first, then:

```bash
git clone https://github.com/lsj5031/kitten-tts-rs.git
cd kitten-tts-rs
cargo build --release
```

Install the executable so the commands below are available in your shell:

```bash
mkdir -p ~/.local/bin
install -m 755 target/release/kitten-tts ~/.local/bin/kitten-tts
export PATH="$HOME/.local/bin:$PATH"
```

These are macOS/Linux shell commands. On Windows, use `target\release\kitten-tts.exe` directly or copy it into a directory on your `PATH`.

### Pre-built Releases

Archives for Linux, macOS, and Windows are published on [GitHub Releases](https://github.com/lsj5031/kitten-tts-rs/releases).

What the release archives actually include:

- **Not bundled on any platform**: `libaudiocpp` (`audio.cpp`), `espeak-ng` / `libespeak-ng`, and model weights are never bundled.
- **Linux (`x86_64-unknown-linux-gnu.tar.gz`)**: Includes `kitten-tts`, ONNX Runtime GPU 1.23.0 (`libonnxruntime*.so*`), CUDA 12.8 runtime libraries (`libcudart`, `libcublas`, `libcublasLt`, `libcufft`, `libcurand`, `libnvrtc`, `libnvrtc-builtins`), and cuDNN 9 (`libcudnn*.so.9`). Keep the `.so` files in the same directory as `kitten-tts`; you only need a compatible NVIDIA driver and `espeak-ng` on `PATH`.
- **Windows (`x86_64-pc-windows-msvc.zip`)**: Includes `kitten-tts.exe` and ONNX Runtime GPU 1.23.0 DLLs (`onnxruntime*.dll`). CUDA 12 and cuDNN runtime DLLs must be installed separately for the ONNX path.
- **macOS (`aarch64-apple-darwin.tar.gz`)**: Includes `kitten-tts` and `libonnxruntime*.dylib`. Because the ONNX path in `kitten-tts` requires CUDA, ONNX synthesis cannot run on macOS; use the native `audio.cpp` engine instead.

## Quick Start

### Path A: macOS or Native `audio.cpp` Engine

1. **Install `espeak-ng`** (and `ffmpeg` if you want `kitten-tts play`):

   ```bash
   # macOS (Homebrew)
   brew install espeak-ng ffmpeg

   # Ubuntu / Debian
   sudo apt install espeak-ng libespeak-ng1 ffmpeg
   ```

2. **Get the `audio.cpp` C API library (`libaudiocpp`, v0.9.2+, ABI major 0)**:

   - **Option 1: Download a prebuilt `lib` archive from [`audio.cpp` Releases](https://github.com/0xShug0/audio.cpp/releases)** (pick `audio-<tag>-lib-<platform>-<backend>`, not `-bin-`). On macOS Apple Silicon:

     ```bash
     curl -LO https://github.com/0xShug0/audio.cpp/releases/download/v0.9.2/audio-v0.9.2-lib-macos-arm64-metal.tar.gz
     mkdir -p audiocpp-lib && tar -xzf audio-v0.9.2-lib-macos-arm64-metal.tar.gz -C audiocpp-lib
     export AUDIOCPP_LIB="$PWD/audiocpp-lib/libs/libaudiocpp.dylib"
     ```

     Keep the companion shared libraries inside `libs/` next to `libaudiocpp`. (For Linux/Windows CUDA builds of `audio.cpp`, also extract the matching `audio-<tag>-cudart-...` archive into `libs/`.)

   - **Option 2: Build `libaudiocpp` from source**:

     ```bash
     git clone https://github.com/0xShug0/audio.cpp.git
     cd audio.cpp
     cmake -S . -B build -DAUDIOCPP_BUILD_C_API=ON
     cmake --build build --target audiocpp --config Release
     ```

     Then set `export AUDIOCPP_LIB=/path/to/libaudiocpp.dylib` (or `.so` / `.dll`), or pass `--native-lib PATH`.

3. **(Optional) Pre-fetch the Mini 0.8 GGUF (~288 MiB) and verify your setup**:

   ```bash
   kitten-tts native fetch
   kitten-tts native doctor
   ```

4. **Generate your first WAV or play speech**:

   On macOS, a plain command uses the native engine with KittenTTS Mini 0.8 automatically (downloading the ~288 MiB GGUF on first run if not already cached):

   ```bash
   kitten-tts synthesize --text "Hello from KittenTTS." --output hello.wav
   kitten-tts play --text "Hello from KittenTTS."
   ```

   On Linux or Windows, pass `--engine native` (or `--model mini-0.8`) and increase `--native-threads` as needed:

   ```bash
   kitten-tts synthesize \
     --engine native \
     --native-threads 8 \
     --text "Hello from the native engine." \
     --output hello.wav
   ```

### Path B: Linux / Windows with NVIDIA GPU (ONNX Runtime)

1. Install `espeak-ng` on `PATH` (and `ffplay` or `pw-play` for `play`).
2. If you built from source instead of using the bundled Linux release directory, point to `libonnxruntime`:

   ```bash
   export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
   ```

3. **Generate your first WAV or play speech** (downloads `KittenML/kitten-tts-nano-0.8-fp32` on first run):

   ```bash
   kitten-tts synthesize --text "Hello from KittenTTS." --output hello.wav
   kitten-tts play --text "Hello from KittenTTS."
   ```

   If your CUDA or cuDNN libraries are in non-standard locations, pass them explicitly:

   ```bash
   kitten-tts synthesize \
     --text "Hello from KittenTTS." \
     --ort-lib /path/to/libonnxruntime.so \
     --cuda-lib-dir /path/to/cuda/lib64 \
     --cudnn-lib-dir /path/to/cudnn/lib \
     --output hello.wav
   ```

## Common Usage

### Text Input (`--text`, `--text-file`, `stdin`)

`synthesize`, `play`, and `stream` accept input from a flag, a file, or standard input:

```bash
# Inline text
kitten-tts synthesize --text "Hello from Rust KittenTTS." --output hello.wav

# Text file
kitten-tts synthesize --text-file README.md --output readme.wav

# Piped stdin
echo "Hello from standard input." | kitten-tts synthesize --output stdin.wav
```

### Output Modes (`synthesize`, `play`, `stream`)

Write a 24 kHz mono WAV file (`output.wav` by default, `--wav-encoding pcm16` or `float32`):

```bash
kitten-tts synthesize --text "Write a WAV file." --output speech.wav
```

Play directly on speakers via `ffplay` or `pw-play` (`--player auto|ffplay|pw-play`, default `--gain 2.5` with anti-clipping limiter enabled; see also [`examples/kitten-say.sh`](examples/kitten-say.sh) and `make say SAY_TEXT="hello"`):

```bash
kitten-tts play --text "Play on speakers." --gain 2.5
```

Stream raw 32-bit little-endian float PCM (`f32le`, mono, 24000 Hz) to `stdout` as each text chunk completes:

```bash
kitten-tts stream --text "Streaming test sentence." > stream.f32

kitten-tts stream --text "Live stream playback." \
  | ffplay -autoexit -nodisp -loglevel error -f f32le -ar 24000 -ch_layout mono -
```

### Voices, Speed, Seed, and Style

- **Voice (`--voice`)**: Omit `--voice` to pick a random voice, or choose one by name. All eight KittenTTS 0.8 voices work on both engines:
  `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.

  ```bash
  kitten-tts synthesize --text "Speaking as Bella." --voice Bella --output bella.wav
  ```

  To inspect canonical voice keys and aliases in an ONNX model's `voices.npz`:

  ```bash
  kitten-tts voices
  ```

- **Speed (`--speed`)**: Speaking rate multiplier between `0.5` and `2.0` (default `1.0`). Supported on ONNX and native `kitten_tts` (Mini 0.8); ignored with a notice on `kitten_tts2`.
- **Seed (`--seed`)**: Makes random voice selection, ONNX style row selection, and native decoder noise reproducible:

  ```bash
  kitten-tts synthesize --text "Reproducible output." --voice Leo --seed 7 --output leo.wav
  ```

- **Style row (`--style-index`, ONNX-only)**: Selects a specific style embedding row within a voice's table in `voices.npz` (omitting it on the ONNX path picks a random row). Because the native engine does not expose style rows, passing `--style-index` requires the ONNX engine:

  ```bash
  kitten-tts synthesize --text "Style test." --voice Bella --style-index 320 --output style.wav
  ```

### Models and Caching

List all model presets and native model commands:

```bash
kitten-tts models list
```

Available `--model` presets:

- `nano-0.8-fp32` -> `KittenML/kitten-tts-nano-0.8-fp32` (default ONNX preset)
- `nano-0.8-int8` -> `KittenML/kitten-tts-nano-0.8-int8` (ONNX)
- `micro-0.8` -> `KittenML/kitten-tts-micro-0.8` (ONNX)
- `mini-0.8` -> `KittenML/kitten-tts-mini-0.8` (supported by both Native GGUF and ONNX; default on macOS)

Pre-fetch or inspect models without synthesizing:

```bash
# Native Mini 0.8 GGUF (~288 MiB, no ONNX Runtime needed)
kitten-tts native fetch
kitten-tts native fetch --cache-dir ./models --force

# ONNX model artifacts (config.json, .onnx, voices.npz)
kitten-tts model fetch --model nano-0.8-fp32
kitten-tts model info --model nano-0.8-fp32
```

For custom ONNX repositories on Hugging Face, pass `--repo-id` (overrides `--model`, ONNX-only):

```bash
kitten-tts synthesize \
  --repo-id KittenML/kitten-tts-nano-0.8-int8 \
  --text "Custom repo test." \
  --output custom.wav
```

Downloads are written to a temporary file and moved into the cache only after the byte count matches `Content-Length`. Native models are also verified for the `GGUF` magic header on download and on every cache hit.

## How Engine Selection Works (`--engine`)

`--engine auto` is the default:

- **On macOS**: A plain command (no `--model` and no ONNX-only flags) runs on the **native engine** with KittenTTS Mini 0.8, because the ONNX path requires CUDA. If the native library cannot be loaded, `kitten-tts` prints a note and falls back to ONNX Runtime.
- **On all platforms, `auto` selects the native engine** when you pass `--model mini-0.8` (falls back to ONNX if the native library is unavailable), or requires the native engine when you pass `--native-lib`, `--native-model`, `--native-family kitten_tts2`, `--native-espeak-lib`, or `--native-espeak-data`.
- **On all platforms, `auto` selects ONNX Runtime** on Linux/Windows by default, or whenever you pass an ONNX-only preset (`nano-0.8-fp32`, `nano-0.8-int8`, `micro-0.8`) or ONNX-only flag (`--repo-id`, `--style-index`, `--vram-profile`, `--ort-lib`, `--cuda-lib-dir`, `--cudnn-lib-dir`, `--cuda-device-id`, `--cuda-memory-limit`, or a non-default `--trim-tail` or `--phonemizer`).
- **`--engine native`** forces the native engine (rejecting `--style-index`, `--vram-profile`, `--repo-id`, and non-`mini-0.8` presets unless `--native-model` is set; `--phonemizer` and `--trim-tail` are ignored with a notice).
- **`--engine onnx`** forces the CUDA ONNX Runtime engine.

> **macOS note**: Avoid passing `--style-index`, `--trim-tail`, `--vram-profile`, or `--phonemizer auto` on macOS, as those flags tell `--engine auto` to use the CUDA-only ONNX engine.

## Advanced: Native Engine (`audio.cpp`)

### Environment Checks (`native doctor` and `native probe`)

`native doctor` checks all three native requirements in one pass (the `libaudiocpp` library and ABI/family, `libespeak-ng` + `espeak-ng-data`, and the cached or explicit GGUF model), reports every problem at once without downloading anything, and exits non-zero if any check fails:

```bash
kitten-tts native doctor --native-lib /path/to/libaudiocpp.dylib
kitten-tts native doctor --family kitten_tts2 --native-model ./kitten-tts2.gguf
kitten-tts native doctor --cache-dir ./models
```

`native probe` inspects the `audio.cpp` library alone (build version, ABI version, registered model families, and task vocabulary):

```bash
kitten-tts native probe --native-lib /path/to/libaudiocpp.dylib
```

### Backends and Worker Threads

- **`--native-backend auto|cpu|metal|cuda|vulkan|hip`** (default `auto`):
  `auto` tries hardware accelerators before `cpu` (`metal` on macOS arm64; `cuda`, `vulkan`, `hip` on Linux/Windows). If an accelerated run returns silent or non-finite audio, the CLI prints a notice and retries once on `cpu` before failing. Silent or non-finite output is never written to a WAV file.
- **macOS Metal vs. CPU**: On Apple Silicon with KittenTTS Mini 0.8, Metal is roughly 2x faster when warm, but has intermittently returned silent or non-finite samples that trigger the automatic CPU retry (~1.5 s extra on retry). Pass `--native-backend cpu` for predictable CPU-only execution.
- **`--native-threads <1..256>`** (default `1`): Sets the worker thread count for the native engine (for example, `--native-threads 8`).
- **eSpeak paths**: For `kitten_tts` (0.8), `libespeak-ng` and `espeak-ng-data` are auto-detected under standard Homebrew and Linux system paths (`/opt/homebrew`, `/usr/local`, `/usr/lib`, `/usr/share`). On Windows or custom installs, pass `--native-espeak-lib PATH` and `--native-espeak-data PATH`.

For full benchmark tables and evaluation details, see [`docs/native-backend-evaluation.md`](docs/native-backend-evaluation.md).

### KittenTTS 2 (`--native-family kitten_tts2`, opt-in)

KittenTTS 2 runs only through the native engine with a local GGUF (such as `kitten-tts2-20261021-q8_0.gguf` from [`audio-cpp/audio.cpp-gguf`](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/KittenTTS2-GGUF), ~3.34 GB for Q8) and requires acknowledging the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md) (which includes commercial registration and revenue/funding limits):

```bash
kitten-tts synthesize \
  --engine native \
  --native-family kitten_tts2 \
  --native-model /path/to/kitten-tts2-20261021-q8_0.gguf \
  --accept-kitten2-license \
  --native-lib /path/to/libaudiocpp.dylib \
  --native-threads 8 \
  --voice Bruno \
  --text "Hello from KittenTTS 2." \
  --output v2.wav
```

KittenTTS 2 notes and limitations:

- Weights are never bundled or downloaded automatically (`kitten-tts native fetch --family kitten_tts2` exits with download instructions).
- Does not use `espeak-ng`.
- Pass a preset voice from the v2 GGUF package (`Bruno`, `German`, etc.) via `--voice`, or omit `--voice` to use the model's default voice.
- The `kitten_tts2` port ignores `--speed` (the CLI prints a notice if set).
- `--seed` is read as a signed 64-bit integer (maximum `9223372036854775807`). CPU and Metal produce different sample counts for the same seed, so determinism is per-backend.
- Validated on macOS arm64 (CPU and Metal) with the 2026-10-21 Q8 GGUF for English, German, multi-chunk text, and `stream`. Native CUDA, voice cloning, other language presets, and listening quality have not been validated in this repository.

## Advanced: ONNX / CUDA Session Tuning

### VRAM Profile Presets

Use `--vram-profile` on the ONNX path to apply curated session presets (overrides individual session flags):

```bash
# Lowest VRAM usage
kitten-tts play --text "hello" --vram-profile minimal

# Default balanced behavior (passes individual flags through)
kitten-tts play --text "hello" --vram-profile balanced

# Maximum performance (uses more VRAM)
kitten-tts play --text "hello" --vram-profile performance
```

| Setting | `minimal` | `balanced` (default) | `performance` |
|---|---|---|---|
| Arena strategy | `same-as-requested` | *(individual flag)* | `next-power-of-two` |
| Conv algo search | `heuristic` | *(individual flag)* | `exhaustive` |
| Graph optimization | Level `1` | *(individual flag)* | Level `3` |
| Memory pattern | off | *(individual flag)* | on |
| Parallel execution | off | *(individual flag)* | on |

### Individual ONNX / CUDA Flags

When `--vram-profile` is omitted or set to `balanced`, individual flags configure the ONNX Runtime session:

- `--ort-lib <PATH>` (or `ORT_DYLIB_PATH`): Path to `libonnxruntime` shared library
- `--cuda-lib-dir <DIR>` / `--cudnn-lib-dir <DIR>`: Directories for CUDA 12 / cuDNN 9 runtime libraries
- `--cuda-device-id <ID>`: CUDA device index (default `0`)
- `--cuda-memory-limit <MiB>`: CUDA memory arena limit in MiB (default `2048`; verified against `nvidia-smi` free VRAM with 512 MiB headroom when `nvidia-smi` is available)
- `--arena-strategy <same-as-requested|next-power-of-two>`: Memory arena extend strategy (default `same-as-requested`)
- `--conv-algo <heuristic|default|exhaustive>`: cuDNN convolution algorithm search (default `heuristic`)
- `--graph-opt <1|3>`: ONNX graph optimization level (default `1`)
- `--memory-pattern`: Enable memory pattern planning
- `--parallel-execution`: Enable parallel execution (sets ORT inter-op threads to auto)
- `--phonemizer <espeak-ng|auto|espeak>`: External phonemizer binary on `PATH` (default `espeak-ng`; `auto` tries `espeak-ng` then `espeak`)
- `--trim-tail <SAMPLES>`: Samples trimmed from the end of each ONNX chunk (default `5000`)

## Runtime Guardrails

Enforced on both engines:

- `--speed`: `0.5` to `2.0` (default `1.0`)
- `--max-chars`: `1` to `2000` characters per chunk (default `400`)
- Input text length: up to `20,000` characters after whitespace normalization, and at most `256` chunks
- `--gain` (`play`): `0.1` to `8.0` (default `2.5`, peak-limited to `0.98` unless `--allow-clipping` is set)

## Troubleshooting

- **macOS tries to load `libonnxruntime` or fails with a CUDA error**:
  Either `libaudiocpp.dylib` could not be loaded (causing `--engine auto` to fall back to ONNX Runtime), or the command included an ONNX-only flag (`--style-index`, `--trim-tail`, `--phonemizer auto`, `--vram-profile`, `--repo-id`) or a non-`mini-0.8` `--model` preset. Run `kitten-tts native doctor` to inspect your native setup.
- **`failed to load native audio.cpp library` or ABI mismatch**:
  Ensure you downloaded the `audio-<tag>-lib-...` C API archive (v0.9.2+, ABI major `0`) rather than the `-bin-` CLI archive, keep all companion libraries in the same `libs/` directory, and set `AUDIOCPP_LIB` or pass `--native-lib PATH`. Verify with `kitten-tts native probe`.
- **`espeak lib: FAILED` / `espeak data: FAILED` or `phonemizer 'espeak-ng' not found`**:
  Install `espeak-ng` (`brew install espeak-ng` on macOS; `sudo apt install espeak-ng libespeak-ng1` on Debian/Ubuntu). For custom install paths with the native engine, pass `--native-espeak-lib` and `--native-espeak-data`.
- **Corrupt or truncated cached model**:
  Re-download the cached weights with `kitten-tts native fetch --force` (for native Mini 0.8) or `kitten-tts model fetch --force` (for ONNX presets).
- **Occasional retry delay on macOS Metal (`invalid audio; retrying with cpu`)**:
  Metal on KittenTTS Mini 0.8 intermittently produces silent or non-finite samples; `--native-backend auto` detects this and retries on CPU. Pass `--native-backend cpu --native-threads 8` to run directly on CPU.
- **`no supported audio player found` when running `play`**:
  Install `ffplay` (FFmpeg) or `pw-play` (PipeWire) on `PATH`, or use `kitten-tts synthesize --output output.wav` and play the WAV file with any audio player.
- **`insufficient GPU VRAM` on the ONNX path**:
  Lower the arena limit with `--cuda-memory-limit 1024` or pass `--vram-profile minimal`.
