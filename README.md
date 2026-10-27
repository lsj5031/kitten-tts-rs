# kitten-tts-rs

Local text-to-speech in Rust, powered by the native [audio.cpp](https://github.com/0xShug0/audio.cpp) engine. Write WAV files, stream PCM, or play speech. Mini 0.8 is the default on every platform; its ~288 MiB GGUF downloads from Hugging Face on first use.

## Quick start

Build with a current Rust toolchain:

```bash
git clone https://github.com/lsj5031/kitten-tts-rs.git
cd kitten-tts-rs
cargo build --release
```

Use `target/release/kitten-tts` below, or put that executable on your `PATH`. On Windows it is `target\release\kitten-tts.exe`. Build from source for the native-only CLI. Old ONNX release binaries have been removed. GitHub-hosted CI checks compilation, tests and basic CLI commands on Linux, macOS and Windows; it does not publish binary artifacts or verify GPU synthesis. Future binary releases must be built and runtime-tested on suitable machines with the required audio.cpp backend.

You also need:

- **audio.cpp C API library**, v0.9.2 tested, ABI major 0. Get a matching `audio-<tag>-lib-<platform>-<backend>` archive from [audio.cpp releases](https://github.com/0xShug0/audio.cpp/releases), not a `-bin-` archive. Keep its companion libraries together. Set `AUDIOCPP_LIB` to `libaudiocpp.dylib` (macOS), `libaudiocpp.so` (Linux), or `audiocpp.dll` (Windows), or use `--native-lib PATH`. For CUDA builds, follow upstream's matching runtime-library instructions.
- **Mini 0.8 only:** `libespeak-ng` and `espeak-ng-data`, not just an `espeak` executable. On macOS: `brew install espeak-ng`. On Debian/Ubuntu: `sudo apt install espeak-ng libespeak-ng1`. Common Homebrew/Linux locations are detected; otherwise pass `--native-espeak-lib PATH --native-espeak-data DIR`.
- **Playback only:** `ffplay` or `pw-play`. WAV and stream output need neither.

These libraries, eSpeak data and model weights are not bundled. To build audio.cpp yourself, see [native setup](docs/native-usage.md).

```bash
export AUDIOCPP_LIB=/absolute/path/to/libs/libaudiocpp.dylib
kitten-tts native fetch       # optional prefetch, reused offline
kitten-tts native doctor      # checks library, eSpeak and model; downloads nothing
kitten-tts synthesize --text "Hello from KittenTTS." --output hello.wav
```

`doctor` reports a missing model until you fetch it or supply `--native-model`.

## KittenTTS 2

KittenTTS 2 requires **your local GGUF** and explicit acceptance of the **Stellon Labs Community License**, including its commercial registration and revenue limits. Review the [weight license and models](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/KittenTTS2-GGUF) before use. It is never automatically downloaded.

```bash
kitten-tts synthesize \
  --native-family kitten_tts2 \
  --native-model /absolute/path/to/kitten-tts2-q8_0.gguf \
  --accept-kitten2-license \
  --text "Hello from KittenTTS 2." --output kitten2.wav
```

KittenTTS 2 uses its model's default voice when omitted. `--speed` is not implemented by this port and prints a notice when changed.

## Common usage

```bash
kitten-tts synthesize --text-file article.txt --output article.wav
echo "Read standard input." | kitten-tts synthesize --output stdin.wav
kitten-tts voices
kitten-tts synthesize --text "Hello." --voice Leo --seed 7 --speed 1.1
kitten-tts stream --text "Raw audio." > speech.f32
kitten-tts play --text "Speak aloud."
kitten-tts models list
```

- WAV defaults to mono 24 kHz PCM16; `--wav-encoding float32` preserves float samples.
- `stream` writes mono 24 kHz little-endian float32 PCM to stdout, one completed text chunk at a time. Diagnostics go to stderr.
- Mini voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo. Omit `--voice` for a random choice. `--seed` makes that choice repeatable and sets decoder noise; exact audio reproducibility depends on the backend.
- `--native-backend auto` tries available accelerators, then CPU. Silent or non-finite accelerated output retries once on CPU; invalid audio is rejected. Use `--native-backend cpu` to force CPU, or select `metal`, `cuda`, `vulkan`, `hip`. Availability depends on your audio.cpp build.
- `--native-threads N` defaults to 1. `play` defaults to gain 2.5 with an anti-clipping limiter.
- Speed must be 0.5–2.0, `--max-chars` 1–2000, normalized input at most 20,000 bytes, and chunk count at most 256.

See `kitten-tts synthesize --help`, `kitten-tts native --help`, and [native reference](docs/native-usage.md) for setup and cache controls.

## Compatibility and verification

This is a breaking native-only change. ONNX inference, Nano/Micro presets, `model fetch/info`, `--repo-id`, `--ort-lib`, ORT/CUDA/cuDNN tuning flags, `--phonemizer`, `--style-index` and `--trim-tail` are removed. Use `native fetch`, Mini 0.8, or a local GGUF instead. `--engine native` and `--engine auto` both select native; `--engine onnx` is an error. Native accelerator choices, including CUDA, remain supported.

End-to-end synthesis is verified on **macOS arm64**, using audio.cpp 0.9.2 with Mini 0.8 and KittenTTS 2 Q8. CPU works for both; Kitten2 Metal passed. Mini Metal can return invalid audio: the latest auto-backend smoke runs successfully retried on CPU. Use `--native-backend cpu` for predictable Mini execution. Linux, Windows, CUDA, Vulkan and HIP are not runtime-verified here. No listening-quality claim is made. [Historical evaluation and measurements](docs/native-backend-evaluation.md) describe the earlier dual-engine implementation.
