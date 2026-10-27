# AGENTS.md

Agent guide for `kitten-tts-rs`.

## Project

Single-package Rust CLI for native KittenTTS inference through the audio.cpp C API. Main implementation and unit tests live in `src/main.rs`. It downloads Mini 0.8 GGUF weights on first use, writes WAV or raw PCM, and supports playback through an external player.

This project is native-only. Do not reintroduce ONNX inference, ORT dependencies, ORT/CUDA/cuDNN tuning options, or runtime bundling. Native audio.cpp accelerator backends (including CUDA) remain supported.

## Validation

Run from the repository root:

```bash
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo build --release
git diff --check
```

Target one test with its fully qualified name:

```bash
cargo test tests::native_mini_is_the_default_on_every_platform -- --exact
```

Smoke checks:

```bash
cargo run -- --help
cargo run -- models list
cargo run -- voices
cargo run -- native doctor
cargo run -- synthesize --text "Hello from KittenTTS." --output output.wav
```

Synthesis requires a user-provided audio.cpp C API shared library (ABI major 0), Mini's libespeak-ng and espeak-ng-data, and a cached or local GGUF. Pass `--native-lib PATH` or set `AUDIOCPP_LIB`. `native doctor` downloads nothing and reports all missing prerequisites. Do not claim runtime verification when prerequisites are missing. Never play audio on the user's speakers unless asked.

Release CI builds Linux x86_64, macOS arm64 and Windows x86_64 binaries in `.github/workflows/release.yml`. Archives contain only the executable, not libraries or weights. Runtime verification so far is macOS arm64; other platforms/backends need independent testing.

## Placement and style

- Keep changes in `src/main.rs` unless scope justifies otherwise; avoid speculative abstractions and broad unrelated refactors.
- Match existing clap derives (`Parser`, `Subcommand`, `Args`, `ValueEnum`), flattened argument groups and typed defaults.
- Group explicit imports at the top; use snake_case functions, PascalCase types and SCREAMING_SNAKE_CASE constants.
- Use `anyhow::Result`, `?`, `bail!` and actionable `.context`/`.with_context` at I/O, HTTP and FFI boundaries.
- Avoid runtime `unwrap`/panics. `expect` is acceptable for static invariants and in tests.
- Keep deterministic, offline-friendly unit tests in the bottom `#[cfg(test)] mod tests`.
- Prefer early returns, focused helpers, rustfmt formatting and few comments.
- Check examples, Makefile, metadata and docs when changing commands.
- Preserve historical measurements in `docs/native-backend-evaluation.md`; current usage belongs in README and `docs/native-usage.md`.

## Runtime behavior to preserve

- Native audio.cpp is the default on every platform, with Mini 0.8 as the default packaged model. `--model mini-0.8` remains accepted.
- `--engine native` and `--engine auto` both use native. There is no ONNX fallback.
- Preserve `native doctor`, `native probe` and `native fetch`.
- Preserve `synthesize`, `stream`, `play`, text/file/stdin input, and PCM16/float32 WAV output.
- Output is mono 24,000 Hz. Do not mislabel audio with an arbitrary sample rate.
- Omitted Mini voice chooses randomly among Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki and Leo. Unseeded requests avoid the last random voice when possible; seeded choices ignore history and remain reproducible. Decoder seed is forwarded to audio.cpp.
- Kitten2 uses the model default voice when omitted; requires a local GGUF and `--accept-kitten2-license`. Never automatically download its weights. Its seed must fit i64; the port does not implement speed changes.
- Native backend auto tries accelerators then CPU: Metal on macOS; CUDA/Vulkan/HIP on Linux and Windows. Retry once on CPU for silent or non-finite accelerated output. Explicit backends do not retry. Invalid output must never be written.
- Never bundle GGUF weights, audio.cpp libraries or libespeak-ng. Mini uses libespeak-ng and its data, not a CLI phonemizer subprocess.
- Check download byte counts, reject empty/truncated transfers, and validate GGUF magic before caching and on reuse. Corrupt Mini caches point to `native fetch --force`.
- Doctor checks library ABI/family, eSpeak paths and GGUF header in one pass, without downloading, and exits non-zero on any failed check.
- Playback defaults to gain 2.5 and an anti-clipping limiter; `--allow-clipping` is explicit opt-out.
- Guardrails: speed 0.5–2.0; max_chars 1–2000; normalized input at most 20,000 bytes; at most 256 chunks; native threads 1–256.
