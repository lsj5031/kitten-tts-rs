# AGENTS.md

Agent operating guide for `kitten-tts-rs`.

## 1) Project Overview

- Rust CLI for KittenTTS ONNX inference.
- Downloads model artifacts from Hugging Face cache-on-first-use.
- Produces WAV files, PCM stream output, and direct playback.
- Primary implementation is in one file: `src/main.rs`.

## 2) Repository Reality Check

- Crate layout: single package (`Cargo.toml` has `[package]`, no workspace).
- Release CI builds with target triples in `.github/workflows/release.yml`.
- Checked for editor/assistant policy files:
  - `.cursorrules`: not present
  - `.cursor/rules/`: not present
  - `.github/copilot-instructions.md`: not present

If those files are added later, treat them as higher-priority repo policy.

## 3) Canonical Commands (Build / Lint / Test)

These commands are grounded in current repo files and observed behavior.

### Build

```bash
cargo build --release
```

CI/release-style cross-target build:

```bash
cargo build --release --target x86_64-unknown-linux-gnu
```

### Lint + Format

No lint/format command is currently enforced by checked-in CI.
Use these for safe local validation on code changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

If clippy reports pre-existing warnings, note that in your final report.

### Tests

Run all tests:

```bash
cargo test
```

List tests before choosing one:

```bash
cargo test -- --list
```

Run one exact unit test (preferred in this repo):

```bash
cargo test tests::model_selection_defaults_to_nano_fp32 -- --exact
```

Single-test tip:

- With `--exact`, use fully-qualified names (`tests::...`) to avoid 0 matches.

### CLI Smoke Commands

Assumes ONNX Runtime is available via `--ort-lib` or `ORT_DYLIB_PATH`.

```bash
cargo run -- models list
cargo run -- model fetch --model nano-0.8-fp32
cargo run -- voices --model nano-0.8-fp32
cargo run -- synthesize --text "Hello from kitten-tts" --output output.wav --phonemizer espeak-ng --ort-lib /path/to/libonnxruntime.so
cargo run -- play --text "Hello from kitten-tts speakers" --phonemizer espeak-ng --ort-lib /path/to/libonnxruntime.so
```

## 4) Source Layout and Placement Rules

- Keep small/medium changes in `src/main.rs` to match current structure.
- Add new CLI options via existing clap patterns in current arg structs/enums.
- Keep unit tests in the `#[cfg(test)] mod tests` section at file bottom.
- Prefer surgical edits; do not perform broad refactors unless explicitly asked.

## 5) Style Conventions (Observed Patterns)

### Imports

- Group imports at file top.
- Use explicit imports and nested brace groups.
- Avoid wildcard imports.

Examples in repo:

- `use anyhow::{Context, Result, anyhow, bail};`
- `use clap::{Args, Parser, Subcommand, ValueEnum};`
- `use std::io::{self, IsTerminal, Read, Write};`

### Naming

- Types/enums/traits: `PascalCase`.
- Functions/methods/tests: `snake_case`.
- Constants/statics: `SCREAMING_SNAKE_CASE`.

Examples:

- `ModelSelection`, `PhonemizerMode`, `ModelPreset`
- `resolve_cache_dir`, `clean_text_basic`, `resolve_style_index`
- `DEFAULT_REPO_ID`, `TOKEN_SPLIT_RE`, `MAX_INPUT_TOKENS`

### clap / CLI Modeling

- Use derives: `Parser`, `Subcommand`, `Args`, `ValueEnum`.
- Reuse argument groups with `#[command(flatten)]`.
- Use `#[arg(long, ...)]` for long-form flags.
- Prefer typed defaults: `default_value_t = ...`.
- Map external enum values using `#[value(name = "...")]`.

### Types and Function Signatures

- Favor strong enums/structs over ad-hoc strings.
- Use explicit return types.
- Use `Result<T>` on fallible paths.
- Keep helper functions focused and composable.

### Error Handling (Important)

- Primary pattern: `anyhow::Result<T>` + `?` propagation.
- Use `bail!(...)` for invalid input/state.
- Convert options to errors with `ok_or_else(|| anyhow!(...))`.
- Add `.context(...)` / `.with_context(...)` around I/O, parsing, HTTP, and process boundaries.
- Include actionable identifiers in messages (path, URL, command, model, voice, etc.).
- Avoid silent fallthrough and hidden error swallowing.

### Panics / unwrap

- Avoid `unwrap()` in runtime/user-triggered paths.
- `expect(...)` is acceptable only for static invariants (e.g., hardcoded regex).

### Tests

- Unit tests live in `src/main.rs` under `#[cfg(test)] mod tests`.
- Use `use super::*;` inside the tests module.
- Use clear behavior-driven test names.
- Keep tests deterministic and offline-friendly.

### Formatting / Readability

- Keep code rustfmt-friendly.
- Prefer simple control flow and early returns.
- Minimize comments; prioritize clear naming and precise errors.

## 6) Runtime/Product Defaults to Preserve

- Default model: `KittenML/kitten-tts-nano-0.8-fp32`.
- Default voice/style behavior:
  - when `--voice` is omitted, choose a random canonical voice
  - when `--style-index` is omitted, choose a random style row
  - `--seed` keeps random voice/style selection reproducible
- Playback default behavior:
  - `play` uses default `--gain 2.5`
  - anti-clipping limiter is enabled by default
  - `--allow-clipping` is available for manual/unsafe gain control
- GPU-only runtime: CUDA execution provider is required; do not add CPU fallback paths.
- ONNX Runtime loading policy:
  - use dynamic loading (`load-dynamic`) with user/system-provided `libonnxruntime`
  - preserve `--ort-lib`, `--cuda-lib-dir`, and `--cudnn-lib-dir` CLI options
  - do not reintroduce `download-binaries` in `ort` features unless explicitly requested
- Default phonemizer: `espeak-ng`.
- `--phonemizer auto`: tries `espeak-ng`, then `espeak` (no basic fallback).
- Default sample rate in code is `24000`.
- CUDA safety profile defaults:
  - Device: `0`
  - Memory arena limit: `2 GiB`
  - cuDNN conv algorithm search: `Heuristic` (not exhaustive)
  - cuDNN max-workspace search: disabled
- Host CPU safety defaults:
  - intra-op threads: `1`
  - inter-op threads: `1`
  - intra-op spinning: disabled
  - inter-op spinning: disabled
- Runtime guardrails:
  - `speed` must remain in `[0.5, 2.0]`
  - `max_chars` must remain in `[1, 2000]`
  - normalized input length must remain `<= 20000` chars
  - chunk count must remain `<= 256`

## 7) Done Criteria for Agent Changes

Before claiming completion on code edits:

1. Run relevant tests (at least targeted; full `cargo test` when broad).
2. Run `cargo build --release` when build/runtime behavior is affected.
3. Run fmt/clippy checks for Rust code changes.
4. Verify affected CLI path with `cargo run -- ...` smoke command(s).
5. Keep behavior and style aligned with this document and existing code.
