# AGENTS.md

This repository contains a Rust command-line interface for KittenTTS ONNX models.

## Purpose

- Fetch KittenTTS model artifacts from Hugging Face on first use.
- Run ONNX inference in Rust.
- Generate speech audio from text or text files.

## Quick Test Commands

```bash
cargo run -- models list
cargo run -- model fetch --model nano-0.8-fp32
cargo run -- voices --model nano-0.8-fp32
cargo run -- synthesize --text "Hello from kitten-tts" --output output.wav --phonemizer basic
cargo run -- play --text "Hello from kitten-tts speakers" --phonemizer basic
```

## Notes

- Default model: `KittenML/kitten-tts-nano-0.8-fp32`
- `--phonemizer auto` prefers `espeak-ng` or `espeak`
- `--phonemizer basic` avoids system phonemizer dependencies
