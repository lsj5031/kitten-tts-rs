# kitten-tts

Rust CLI for KittenTTS ONNX models.

## Features

- Pure Rust ONNX Runtime inference
- First-run model download and cache
- Multiple model presets and custom repo ID support
- Voice aliases and canonical voice IDs
- File output (`synthesize`) and stdout streaming (`stream`)
- Direct speaker playback (`play`)
- Optional system phonemizer with built-in fallback

## Install and Run

```bash
cargo build
cargo run -- models list
```

## Basic Usage

Synthesize to WAV:

```bash
cargo run -- synthesize \
  --text "Hello from Rust KittenTTS." \
  --output hello.wav \
  --model nano-0.8-fp32 \
  --phonemizer basic
```

Tune tone/prosody by overriding style index (0..399):

```bash
cargo run -- synthesize \
  --text "Style test sentence" \
  --voice Bella \
  --phonemizer espeak-ng \
  --style-index 320 \
  --output styled.wav
```

Read input from a file:

```bash
cargo run -- synthesize \
  --text-file README.md \
  --output readme.wav \
  --model nano-0.8-fp32 \
  --phonemizer basic
```

Stream raw f32 PCM to stdout:

```bash
cargo run -- stream --text "Streaming test" --phonemizer basic > stream.f32
```

Play directly to your default audio device:

```bash
cargo run -- play --text "Hello from speakers" --phonemizer basic
```
