# Native setup and reference

## Build audio.cpp

Use a release with C API ABI major 0 (v0.9.2 is tested):

```bash
git clone --branch v0.9.2 https://github.com/0xShug0/audio.cpp.git
cd audio.cpp
cmake -S . -B build -DAUDIOCPP_BUILD_C_API=ON
cmake --build build --target audiocpp --config Release
```

Set `AUDIOCPP_LIB` to the resulting shared library, or pass `--native-lib PATH`. Keep companion shared libraries available to the platform loader. Backend support depends on the audio.cpp build and its runtime prerequisites; consult upstream for accelerator build flags.

Library lookup uses `--native-lib`, then `AUDIOCPP_LIB`, then the platform library name. A successful Rust build does not prove the runtime library is available.

## Diagnose and fetch

```bash
kitten-tts native probe --native-lib /path/to/libaudiocpp.dylib
kitten-tts native fetch --cache-dir /path/to/cache
kitten-tts native doctor --cache-dir /path/to/cache
kitten-tts native doctor --family kitten_tts2 --native-model /path/to/kitten2.gguf
```

`probe` checks ABI, families and tasks. `doctor` checks library, eSpeak paths (Mini only), and model without downloading; it reports every failed section and exits non-zero. It does not run inference, validate Kitten2 license acceptance, or prove a GPU backend works.

Mini uses the platform cache under `native/kitten-tts-mini-0.8-orig.gguf`. `--cache-dir DIR` overrides it for synthesis, fetch and doctor. `--native-model PATH` bypasses weight downloading. Cache hits and local models require a GGUF header; downloads must be non-empty, match the advertised transfer size and have a GGUF header before being cached. These checks are not a model checksum or full integrity proof. Repair a corrupt Mini cache with `native fetch --force` (and the same `--cache-dir`, if used).

Kitten2 requires `--native-family kitten_tts2 --native-model PATH --accept-kitten2-license`. `native fetch --family kitten_tts2` deliberately fails without downloading.

## Output and controls

- Inputs: `--text`, `--text-file` (mutually exclusive), or piped stdin.
- `--native-backend auto|cpu|metal|cuda|vulkan|hip`, `--native-threads 1..256`.
- Auto tries Metal on macOS, CUDA/Vulkan/HIP on Linux or Windows, then CPU. It retries silent or non-finite accelerated audio once on CPU. Explicit backends do not retry.
- `--voice NAME`, `--seed U64`, `--speed 0.5..2.0`, `--max-chars 1..2000`.
- Kitten2 seed must fit a signed 64-bit integer. Its port does not apply speed changes.
- Unseeded Mini voice selection avoids the previous random voice when possible, recording it in the cache. Seeded selection ignores that history. Mini style handling stays inside audio.cpp; style rows are not exposed by this CLI.
- Output is fixed at 24 kHz mono. `--sample-rate 24000` remains accepted for compatibility; other values are rejected, not relabelled or resampled.
- `synthesize --output PATH --wav-encoding pcm16|float32`.
- `stream` emits raw float32 little-endian PCM without a header.
- `play --player auto|ffplay|pw-play --gain 0.1..8.0`; default gain 2.5 with limiting. `--allow-clipping` disables the limiter.
- `examples/kitten-say.sh "Hello"` and `make say SAY_TEXT="Hello"` wrap `play`. They require the same native prerequisites.

For a PCM consumer:

```bash
kitten-tts stream --text "Streaming speech." \
  | ffplay -autoexit -nodisp -loglevel error -f f32le -ar 24000 -ch_layout mono -
```
