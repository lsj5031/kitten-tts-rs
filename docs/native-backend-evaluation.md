# Native multi-model TTS backend: evaluation and decision

Status: decision record for [#3](https://github.com/lsj5031/kitten-tts-rs/issues/3),
researched 2026-10-24 and updated the same day with the implemented native engine
and reproduced measurements. Sections under [Measured results](#measured-results)
are local runs; everything else is upstream or license text. No listening tests
were performed, so audio quality is only assessed through duration, level and
null comparison against upstream's CLI.

## Decision

Adopt a native, GGML-based multi-model engine as the **preferred** synthesis path,
starting with the `audio.cpp` C ABI, and keep the current CUDA/ONNX Runtime path
as a fallback until the native path reaches feature parity.

Concretely:

- **Preference order.** When a compatible native library and model package are
  available, use the native engine. Fall back to ONNX Runtime when the native
  path cannot serve the request, and say so on stderr instead of falling back
  silently.
- **Why.** The goal is a more universal, more flexible, faster runtime. The
  current path is CUDA-only and tied to ONNX Runtime; the native path targets
  CPU, NVIDIA, AMD, Intel and Apple silicon from one weight format.
- **No removal yet.** The ONNX Runtime path stays. It is currently the only path
  that exposes the style-row control (`--style-index`) that this CLI is built
  around.
- **CLI contract is stable.** Commands, flag names and defaults stay as they are.
  New selection flags are additive, and the runtime guardrails (`--speed` in
  `[0.5, 2.0]`, `--max-chars` in `[1, 2000]`, input `<= 20000` chars,
  `<= 256` chunks) continue to be enforced by the CLI regardless of engine.
- **Reuse, don't rebuild.** No new model framework is written here. The native
  path is a binding to an existing engine.
- **Implemented selection.** `--engine auto` (the default) uses the native engine
  when it can serve the request: `--model mini-0.8`, an explicit `--native-model`,
  `--native-family kitten_tts2`, or a default request on macOS, where the
  CUDA-only ONNX path cannot run. Explicit ONNX presets (`nano-0.8-fp32`) and
  ONNX-only flags (`--repo-id`, `--style-index`, `--vram-profile`, `--trim-tail`,
  `--phonemizer`, ORT/CUDA options) keep ONNX Runtime. `--engine native` forces
  the native path. No ORT CPU fallback was added.
- **Hardware acceleration by default.** `--native-backend auto` (the default)
  selects Metal on macOS arm64 and CUDA for `kitten_tts2` elsewhere, then CPU. If
  an accelerated run returns silent or non-finite audio, it is retried on CPU
  before the command fails, so the guard above never has to reject a run that
  the other backend can complete. Silent or non-finite output is never written
  to a WAV.
- **Explicit model prefetch.** `kitten-tts native fetch` downloads the Mini 0.8
  GGUF without synthesizing, and `models list` points at it. KittenTTS 2 weights
  remain a manual download.
- **Verified downloads.** Files are written to a temporary path and renamed into
  the cache only after the received byte count matches `Content-Length` and is
  non-empty. Native models must start with the GGUF magic, checked after download
  and on every cache hit, so a truncated or substituted file is rejected with
  `native fetch --force` guidance instead of reaching the engine.
- **One-pass environment check.** `kitten-tts native doctor` verifies the
  library (load, ABI, requested family), the eSpeak library and data paths, and
  the model (cached or `--native-model`) in a single run, reporting every
  failure instead of stopping at the first. It downloads nothing and exits
  non-zero when any check fails.

Decision owner: repository maintainer. Revisit this record if the audio.cpp ABI
breaks its major version, if the Kitten ports regress, or if licensing terms
change.

## What changed since the issue was filed

The issue warns "do not assume KittenTTS 2 support". At the reviewed revision that
caution is outdated in a useful direction: `audio.cpp` already ships both Kitten
families as community model ports.

| Port | Upstream doc at reviewed revision | Support |
|---|---|---|
| `kitten_tts` | `docs/models/kitten_tts.md` | KittenTTS **Mini 0.8**, 80M params, English, 24 kHz mono, eight built-in voices, shared eSpeak-ng frontend, GGUF FP32 |
| `kitten_tts2` | `docs/models/kitten_tts2.md` | KittenTTS **2**: Qwen3 + S3 meanflow + HiFT, multilingual presets, reference-audio cloning, Q8/Q4 GGUF, no Python/LibTorch/ONNX Runtime at runtime |

Two consequences follow.

1. The primary candidate is no longer only a *future* host for this CLI's models;
   it can run them today.
2. The v2 port is the more interesting one for this project's stated direction
   (multilingual, cloning, quantized weights, CPU+CUDA), while the 0.8 port is
   the migration target for the existing presets.

Both are community ports, not core families. Both have now been exercised end to
end on CPU with real weights (see measured results), including v2 Q8 English and
German presets, streaming and multi-request text.

## Candidates

| | audio.cpp | sherpa-onnx | kitten-tts-2-cpp |
|---|---|---|---|
| License | Apache-2.0 (LICENSE text verified) | Apache-2.0 | MIT (library) |
| Runtime | GGML, no ONNX Runtime | ONNX Runtime (bundled) | llama.cpp/GGML fork + CPU LibTorch |
| Backends | CPU, CUDA, Vulkan, Metal, HIP/ROCm | CPU, CUDA, DirectML, CoreML and other ORT providers | CPU only (CUDA/Metal compile flags exist but the runner forces CPU) |
| Kitten 0.8 | `kitten_tts` port, all four presets not verified (Mini 0.8 documented) | All four 0.8 variants documented | No |
| Kitten 2 | `kitten_tts2` port | Not documented | Yes, only model supported |
| Embedding API | Versioned C ABI (`audiocpp.h`), per-platform release archives | C/C++ API plus Rust crates | CLI only |
| Streaming | Pull-based session API, requires `mode="streaming"` | Yes | No (batch CLI) |
| Cancellation | No cancel/abort entry point at reviewed revision | Yes (streaming API) | n/a |
| Distribution cost | One shared library per platform/backend, plus GGUF weights | ONNX Runtime + model files | CPU LibTorch runtime + a 1.03 GB GGUF |
| Fit for this repo | Best: removes ORT, adds hardware backends, covers v0.8 and v2 | Weakest differentiator: still ORT under the hood | Narrow: one model, no GPU, heavy dependency |

### Why not sherpa-onnx

It is mature, Apache-2.0, documented for all four Kitten 0.8 variants, and has
Rust bindings. But it does not change the core constraint: inference is still
ONNX Runtime, so hardware coverage is whatever ORT providers ship, and we would
add a second runtime wrapper for the same models. It is a reasonable
**containment** option if the native path stalls, not the preferred direction.

### Why not kitten-tts-2-cpp

It is the most direct path to v2, but it is CPU-only, depends on CPU LibTorch,
uses a non-standard `TQ2_1` GGUF that only its own fork can load, and its CLI does
not implement upstream's synthesis streaming. That contradicts "more universal,
more performant, more flexible".

## Capability gaps to design around

Verified against the reviewed audio.cpp revision, not guessed:

1. **`--style-index` has no native equivalent.** The `kitten_tts` port exposes
   `voice_id`, `speed`, `seed`, `text_chunk_mode`, `text_chunk_size` and session
   options for weight types, graph capacity and eSpeak paths. There is no request
   option for a style embedding row. This is the single largest parity loss and
   the reason the ORT path stays selectable.
2. **No streaming for the Kitten ports.** The `kitten_tts` doc does not list a
   stream runtime, and `kitten_tts2` explicitly says streaming "is not
   implemented or validated" for that port. Implemented as the issue predicted:
   `stream` synthesises per chunk and emits PCM as it goes, and short text that
   fits in one chunk is one request. It is not incremental audio, so time to
   first audio equals generation time for that chunk.
3. **No cancellation in the C ABI.** The header exposes start/push/next/finish/
   reset but no cancel or abort. Interrupting playback means tearing down the
   session handle from the controlling thread, or running the engine in a worker
   process/thread and dropping its handles. Do not promise Ctrl-C mid-synthesis
   latency until this is measured.
4. **Voice mapping is not one-to-one.** Our canonical voices are whatever keys
   the downloaded `voices.npz` contains, with aliases such as `Leo` and `Bella`.
   The 0.8 port documents eight packaged names (`Bella`, `Jasper`, `Luna`,
   `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`) and a default ID of `expr-voice-5-m`,
   so native voices are resolved from the GGUF package, not from `voices.npz`.
   Overlap with each ONNX preset's archive must be checked before `--voice` is
   documented as interchangeable, and the `voices` command needs a native branch
   that lists packaged voices rather than style rows.
5. **Weight format changes the download story.** Our cache currently fetches
   ONNX + `voices.npz` + config from Hugging Face. The native path needs GGUF
   packages (`audio-cpp/audio.cpp-gguf`, `KittenTTS2-GGUF`), installed through
   audio.cpp's model manager or downloaded directly. Cache keys must not collide
   with the ONNX presets.
6. **eSpeak-ng is loaded in-process by the port.** The ONNX path invokes the
   `espeak-ng` binary, keeping GPL-3.0 at arm's length. The `kitten_tts` port
   loads `libespeak-ng` in-process and accepts `kitten_tts.espeak_library_path`
   and `kitten_tts.espeak_data_path` session options. Implemented: the CLI passes
   those options from `--native-espeak-lib`/`--native-espeak-data`, or from
   auto-detected Homebrew/system paths. The GPL-3.0 library is never bundled,
   so the Apache-2.0 binary only ever points at a separately installed library.
7. **ABI discipline.** audio.cpp gates on a major ABI version and ships one
   archive per platform and backend. Our loading policy already prefers an
   explicit path (`--ort-lib`, `ORT_DYLIB_PATH`) with dynamic loading; the native
   equivalent is `--native-lib` plus `AUDIOCPP_LIB`, a hard refusal when the ABI
   major does not match, and a probe that reports the library's build version.
8. **Backend validity is not guaranteed, and failures can be intermittent.**
   With audio.cpp 0.9.2 and the Mini 0.8 FP32 GGUF, Metal returned non-finite or
   silent samples in two runs among about a dozen, separated by successful
   runs; CPU never failed. Because an accelerator can fail on an untested input
   or host, `auto` validates the samples and retries on CPU instead of treating
   the backend as trusted. GPU output is not sample-identical to CPU output, so
   determinism is per-backend.

## Licensing and distribution

- `audio.cpp` source: Apache-2.0 (LICENSE text verified at the reviewed revision;
  GitHub's license detector reports "Other" for this repository despite the
  Apache-2.0 text, so rely on the file, not the badge).
- `kitten_tts` weights (KittenTTS Mini 0.8): Apache-2.0 per audio.cpp's
  `docs/model_licenses.md`, commercial use allowed.
- `kitten_tts2` weights: Stellon Labs Community License. Commercial use is
  permitted only with registration and is conditioned on revenue/funding
  thresholds; distribution requires shipping the agreement, a notice file and
  "Powered by KittenML" attribution. This is **not** compatible with presenting
  v2 as an unconditional feature of an Apache-2.0 tool. Default the native path
  to 0.8 weights; surface v2 as opt-in with the license notice.
- eSpeak-ng: GPL-3.0. The CLI auto-detects or takes an explicit path to the
  user's installed library and never bundles or redistributes it. See gap 6.
- v2 gating: `--native-family kitten_tts2` requires both a local `--native-model`
  GGUF and `--accept-kitten2-license`; there is no automatic v2 download.
- Do not redistribute weights in release archives; keep fetching into the user
  cache so each model keeps its own license.

## Migration plan

Phases are sequential; each has an exit criterion that is testable without
listening tests.

**P0 — Decide and pin (done by this document).**
Pin the engine revision and document the decision. Exit: this file is merged and
the issue is updated.

**P1 — Native library loader + engine probe (implemented, no synthesis).**
`kitten-tts native probe` loads `libaudiocpp` through `--native-lib` or
`AUDIOCPP_LIB`, checks `audiocpp_abi_version()` major, and reports the build
version, ABI version, registered families and task vocabulary. The loader is
small and always compiled; it only activates when pointed at a library, so no
build feature was needed. The existing commands keep using ONNX Runtime.
Exit: met — probe output identifies the native library, ABI version, families,
and whether `kitten_tts`/`kitten_tts2` are present, with unit tests for ABI-major
refusal, explicit-path precedence and the missing-library error path.

**P2 — 0.8 synthesis through the native engine (implemented and verified).**
`--engine native` maps `synthesize`, `stream` and `play` onto a `kitten_tts`
session: text, voice, speed, seed, and chunking (one request when the cleaned
text fits `--max-chars`). `--style-index`, `--vram-profile` and `--repo-id` are
rejected with actionable errors. Exit: met for macOS arm64 CPU — WAV and PCM
output, deterministic runs, upstream parity within 1 PCM16 LSB, tests for
engine/flag combinations. Native CUDA could not be tested on this host.

**P3 — Prefer the native engine (implemented).**
`--engine auto` is the default and prefers native when it can serve the request
(`--model mini-0.8`, explicit `--native-model`, the v2 family, or a default
request on macOS); explicit ONNX presets and ONNX-only flags otherwise use ONNX
Runtime. A native preparation failure prints a one-line notice before falling
back. Exit: met — selection rules are documented and unit-tested.

**P4 — Optional v2 family (implemented and validated on CPU).**
KittenTTS 2 is opt-in through `--native-family kitten_tts2` with a local GGUF and
`--accept-kitten2-license`; v2 weights are never downloaded or bundled. Exit:
met — license notice in `--help` and README; v2 cannot mix with the 0.8 cache
because it requires a user-supplied path. The 2026-10-21 Q8 GGUF was run on
macOS arm64 CPU for English, German, streaming and multi-request text. CUDA,
cloning, the other language presets and listening quality remain unvalidated.

**Non-goals.** Rewriting the synthesis pipeline, changing the CLI's command
names, dropping CUDA ORT releases, bundling weights, and implementing another
model framework.

## Measured results

Machine: Apple M5 (arm64), macOS, audio.cpp 0.9.2 release C API (Metal build,
CPU compute), KittenTTS Mini 0.8 FP32 GGUF (302,167,648 bytes). Text: "Hello from
the native Kitten T T S engine. This is a real synthesis test." Voice `Leo`,
`--seed 7`, 6.59 s of output (158,200 samples at 24 kHz). One run per column.

| Metric | Native CPU, 1 thread | Native CPU, 8 threads |
|---|---:|---:|
| Session ready (library + model load) | 597 ms | 170 ms |
| Total synthesis | 5.79 s | 1.79 s |
| Real-time factor | 0.88 | 0.27 |
| Peak RSS | 1.12 GiB | 1.12 GiB |
| Peak VRAM | n/a (unified memory) | n/a |
| Packaged footprint | 31 MiB library + 288 MiB GGUF | same |

- The session-ready numbers span first-run and warm file-cache cases; 170 ms is
  the warm case.
- Repeated runs with the same seed and backend are byte-identical (SHA-256).
- `stream` produced the same 158,200 samples as `synthesize`.
- Against upstream `audiocpp_cli` with the same model, voice, rate and seed, the
  PCM16 payloads differ by at most 1 LSB (mean 0.48 LSB over 158,200 samples),
  consistent with WAV rounding rather than model divergence.
- ONNX Runtime comparison was not possible: this host has no CUDA device and the
  ONNX path is CUDA-only.
- Metal is faster than CPU once warmed: 0.79-0.89 s vs 1.71 s for the same
  6.59 s of audio at 8 threads (RTF 0.12-0.14 vs 0.26). `auto` selects Metal on
  macOS arm64 and produced valid audio in three consecutive 6.59 s runs
  (158,200 samples, RMS 0.1486).
- The retry path is not theoretical: a later short-text run returned invalid
  audio from Metal and `auto` logged the retry, switched to CPU and wrote valid
  output. Metal failures are therefore intermittent (two observed, separated by
  successes); this is why the fallback exists.
- Consequence for latency: `auto` is about 2x faster when Metal succeeds, and
  slower by one failed attempt when it does not. `--native-backend cpu` is the
  predictable choice.
- No listening tests were run, so quality is inferred from duration, level,
  determinism and upstream parity only.

Verified gates:

- Output is 24 kHz mono and matches upstream duration exactly for one request.
- Same seed and voice are repeatable; random voice selection is seedable.
- Long input still chunks; short multi-sentence input is one request.
- Missing library, missing model, ABI-major mismatch, invalid voice, silent or
  non-finite output, and missing v2 license acknowledgement all fail with
  actionable messages.
- Offline synthesis works from the cache after the model is present.

### KittenTTS 2 (Q8)

Same machine, 8 threads, Q8 GGUF (3,343,404,128 bytes). Session ready 12.1-14.2 s,
peak RSS 3.46 GiB. English "Hello from Kitten T T S two." with voice `Bruno` and
`--seed 42`: 68,640 samples (2.86 s), generation 4.9-5.0 s (RTF 1.7), and
byte-identical across repeated runs. German preset: 108,480 samples (4.52 s) in
8.44 s (RTF 1.87). A three-request long text: 381,120 samples (15.88 s) in
19.27 s (RTF 1.21). Streaming produced the same 68,640 samples as `synthesize`.
`--speed 1.5` produced bit-identical audio to 1.0, confirming the port ignores
speed; the CLI now prints a notice.

Metal also runs v2 and is faster: the same English request produced 75,360
samples (3.14 s) in 2.92 s generation (RTF 0.93) including a 13.9 s session load,
versus 4.9 s generation on CPU. The sample count differs from CPU (75,360 vs
68,640), so determinism is per backend; output level was valid (RMS 0.117).

Exercised: macOS arm64, native CPU and Metal — Mini 0.8 at 1 and 8 threads
(single and multi-request), KittenTTS 2 Q8 (English, German, streaming,
multi-request). Not exercised: native CUDA, Vulkan, HIP, Linux, Windows, v2
cloning and remaining language presets, listening quality.

## CLI compatibility map

| Current flag | Native 0.8 behaviour |
|---|---|
| `--engine auto\|onnx\|native` | New; `auto` prefers native when it can serve the request, including the macOS default |
| `--model` presets | Only `mini-0.8` has a packaged GGUF; other presets error unless `--native-model` overrides |
| `--repo-id` | ONNX-only; error when combined with the native engine |
| `--voice` | Supported via `voice_id`; invalid names list the available 0.8 voices |
| `--style-index` | Unsupported; error with guidance |
| `--speed` | Supported; keep `[0.5, 2.0]` validation |
| `--seed` | Supported via the `seed` request option |
| `--phonemizer` | Ignored with a notice; the native 0.8 frontend uses its own eSpeak-ng |
| `--trim-tail` | Ignored with a notice; the port trims its own output |
| `--vram-profile` | ONNX-only; error with guidance |
| `--arena-strategy`, `--conv-algo`, `--graph-opt`, `--memory-pattern`, `--parallel-execution`, `--ort-lib`, `--cuda-lib-dir`, `--cudnn-lib-dir` | Unused on the native path; ignored so wrapper scripts keep working |
| `--native-lib`, `--native-backend`, `--native-threads`, `--native-model`, `--native-family`, `--native-espeak-lib`, `--native-espeak-data`, `--accept-kitten2-license` | New native controls |
| `stream`, `play`, `synthesize` | Same flags, native audio path |

## Open questions

- Does the `kitten_tts` port expose style rows through any request option not
  documented in its model doc? If yes, gap 1 shrinks and the ORT path becomes
  removable sooner.
- Will audio.cpp add cancellation to the C ABI? If not, decide between a worker
  thread with handle teardown and a subprocess boundary.
- Which GPU backends are valid for `kitten_tts` on other hosts? CUDA was not
  available here, and Metal fails intermittently on this model, so `auto`
  validates output and falls back to CPU. Revisit when upstream publishes a
  support matrix or a fix; once Metal is reliable, the retry can stay as a cheap
  safety net without changing selection.
- Does the project want automatic v2 downloads after license review? Today v2
  requires a user-supplied GGUF, which keeps the 2.6-3.3 GB weights and the
  Stellon terms at arm's length.
- Should the eight 0.8 voice names be exposed by a `native voices` command
  instead of README text and error messages?
- Should `--speed` on `kitten_tts2` be rejected instead of announced and ignored,
  now that the port is known not to implement it?

## Sources

All URLs were read at the revisions below on 2026-10-24.

- audio.cpp, revision `dd92e32b6e268178ff43f4fa3ca36f0b380491e0` (committed
  2026-10-24): `README.md`, `LICENSE`, `include/audiocpp.h`, `docs/c_api.md`,
  `docs/models/kitten_tts.md`, `docs/models/kitten_tts2.md`,
  `docs/model_licenses.md`.
- kitten-tts-2-cpp, revision `2b9912e72245f5597c664d2be44a6e7ba0a3f293`
  (committed 2026-10-21): `README.md`.
- sherpa-onnx: repository metadata (Apache-2.0), `rust-api-examples/`,
  crates.io `sherpa-onnx` and `sherpa-onnx-sys` 2.0.0 (published 2026-10-23).
- audio.cpp release `v0.9.2`, assets `audio-v0.9.2-lib-macos-arm64-metal.tar.gz`
  and `audio-v0.9.2-bin-macos-arm64-metal.tar.gz`.
- Hugging Face `audio-cpp/audio.cpp-gguf`, `KittenTTS-GGUF/`
  (`kitten-tts-mini-0.8-orig.gguf`, 302,167,648 bytes) and `KittenTTS2-GGUF/`
  (2.2-3.3 GB packages).
- Hugging Face `KittenML/kitten-tts-2`, `LICENSE.md` (Stellon Labs Community
  License).
- espeak-ng: GPL-3.0 per repository license metadata; Homebrew 1.52.0 used for
  the local runtime, not redistributed.
