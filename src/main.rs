use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use directories::ProjectDirs;
use libloading::Library;
use ndarray::{Array1, Array2};
use once_cell::sync::{Lazy, OnceCell};
use ort::{
    ep, inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use regex::Regex;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, IsTerminal, Read, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use zip::ZipArchive;

const DEFAULT_REPO_ID: &str = "KittenML/kitten-tts-nano-0.8-fp32";
const DEFAULT_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_MAX_CHARS: usize = 400;
const DEFAULT_TRIM_TAIL: usize = 5_000;
const DEFAULT_PLAY_GAIN: f32 = 2.5;
const DEFAULT_CUDA_MEMORY_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_CUDA_DEVICE_ID: i32 = 0;
const SAFE_INTRA_THREADS: usize = 1;
const SAFE_INTER_THREADS: usize = 1;
const MAX_INPUT_TOKENS: usize = 510;
const MAX_TOTAL_INPUT_CHARS: usize = 20_000;
const MAX_SYNTHESIS_CHUNKS: usize = 256;
const MAX_CHARS_HARD_LIMIT: usize = 2_000;
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 2.0;
const MIN_OUTPUT_GAIN: f32 = 0.1;
const MAX_OUTPUT_GAIN: f32 = 8.0;
const VRAM_HEADROOM_MIB: u64 = 512;
const SAFE_PLAYBACK_PEAK: f32 = 0.98;
const TOKEN_PAD_ID: i64 = 0;
const MAX_NPY_ENTRY_BYTES: u64 = 50 * 1024 * 1024;
const AUDIOCPP_ABI_MAJOR: u32 = 0;
const AUDIOCPP_LIB_ENV: &str = "AUDIOCPP_LIB";
const AUDIOCPP_OK: c_int = 0;
const MAX_NATIVE_FAMILIES: usize = 4096;
const MAX_NATIVE_TASKS: usize = 64;
const NATIVE_MODEL_REPO: &str = "audio-cpp/audio.cpp-gguf";
const NATIVE_KITTEN_TTS_DIR: &str = "KittenTTS-GGUF";
const NATIVE_KITTEN_TTS_FILE: &str = "kitten-tts-mini-0.8-orig.gguf";
const NATIVE_KITTEN_TTS_VOICES: [&str; 8] = [
    "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
];

static SPACES_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));
static ORT_INIT: OnceCell<()> = OnceCell::new();

#[derive(Debug, Parser)]
#[command(name = "kitten-tts")]
#[command(about = "Rust CLI for KittenTTS ONNX models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Synthesize(SynthesizeArgs),
    Stream(StreamArgs),
    Play(PlayArgs),
    Voices(VoicesArgs),
    Model(ModelCommandTop),
    Models(ModelsCommandTop),
    Native(NativeCommandTop),
}

#[derive(Debug, Clone, Args)]
struct SynthesisArgs {
    #[command(flatten)]
    model: ModelSelection,
    #[command(flatten)]
    text: TextInput,
    #[arg(
        long,
        help = "Voice name/alias. If omitted, pick a random available voice"
    )]
    voice: Option<String>,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    #[arg(long, default_value_t = DEFAULT_SAMPLE_RATE)]
    sample_rate: u32,
    #[arg(long, default_value_t = DEFAULT_MAX_CHARS)]
    max_chars: usize,
    #[arg(long, default_value_t = DEFAULT_TRIM_TAIL)]
    trim_tail: usize,
    #[arg(long)]
    style_index: Option<usize>,
    #[arg(long, help = "Random seed for reproducible voice/style selection")]
    seed: Option<u64>,
    #[arg(long, value_enum, default_value_t = PhonemizerMode::EspeakNg)]
    phonemizer: PhonemizerMode,
    #[arg(
        long,
        value_enum,
        default_value_t = EngineChoice::Auto,
        help = "Synthesis engine: auto (native when supported, else ONNX), onnx, or native"
    )]
    engine: EngineChoice,
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to the audio.cpp C API library for the native engine (default: AUDIOCPP_LIB or platform library name)"
    )]
    native_lib: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        default_value_t = NativeBackend::Auto,
        help = "Native backend: auto, cpu, cuda, metal, vulkan, or hip"
    )]
    native_backend: NativeBackend,
    #[arg(
        long,
        default_value_t = 1,
        help = "Worker threads for the native engine"
    )]
    native_threads: i32,
    #[arg(
        long,
        value_name = "PATH",
        help = "Local GGUF for the native engine (default: download KittenTTS Mini 0.8)"
    )]
    native_model: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        default_value_t = NativeFamily::KittenTts,
        help = "Native model family"
    )]
    native_family: NativeFamily,
    #[arg(
        long,
        help = "Acknowledge the KittenTTS 2 weight license (required for --native-family kitten_tts2)"
    )]
    accept_kitten2_license: bool,
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to libespeak-ng for the native KittenTTS 0.8 frontend (default: auto-detect)"
    )]
    native_espeak_lib: Option<PathBuf>,
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to espeak-ng-data for the native KittenTTS 0.8 frontend (default: auto-detect)"
    )]
    native_espeak_data: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct SynthesizeArgs {
    #[command(flatten)]
    synthesis: SynthesisArgs,
    #[arg(long, default_value = "output.wav")]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = WavEncoding::Pcm16)]
    wav_encoding: WavEncoding,
}

#[derive(Debug, Args)]
struct StreamArgs {
    #[command(flatten)]
    synthesis: SynthesisArgs,
}

#[derive(Debug, Args)]
struct PlayArgs {
    #[command(flatten)]
    synthesis: SynthesisArgs,
    #[arg(long, value_enum, default_value_t = PlayerMode::Auto)]
    player: PlayerMode,
    #[arg(
        long,
        default_value_t = DEFAULT_PLAY_GAIN,
        help = "Playback gain multiplier applied before speaker output"
    )]
    gain: f32,
    #[arg(
        long,
        default_value_t = false,
        help = "Allow clipping instead of automatically limiting playback gain"
    )]
    allow_clipping: bool,
}

#[derive(Debug, Args)]
struct VoicesArgs {
    #[command(flatten)]
    model: ModelSelection,
}

#[derive(Debug, Args)]
struct ModelCommandTop {
    #[command(subcommand)]
    command: ModelCommands,
}

#[derive(Debug, Subcommand)]
enum ModelCommands {
    Fetch(ModelFetchArgs),
    Info(ModelInfoArgs),
}

#[derive(Debug, Args)]
struct ModelFetchArgs {
    #[command(flatten)]
    model: ModelSelection,
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
struct ModelInfoArgs {
    #[command(flatten)]
    model: ModelSelection,
}

#[derive(Debug, Args)]
struct ModelsCommandTop {
    #[command(subcommand)]
    command: ModelsCommands,
}

#[derive(Debug, Subcommand)]
enum ModelsCommands {
    List,
}

#[derive(Debug, Args)]
#[command(
    about = "Native audio.cpp engine: probe the library and download models",
    long_about = "Native audio.cpp engine control. `doctor` checks everything `--engine native` needs in one pass: the library and its ABI/families, the eSpeak runtime, and the model (cached or explicit). `probe` reports library details only; `fetch` downloads the Mini 0.8 GGUF so `--engine native` can run offline. None of these commands need ONNX Runtime."
)]
struct NativeCommandTop {
    #[command(subcommand)]
    command: NativeCommands,
}

#[derive(Debug, Subcommand)]
enum NativeCommands {
    /// Check the audio.cpp library, its ABI and available model families
    Probe(NativeProbeArgs),
    /// Download the native model into the cache
    #[command(
        long_about = "Download the KittenTTS Mini 0.8 GGUF used by `--engine native --model mini-0.8` into the cache directory. Re-running is cheap: an existing file is reused unless --force is passed. KittenTTS 2 weights are licensed differently and are never downloaded by this command."
    )]
    Fetch(NativeFetchArgs),
    /// Check the library, eSpeak and cached model in one pass
    #[command(
        long_about = "Check the three things `--engine native` needs, reporting every problem in one run instead of failing at the first one: the audio.cpp library (load, ABI, requested family), the eSpeak runtime required by kitten_tts, and the model (the cached Mini 0.8 GGUF or a path passed with --native-model). Nothing is downloaded. Exits non-zero when a check fails."
    )]
    Doctor(NativeDoctorArgs),
}

#[derive(Debug, Args)]
struct NativeFetchArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = NativeFamily::KittenTts,
        help = "Native model family to download"
    )]
    family: NativeFamily,
    #[arg(long, help = "Cache directory (default: platform cache directory)")]
    cache_dir: Option<PathBuf>,
    #[arg(long, help = "Re-download even when the file is already cached")]
    force: bool,
}

#[derive(Debug, Args)]
struct NativeProbeArgs {
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to the audio.cpp C API library (default: AUDIOCPP_LIB or platform library name)"
    )]
    native_lib: Option<PathBuf>,
    #[arg(
        long,
        help = "Fail unless this model family is available (for example kitten_tts)"
    )]
    family: Option<String>,
}

#[derive(Debug, Args)]
struct NativeDoctorArgs {
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to the audio.cpp C API library (default: AUDIOCPP_LIB or platform library name)"
    )]
    native_lib: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        default_value_t = NativeFamily::KittenTts,
        help = "Model family to check the library and model for"
    )]
    family: NativeFamily,
    #[arg(
        long,
        value_name = "PATH",
        help = "Local GGUF to check instead of the cached Mini 0.8 model"
    )]
    native_model: Option<PathBuf>,
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to libespeak-ng (default: auto-detect Homebrew/system paths)"
    )]
    native_espeak_lib: Option<PathBuf>,
    #[arg(
        long,
        value_name = "PATH",
        help = "Path to espeak-ng-data (default: auto-detect Homebrew/system paths)"
    )]
    native_espeak_data: Option<PathBuf>,
    #[arg(
        long,
        value_name = "DIR",
        help = "Cache directory (default: platform cache directory)"
    )]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
struct ModelSelection {
    #[arg(long, value_enum)]
    model: Option<ModelPreset>,
    #[arg(long)]
    repo_id: Option<String>,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(
        long,
        help = "Path to libonnxruntime shared library for dynamic loading"
    )]
    ort_lib: Option<PathBuf>,
    #[arg(long, help = "Directory containing CUDA runtime shared libraries")]
    cuda_lib_dir: Option<PathBuf>,
    #[arg(long, help = "Directory containing cuDNN shared libraries")]
    cudnn_lib_dir: Option<PathBuf>,
    #[arg(long, help = "CUDA device index (default: 0)")]
    cuda_device_id: Option<i32>,
    #[arg(
        long,
        help = "CUDA memory arena limit in MiB (default: 2048). Lower this on GPUs with limited VRAM to avoid driver crashes"
    )]
    cuda_memory_limit: Option<usize>,
    #[arg(
        long,
        value_enum,
        default_value_t = ArenaStrategy::SameAsRequested,
        help = "Memory arena extend strategy: same-as-requested (minimize allocation) or next-power-of-two (better performance)"
    )]
    arena_strategy: ArenaStrategy,
    #[arg(
        long,
        value_enum,
        default_value_t = ConvAlgo::Heuristic,
        help = "cuDNN convolution algorithm search: heuristic (low memory) or exhaustive (better performance)"
    )]
    conv_algo: ConvAlgo,
    #[arg(
        long,
        value_enum,
        default_value_t = GraphOpt::Level1,
        help = "ONNX graph optimization level: basic (1) or full (3). Higher levels may use more VRAM but optimize better"
    )]
    graph_opt: GraphOpt,
    #[arg(
        long,
        default_value_t = false,
        help = "Enable memory pattern planning (pre-allocate buffers) for faster execution"
    )]
    memory_pattern: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Enable parallel execution (concurrent kernels) for potential speedup"
    )]
    parallel_execution: bool,
    #[arg(
        long,
        value_enum,
        help = "VRAM profile: minimal (lowest VRAM), balanced (default), performance (fastest)"
    )]
    vram_profile: Option<VramProfile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GraphOpt {
    #[value(name = "1")]
    Level1,
    #[value(name = "3")]
    Level3,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VramProfile {
    #[value(name = "minimal")]
    Minimal,
    #[value(name = "balanced")]
    Balanced,
    #[value(name = "performance")]
    Performance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ArenaStrategy {
    #[value(name = "same-as-requested")]
    SameAsRequested,
    #[value(name = "next-power-of-two")]
    NextPowerOfTwo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ConvAlgo {
    #[value(name = "heuristic")]
    Heuristic,
    #[value(name = "default")]
    Default,
    #[value(name = "exhaustive")]
    Exhaustive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedSessionConfig {
    arena_strategy: ArenaStrategy,
    conv_algo: ConvAlgo,
    graph_opt: GraphOpt,
    memory_pattern: bool,
    parallel_execution: bool,
    inter_threads: usize,
}

fn resolve_session_config(cfg: &OrtRuntimeConfig) -> ResolvedSessionConfig {
    match cfg.vram_profile {
        Some(VramProfile::Minimal) => ResolvedSessionConfig {
            arena_strategy: ArenaStrategy::SameAsRequested,
            conv_algo: ConvAlgo::Heuristic,
            graph_opt: GraphOpt::Level1,
            memory_pattern: false,
            parallel_execution: false,
            inter_threads: SAFE_INTER_THREADS,
        },
        Some(VramProfile::Performance) => ResolvedSessionConfig {
            arena_strategy: ArenaStrategy::NextPowerOfTwo,
            conv_algo: ConvAlgo::Exhaustive,
            graph_opt: GraphOpt::Level3,
            memory_pattern: true,
            parallel_execution: true,
            inter_threads: 0, // let ORT decide
        },
        Some(VramProfile::Balanced) | None => ResolvedSessionConfig {
            arena_strategy: cfg.arena_strategy,
            conv_algo: cfg.conv_algo,
            graph_opt: cfg.graph_opt,
            memory_pattern: cfg.memory_pattern,
            parallel_execution: cfg.parallel_execution,
            inter_threads: if cfg.parallel_execution {
                0
            } else {
                SAFE_INTER_THREADS
            },
        },
    }
}

#[derive(Debug, Clone)]
struct OrtRuntimeConfig {
    ort_lib: Option<PathBuf>,
    cuda_lib_dir: Option<PathBuf>,
    cudnn_lib_dir: Option<PathBuf>,
    cuda_device_id: i32,
    cuda_memory_limit_bytes: usize,
    arena_strategy: ArenaStrategy,
    conv_algo: ConvAlgo,
    graph_opt: GraphOpt,
    memory_pattern: bool,
    parallel_execution: bool,
    vram_profile: Option<VramProfile>,
}

#[derive(Debug, Clone, Args)]
struct TextInput {
    #[arg(long, conflicts_with = "text_file")]
    text: Option<String>,
    #[arg(long, conflicts_with = "text")]
    text_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum WavEncoding {
    Pcm16,
    Float32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PhonemizerMode {
    Auto,
    EspeakNg,
    Espeak,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PlayerMode {
    Auto,
    Ffplay,
    PwPlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EngineChoice {
    Auto,
    Onnx,
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum NativeBackend {
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Hip,
}

impl NativeBackend {
    fn candidates(self) -> Vec<&'static str> {
        match self {
            Self::Auto => {
                let mut backends = Vec::new();
                if cfg!(target_os = "macos") {
                    backends.push("metal");
                } else if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
                    backends.push("cuda");
                    backends.push("vulkan");
                    backends.push("hip");
                }
                backends.push("cpu");
                backends
            }
            Self::Cpu => vec!["cpu"],
            Self::Cuda => vec!["cuda"],
            Self::Metal => vec!["metal"],
            Self::Vulkan => vec!["vulkan"],
            Self::Hip => vec!["hip"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum NativeFamily {
    #[value(name = "kitten_tts")]
    KittenTts,
    #[value(name = "kitten_tts2")]
    KittenTts2,
}

impl NativeFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::KittenTts => "kitten_tts",
            Self::KittenTts2 => "kitten_tts2",
        }
    }

    fn canonical_voices(self) -> &'static [&'static str] {
        match self {
            Self::KittenTts => &NATIVE_KITTEN_TTS_VOICES,
            Self::KittenTts2 => &[],
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ModelPreset {
    #[value(name = "nano-0.8-int8")]
    Nano08Int8,
    #[value(name = "nano-0.8-fp32")]
    Nano08Fp32,
    #[value(name = "micro-0.8")]
    Micro08,
    #[value(name = "mini-0.8")]
    Mini08,
}

impl ModelPreset {
    fn repo_id(self) -> &'static str {
        match self {
            Self::Nano08Int8 => "KittenML/kitten-tts-nano-0.8-int8",
            Self::Nano08Fp32 => "KittenML/kitten-tts-nano-0.8-fp32",
            Self::Micro08 => "KittenML/kitten-tts-micro-0.8",
            Self::Mini08 => "KittenML/kitten-tts-mini-0.8",
        }
    }

    fn all() -> [Self; 4] {
        [
            Self::Nano08Int8,
            Self::Nano08Fp32,
            Self::Micro08,
            Self::Mini08,
        ]
    }
}

impl ModelSelection {
    fn resolve_repo_id(&self) -> String {
        if let Some(repo_id) = &self.repo_id {
            return repo_id.clone();
        }
        if let Some(preset) = self.model {
            return preset.repo_id().to_string();
        }
        DEFAULT_REPO_ID.to_string()
    }

    fn resolve_cache_dir(&self) -> Result<PathBuf> {
        resolve_cache_root(self.cache_dir.as_deref())
    }

    fn resolve_ort_runtime_config(&self) -> OrtRuntimeConfig {
        OrtRuntimeConfig {
            ort_lib: self.ort_lib.clone(),
            cuda_lib_dir: self.cuda_lib_dir.clone(),
            cudnn_lib_dir: self.cudnn_lib_dir.clone(),
            cuda_device_id: self.cuda_device_id.unwrap_or(DEFAULT_CUDA_DEVICE_ID),
            cuda_memory_limit_bytes: self
                .cuda_memory_limit
                .map(|mib| mib * 1024 * 1024)
                .unwrap_or(DEFAULT_CUDA_MEMORY_LIMIT_BYTES),
            arena_strategy: self.arena_strategy,
            conv_algo: self.conv_algo,
            graph_opt: self.graph_opt,
            memory_pattern: self.memory_pattern,
            parallel_execution: self.parallel_execution,
            vram_profile: self.vram_profile,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(rename = "type")]
    model_type: String,
    model_file: String,
    voices: String,
    #[serde(default)]
    speed_priors: HashMap<String, f32>,
    #[serde(default)]
    voice_aliases: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct CacheManifest {
    repo_id: String,
    fetched_at_unix: u64,
    model_file: String,
    voices_file: String,
}

#[derive(Debug)]
struct ModelArtifacts {
    repo_id: String,
    cache_dir: PathBuf,
    config_path: PathBuf,
    model_path: PathBuf,
    voices_path: PathBuf,
    config: ModelConfig,
}

#[derive(Debug)]
struct VoiceStyle {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl VoiceStyle {
    fn row(&self, index: usize) -> Result<&[f32]> {
        if self.cols == 0 {
            bail!("invalid voice style: zero columns");
        }
        if index >= self.rows {
            bail!("style row index out of bounds: {index} >= {}", self.rows);
        }
        let start = index * self.cols;
        let end = start + self.cols;
        Ok(&self.data[start..end])
    }
}

#[derive(Debug)]
struct VoiceTable {
    voices: BTreeMap<String, VoiceStyle>,
}

impl VoiceTable {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open voices archive: {}", path.display()))?;
        let mut archive = ZipArchive::new(file)
            .with_context(|| format!("failed to read npz archive: {}", path.display()))?;

        let mut voices = BTreeMap::new();
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();
            if !name.ends_with(".npy") {
                continue;
            }

            let mut raw = Vec::new();
            (&mut entry)
                .take(MAX_NPY_ENTRY_BYTES + 1)
                .read_to_end(&mut raw)?;
            if raw.len() as u64 > MAX_NPY_ENTRY_BYTES {
                bail!(
                    "npy entry '{name}' exceeds maximum allowed size ({MAX_NPY_ENTRY_BYTES} bytes)"
                );
            }
            let (rows, cols, data) =
                parse_npy_f32(&raw).with_context(|| format!("failed parsing npy entry: {name}"))?;

            let key = name.trim_end_matches(".npy").to_string();
            voices.insert(key, VoiceStyle { rows, cols, data });
        }

        if voices.is_empty() {
            bail!("voices archive did not contain any .npy entries");
        }

        Ok(Self { voices })
    }

    fn style_row(&self, voice: &str, index: usize) -> Result<&[f32]> {
        let style = self
            .voices
            .get(voice)
            .ok_or_else(|| anyhow!("voice '{voice}' not found in voices archive"))?;
        style.row(index)
    }

    fn canonical_voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    fn max_style_rows(&self, voice: &str) -> Result<usize> {
        let style = self
            .voices
            .get(voice)
            .ok_or_else(|| anyhow!("voice '{voice}' not found in voices archive"))?;
        Ok(style.rows)
    }
}

#[derive(Debug)]
struct TextCleaner {
    word_index_dictionary: HashMap<char, i64>,
}

impl TextCleaner {
    fn new() -> Self {
        let pad = "$";
        let punctuation = ";:,.!?¡¿—…\"«»“” ";
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

        let symbols: Vec<char> = pad
            .chars()
            .chain(punctuation.chars())
            .chain(letters.chars())
            .chain(letters_ipa.chars())
            .collect();

        let mut map = HashMap::with_capacity(symbols.len());
        for (idx, ch) in symbols.into_iter().enumerate() {
            map.insert(ch, idx as i64);
        }

        Self {
            word_index_dictionary: map,
        }
    }

    fn encode_into(&self, text: &str, out: &mut Vec<i64>) {
        for ch in text.chars() {
            if let Some(index) = self.word_index_dictionary.get(&ch) {
                out.push(*index);
            }
        }
    }
}

trait Phonemizer {
    fn name(&self) -> &str;
    fn phonemize(&self, text: &str) -> Result<String>;
}

#[derive(Debug)]
struct EspeakPhonemizer {
    program: String,
}

impl Phonemizer for EspeakPhonemizer {
    fn name(&self) -> &str {
        &self.program
    }

    fn phonemize(&self, text: &str) -> Result<String> {
        let output = Command::new(&self.program)
            .args(["-q", "--ipa=3", "-v", "en-us", text])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| {
                format!(
                    "failed to execute phonemizer program '{}': ensure it is installed",
                    self.program
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("phonemizer '{}' failed: {}", self.program, stderr.trim());
        }

        let stdout = String::from_utf8(output.stdout)
            .context("phonemizer emitted non-utf8 output")?
            .trim()
            .to_string();

        if stdout.is_empty() {
            bail!(
                "phonemizer '{}' returned empty phoneme output",
                self.program
            );
        }

        Ok(stdout)
    }
}

fn detect_phonemizer(mode: PhonemizerMode) -> Result<Box<dyn Phonemizer>> {
    match mode {
        PhonemizerMode::Auto => {
            if executable_in_path("espeak-ng") {
                return Ok(Box::new(EspeakPhonemizer {
                    program: "espeak-ng".to_string(),
                }));
            }
            if executable_in_path("espeak") {
                return Ok(Box::new(EspeakPhonemizer {
                    program: "espeak".to_string(),
                }));
            }
            bail!("no espeak phonemizer found on PATH (tried: espeak-ng, espeak)")
        }
        PhonemizerMode::EspeakNg => {
            if !executable_in_path("espeak-ng") {
                bail!("phonemizer 'espeak-ng' not found on PATH");
            }
            Ok(Box::new(EspeakPhonemizer {
                program: "espeak-ng".to_string(),
            }))
        }
        PhonemizerMode::Espeak => {
            if !executable_in_path("espeak") {
                bail!("phonemizer 'espeak' not found on PATH");
            }
            Ok(Box::new(EspeakPhonemizer {
                program: "espeak".to_string(),
            }))
        }
    }
}

fn executable_in_path(command: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    let paths = std::env::split_paths(&path_var);

    #[cfg(windows)]
    let exts: Vec<String> = std::env::var_os("PATHEXT")
        .map(|v| {
            v.to_string_lossy()
                .split(';')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![".exe".to_string(), ".bat".to_string(), ".cmd".to_string()]);

    for dir in paths {
        let candidate = dir.join(command);
        if candidate.is_file() {
            return true;
        }
        #[cfg(windows)]
        {
            for ext in &exts {
                let with_ext = dir.join(format!("{command}{ext}"));
                if with_ext.is_file() {
                    return true;
                }
            }
        }
    }

    false
}

fn detect_player(mode: PlayerMode) -> Result<&'static str> {
    match mode {
        PlayerMode::Auto => {
            if executable_in_path("ffplay") {
                return Ok("ffplay");
            }
            if executable_in_path("pw-play") {
                return Ok("pw-play");
            }
            bail!(
                "no supported audio player found. Install 'ffplay' or 'pw-play', or use `synthesize` and play the wav manually"
            )
        }
        PlayerMode::Ffplay => {
            if !executable_in_path("ffplay") {
                bail!("player 'ffplay' not found on PATH");
            }
            Ok("ffplay")
        }
        PlayerMode::PwPlay => {
            if !executable_in_path("pw-play") {
                bail!("player 'pw-play' not found on PATH");
            }
            Ok("pw-play")
        }
    }
}

struct Synthesizer {
    session: Session,
    voice_table: VoiceTable,
    text_cleaner: TextCleaner,
    speed_priors: HashMap<String, f32>,
    voice_aliases: HashMap<String, String>,
    random_voice_state_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct SynthesisConfig {
    speed: f32,
    max_chars: usize,
    trim_tail: usize,
    style_index: Option<usize>,
    seed: Option<u64>,
}

impl From<&SynthesisArgs> for SynthesisConfig {
    fn from(args: &SynthesisArgs) -> Self {
        Self {
            speed: args.speed,
            max_chars: args.max_chars,
            trim_tail: args.trim_tail,
            style_index: args.style_index,
            seed: args.seed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn from_seed(seed: u64) -> Self {
        // Avoid a zero-only cycle in xorshift.
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64* PRNG; fast and sufficient for style/voice selection.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn gen_index(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper_exclusive
    }
}

impl Synthesizer {
    fn new(artifacts: &ModelArtifacts, ort_runtime: &OrtRuntimeConfig) -> Result<Self> {
        init_ort(ort_runtime)?;

        let arena_mib = (ort_runtime.cuda_memory_limit_bytes / (1024 * 1024)) as u64;
        check_gpu_vram(ort_runtime.cuda_device_id, arena_mib)?;

        let resolved = resolve_session_config(ort_runtime);

        if let Some(profile) = ort_runtime.vram_profile {
            eprintln!("[VRAM Profile] {:?}", profile);
        }
        eprintln!(
            "[Session] arena={:?}, conv={:?}, graph_opt={:?}, mem_pattern={}, parallel={}, inter_threads={}",
            resolved.arena_strategy,
            resolved.conv_algo,
            resolved.graph_opt,
            resolved.memory_pattern,
            resolved.parallel_execution,
            resolved.inter_threads,
        );

        let arena_strategy = match resolved.arena_strategy {
            ArenaStrategy::SameAsRequested => ort::ep::ArenaExtendStrategy::SameAsRequested,
            ArenaStrategy::NextPowerOfTwo => ort::ep::ArenaExtendStrategy::NextPowerOfTwo,
        };
        let conv_algo = match resolved.conv_algo {
            ConvAlgo::Heuristic => ep::cuda::ConvAlgorithmSearch::Heuristic,
            ConvAlgo::Default => ep::cuda::ConvAlgorithmSearch::Default,
            ConvAlgo::Exhaustive => ep::cuda::ConvAlgorithmSearch::Exhaustive,
        };
        let graph_opt_level = match resolved.graph_opt {
            GraphOpt::Level1 => GraphOptimizationLevel::Level1,
            GraphOpt::Level3 => GraphOptimizationLevel::All,
        };

        let cuda_ep = ep::CUDA::default()
            .with_device_id(ort_runtime.cuda_device_id)
            .with_memory_limit(ort_runtime.cuda_memory_limit_bytes)
            .with_conv_algorithm_search(conv_algo)
            .with_conv_max_workspace(false)
            .with_arena_extend_strategy(arena_strategy)
            .build()
            .error_on_failure();

        let session_builder = Session::builder()?
            .with_no_environment_execution_providers()?
            .with_parallel_execution(resolved.parallel_execution)?
            .with_intra_threads(SAFE_INTRA_THREADS)?
            .with_inter_threads(resolved.inter_threads)?
            .with_intra_op_spinning(false)?
            .with_inter_op_spinning(false)?
            .with_memory_pattern(resolved.memory_pattern)?
            .with_deterministic_compute(true)?;

        let session = session_builder
            .with_execution_providers([cuda_ep])
            .context(
                "failed enabling CUDA execution provider (GPU is required; ensure CUDA 12 + cuDNN runtime libraries are installed)",
            )?
            .with_optimization_level(graph_opt_level)?
            .commit_from_file(&artifacts.model_path)
            .with_context(|| {
                format!(
                    "failed to load onnx model from {}",
                    artifacts.model_path.display()
                )
            })?;

        let voice_table = VoiceTable::load(&artifacts.voices_path)?;

        Ok(Self {
            session,
            voice_table,
            text_cleaner: TextCleaner::new(),
            speed_priors: artifacts.config.speed_priors.clone(),
            voice_aliases: artifacts.config.voice_aliases.clone(),
            random_voice_state_path: artifacts.cache_dir.join("last_random_voice.txt"),
        })
    }

    fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        config: SynthesisConfig,
        phonemizer: &dyn Phonemizer,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.synthesize_streaming(text, voice, config, phonemizer, |chunk| {
            output.extend_from_slice(chunk);
            Ok(())
        })?;
        Ok(output)
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        config: SynthesisConfig,
        phonemizer: &dyn Phonemizer,
        mut on_chunk: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let chunks = prepare_synthesis_chunks(text, config.speed, config.max_chars)?;

        let mut rng = SimpleRng::from_seed(config.seed.unwrap_or_else(runtime_entropy_seed));
        let canonical_voice = self.resolve_voice(voice, &mut rng)?;
        let max_rows = self.voice_table.max_style_rows(&canonical_voice)?;
        let style_ref_id = resolve_style_index(max_rows, config.style_index, &mut rng);
        if voice.is_none() {
            eprintln!("using random voice: {canonical_voice}");
        }
        if config.style_index.is_none() {
            eprintln!("using random style-index: {style_ref_id}");
        }

        for chunk in chunks {
            let chunk_audio = self.synthesize_single_chunk(
                &chunk,
                &canonical_voice,
                config.speed,
                config.trim_tail,
                style_ref_id,
                phonemizer,
            )?;
            on_chunk(&chunk_audio)?;
        }

        Ok(())
    }

    fn resolve_voice(&self, voice: Option<&str>, rng: &mut SimpleRng) -> Result<String> {
        if let Some(voice) = voice {
            let resolved = self
                .voice_aliases
                .get(voice)
                .cloned()
                .unwrap_or_else(|| voice.to_string());

            if self.voice_table.voices.contains_key(&resolved) {
                return Ok(resolved);
            }

            bail!("voice '{voice}' is not available. Run `kitten-tts voices` to list voices");
        }

        let canonical = self.voice_table.canonical_voices();
        if canonical.is_empty() {
            bail!("no voices available in voice table");
        }

        let previous = load_last_random_voice(&self.random_voice_state_path);
        let mut candidates = canonical;
        if let Some(previous) = previous
            && candidates.len() > 1
        {
            candidates.retain(|v| v != &previous);
        }

        let selected = candidates[rng.gen_index(candidates.len())].clone();
        if let Err(err) = store_last_random_voice(&self.random_voice_state_path, &selected) {
            eprintln!(
                "warning: could not persist last random voice to {}: {err}",
                self.random_voice_state_path.display()
            );
        }
        Ok(selected)
    }

    fn synthesize_single_chunk(
        &mut self,
        chunk: &str,
        voice: &str,
        speed: f32,
        trim_tail: usize,
        style_ref_id: usize,
        phonemizer: &dyn Phonemizer,
    ) -> Result<Vec<f32>> {
        let mut tokens = tokenize_phonemes(
            &self.text_cleaner,
            &phonemizer.phonemize(chunk).with_context(|| {
                format!(
                    "failed phonemizing chunk using backend '{}': {chunk}",
                    phonemizer.name()
                )
            })?,
        );

        if tokens.is_empty() {
            bail!("tokenizer produced empty token sequence for chunk '{chunk}'");
        }
        if tokens.len() > MAX_INPUT_TOKENS {
            bail!(
                "phonemized chunk produced {} tokens (max {MAX_INPUT_TOKENS}). \
                 Reduce --max-chars to produce shorter chunks",
                tokens.len()
            );
        }

        tokens.insert(0, TOKEN_PAD_ID);
        tokens.push(TOKEN_PAD_ID);

        let style = self.voice_table.style_row(voice, style_ref_id)?.to_vec();
        let adjusted_speed = speed * self.speed_priors.get(voice).copied().unwrap_or(1.0);

        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens)
            .context("failed building input_ids tensor")?;
        let style = Array2::from_shape_vec((1, style.len()), style)
            .context("failed building style tensor")?;
        let speed = Array1::from_vec(vec![adjusted_speed]);

        let outputs = self.session.run(inputs![
            "input_ids" => Tensor::from_array(input_ids)?,
            "style" => Tensor::from_array(style)?,
            "speed" => Tensor::from_array(speed)?
        ])?;

        if outputs.len() == 0 {
            bail!("onnx model returned no output tensors");
        }
        let first_output = &outputs[0];
        let (_, data) = first_output
            .try_extract_tensor::<f32>()
            .context("failed extracting f32 output tensor")?;
        let mut audio = data.to_vec();

        if trim_tail >= audio.len() {
            audio.clear();
        } else {
            let keep = audio.len() - trim_tail;
            audio.truncate(keep);
        }

        Ok(audio)
    }
}

fn resolve_style_index(
    max_rows: usize,
    explicit_style_index: Option<usize>,
    rng: &mut SimpleRng,
) -> usize {
    if max_rows == 0 {
        return 0;
    }
    let max_index = max_rows.saturating_sub(1);
    match explicit_style_index {
        Some(i) => usize::min(i, max_index),
        None => rng.gen_index(max_rows),
    }
}

fn runtime_entropy_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_nanos();
    let pid = u64::from(std::process::id());
    (nanos as u64) ^ (nanos >> 64) as u64 ^ pid.rotate_left(13)
}

fn load_last_random_voice(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let voice = raw.trim();
    if voice.is_empty() {
        return None;
    }
    Some(voice.to_string())
}

fn store_last_random_voice(path: &Path, voice: &str) -> Result<()> {
    fs::write(path, voice)
        .with_context(|| format!("failed writing random voice state file {}", path.display()))
}

fn check_gpu_vram(device_id: i32, arena_mib: u64) -> Result<()> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            &format!("--id={device_id}"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            eprintln!("warning: nvidia-smi not found; skipping VRAM availability check");
            return Ok(());
        }
    };

    if !output.status.success() {
        eprintln!("warning: nvidia-smi failed; skipping VRAM availability check");
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let free_mib: u64 = match stdout.trim().parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("warning: could not parse nvidia-smi output; skipping VRAM check");
            return Ok(());
        }
    };

    let needed = arena_mib + VRAM_HEADROOM_MIB;
    if free_mib < needed {
        bail!(
            "insufficient GPU VRAM on device {device_id}: {free_mib} MiB free, \
             but ~{needed} MiB needed ({arena_mib} MiB arena + {VRAM_HEADROOM_MIB} MiB headroom). \
             Lower --cuda-memory-limit or close GPU-intensive applications to avoid a system crash"
        );
    }

    eprintln!("GPU device {device_id}: {free_mib} MiB free VRAM (need ~{needed} MiB)");
    Ok(())
}

fn resolve_ort_lib(config: &OrtRuntimeConfig) -> PathBuf {
    // Priority order:
    // 1. --ort-lib CLI argument
    // 2. ORT_DYLIB_PATH environment variable
    // 3. Bundled next to executable (for releases)
    // 4. System library search (for existing installations)

    if let Some(ref lib) = config.ort_lib {
        return lib.clone();
    }

    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from) {
        return path;
    }

    #[cfg(target_os = "windows")]
    let lib = "onnxruntime.dll";
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let lib = "libonnxruntime.so";
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let lib = "libonnxruntime.dylib";

    if let Ok(exe_path) = std::env::current_exe() {
        let mut bundled = exe_path;
        bundled.pop();
        bundled.push(lib);
        if bundled.exists() {
            return bundled;
        }
    }

    // Fall back to system library (let dynamic linker find it)
    PathBuf::from(lib)
}

fn init_ort(config: &OrtRuntimeConfig) -> Result<()> {
    ORT_INIT.get_or_try_init(|| -> Result<()> {
        if config.cuda_lib_dir.is_some() || config.cudnn_lib_dir.is_some() {
            ep::cuda::preload_dylibs(
                config.cuda_lib_dir.as_deref(),
                config.cudnn_lib_dir.as_deref(),
            )
            .context("failed preloading CUDA/cuDNN shared libraries")?;
        } else if config.ort_lib.is_none() {
            // Auto-preload bundled CUDA libraries when no explicit paths provided
            if let Ok(exe_path) = std::env::current_exe() {
                let mut cuda_dir = exe_path;
                cuda_dir.pop();
                if cuda_dir.join("libcublasLt.so.12").exists() {
                    ep::cuda::preload_dylibs(Some(cuda_dir.as_path()), None)
                        .context("failed preloading bundled CUDA libraries")?;
                }
            }
        }

        let ort_lib = resolve_ort_lib(config);

        let _ = ort::init_from(&ort_lib)
            .with_context(|| {
                format!(
                    "failed loading ONNX Runtime shared library from {}",
                    ort_lib.display()
                )
            })?
            .with_name("kitten-tts")
            .commit();

        Ok(())
    })?;
    Ok(())
}

fn tokenize_phonemes(cleaner: &TextCleaner, phonemes: &str) -> Vec<i64> {
    let mut tokens = Vec::new();
    let collapsed = SPACES_RE.replace_all(phonemes.trim(), " ");
    cleaner.encode_into(&collapsed, &mut tokens);
    tokens
}

fn clean_text_basic(text: &str) -> String {
    SPACES_RE.replace_all(text.trim(), " ").to_string()
}

fn ensure_punctuation(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if matches!(
        trimmed.chars().last(),
        Some('.' | '!' | '?' | ',' | ';' | ':')
    ) {
        trimmed.to_string()
    } else {
        format!("{trimmed},")
    }
}

fn prepare_synthesis_chunks(text: &str, speed: f32, max_chars: usize) -> Result<Vec<String>> {
    if !speed.is_finite() || !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
        bail!("speed must be between {MIN_SPEED} and {MAX_SPEED}");
    }
    if max_chars == 0 || max_chars > MAX_CHARS_HARD_LIMIT {
        bail!(
            "max_chars must be between 1 and {MAX_CHARS_HARD_LIMIT} to avoid unsafe runtime load"
        );
    }
    let cleaned = clean_text_basic(text);
    if cleaned.is_empty() {
        bail!("input text is empty after basic normalization");
    }
    if cleaned.len() > MAX_TOTAL_INPUT_CHARS {
        bail!(
            "input text too long ({} chars). Max allowed is {} for runtime safety",
            cleaned.len(),
            MAX_TOTAL_INPUT_CHARS
        );
    }

    let chunks = chunk_text(&cleaned, max_chars);
    if chunks.is_empty() {
        bail!("input text produced zero chunks");
    }
    if chunks.len() > MAX_SYNTHESIS_CHUNKS {
        bail!(
            "input produced too many chunks ({}). Max allowed is {} for runtime safety",
            chunks.len(),
            MAX_SYNTHESIS_CHUNKS
        );
    }
    Ok(chunks)
}

fn prepare_native_chunks(text: &str, speed: f32, max_chars: usize) -> Result<Vec<String>> {
    let chunks = prepare_synthesis_chunks(text, speed, max_chars)?;
    let cleaned = clean_text_basic(text);
    if cleaned.len() <= max_chars {
        Ok(vec![cleaned])
    } else {
        Ok(chunks)
    }
}

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut in_punct = false;

    for c in text.chars() {
        let is_punct = c == '.' || c == '!' || c == '?';
        if is_punct {
            current_sentence.push(c);
            in_punct = true;
        } else {
            if in_punct {
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
                in_punct = false;
            }
            current_sentence.push(c);
        }
    }
    let final_sentence = current_sentence.trim();
    if !final_sentence.is_empty() {
        sentences.push(final_sentence.to_string());
    }

    for sentence in sentences {
        if sentence.is_empty() {
            continue;
        }

        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(&sentence));
            continue;
        }

        let mut current = String::new();
        for word in sentence.split_whitespace() {
            if current.is_empty() {
                current.push_str(word);
                continue;
            }
            if current.len() + 1 + word.len() <= max_len {
                current.push(' ');
                current.push_str(word);
            } else {
                chunks.push(ensure_punctuation(&current));
                current.clear();
                current.push_str(word);
            }
        }

        if !current.is_empty() {
            chunks.push(ensure_punctuation(&current));
        }
    }

    chunks
}

fn safe_file_name(value: &str) -> Result<&str> {
    let path = Path::new(value);
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => bail!("unsafe file path in model config: '{value}'"),
        }
    }

    if value.contains('/') || value.contains('\\') {
        bail!("unsafe nested file path in model config: '{value}'");
    }
    Ok(value)
}

fn cache_model_dir(cache_root: &Path, repo_id: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(repo_id.as_bytes());
    let digest = hasher.finalize();
    let short = hex::encode(&digest[..8]);
    let sanitized: String = repo_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();

    cache_root
        .join("models")
        .join(format!("{sanitized}_{short}"))
}

fn ensure_model_cached(
    client: &Client,
    cache_root: &Path,
    repo_id: &str,
    force: bool,
) -> Result<ModelArtifacts> {
    let model_dir = cache_model_dir(cache_root, repo_id);
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("failed creating cache dir {}", model_dir.display()))?;
    let mut cache_updated = false;

    let config_path = model_dir.join("config.json");
    if force || !config_path.exists() {
        download_repo_file(client, repo_id, "config.json", &config_path)
            .with_context(|| format!("failed downloading config.json for {repo_id}"))?;
        cache_updated = true;
    }

    let config: ModelConfig = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .with_context(|| format!("failed reading config at {}", config_path.display()))?,
    )
    .with_context(|| format!("failed parsing config at {}", config_path.display()))?;

    if config.model_type != "ONNX1" && config.model_type != "ONNX2" {
        bail!(
            "unsupported model type '{}': expected ONNX1 or ONNX2",
            config.model_type
        );
    }

    let model_file = safe_file_name(&config.model_file)?;
    let voices_file = safe_file_name(&config.voices)?;

    let model_path = model_dir.join(model_file);
    let voices_path = model_dir.join(voices_file);

    if force || !model_path.exists() {
        download_repo_file(client, repo_id, model_file, &model_path)
            .with_context(|| format!("failed downloading model file {model_file}"))?;
        cache_updated = true;
    }

    if force || !voices_path.exists() {
        download_repo_file(client, repo_id, voices_file, &voices_path)
            .with_context(|| format!("failed downloading voices file {voices_file}"))?;
        cache_updated = true;
    }

    let manifest_path = model_dir.join("manifest.json");
    if cache_updated || !manifest_path.exists() {
        let manifest = CacheManifest {
            repo_id: repo_id.to_string(),
            fetched_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model_file: model_file.to_string(),
            voices_file: voices_file.to_string(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, manifest_json)
            .with_context(|| format!("failed writing manifest at {}", manifest_path.display()))?;
    }

    Ok(ModelArtifacts {
        repo_id: repo_id.to_string(),
        cache_dir: model_dir,
        config_path,
        model_path,
        voices_path,
        config,
    })
}

fn download_repo_file(client: &Client, repo_id: &str, file_name: &str, dest: &Path) -> Result<()> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{file_name}?download=true");

    let temp_path = dest.with_extension("download.tmp");
    let result = (|| -> Result<()> {
        let mut response = client
            .get(&url)
            .send()
            .with_context(|| format!("http request failed for {url}"))?
            .error_for_status()
            .with_context(|| format!("download failed for {url}"))?;

        let mut file = File::create(&temp_path)
            .with_context(|| format!("failed creating temp file {}", temp_path.display()))?;
        io::copy(&mut response, &mut file)
            .with_context(|| format!("failed writing to temp file {}", temp_path.display()))?;
        file.flush()?;

        fs::rename(&temp_path, dest).with_context(|| {
            format!(
                "failed moving temp file {} to {}",
                temp_path.display(),
                dest.display()
            )
        })?;

        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn read_text_input(text_input: &TextInput) -> Result<String> {
    if let Some(text) = &text_input.text {
        return Ok(text.clone());
    }

    if let Some(path) = &text_input.text_file {
        return fs::read_to_string(path)
            .with_context(|| format!("failed reading text file {}", path.display()));
    }

    if io::stdin().is_terminal() {
        bail!("provide input text via --text, --text-file, or pipe stdin");
    }

    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn write_wav(path: &Path, sample_rate: u32, encoding: WavEncoding, audio: &[f32]) -> Result<()> {
    let spec = match encoding {
        WavEncoding::Pcm16 => hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
        WavEncoding::Float32 => hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed creating wav file {}", path.display()))?;

    match encoding {
        WavEncoding::Pcm16 => {
            for sample in audio {
                let clamped = sample.clamp(-1.0, 1.0);
                let scaled = (clamped * i16::MAX as f32) as i16;
                writer.write_sample(scaled)?;
            }
        }
        WavEncoding::Float32 => {
            for sample in audio {
                writer.write_sample(*sample)?;
            }
        }
    }

    writer.finalize()?;
    Ok(())
}

fn apply_playback_gain(audio: &[f32], gain: f32, allow_clipping: bool) -> Result<(Vec<f32>, f32)> {
    if !gain.is_finite() || !(MIN_OUTPUT_GAIN..=MAX_OUTPUT_GAIN).contains(&gain) {
        bail!("gain must be between {MIN_OUTPUT_GAIN} and {MAX_OUTPUT_GAIN}");
    }

    let peak = audio
        .iter()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
    let mut applied_gain = gain;
    if !allow_clipping && peak > 0.0 {
        let max_gain_without_clipping = SAFE_PLAYBACK_PEAK / peak;
        if applied_gain > max_gain_without_clipping {
            applied_gain = max_gain_without_clipping;
        }
    }

    let gained = if (applied_gain - 1.0).abs() < f32::EPSILON {
        audio.to_vec()
    } else {
        audio.iter().map(|sample| *sample * applied_gain).collect()
    };

    Ok((gained, applied_gain))
}

fn play_audio(
    audio: &[f32],
    sample_rate: u32,
    player_mode: PlayerMode,
    gain: f32,
    allow_clipping: bool,
) -> Result<()> {
    let player = detect_player(player_mode)?;
    let (playback_audio, applied_gain) = apply_playback_gain(audio, gain, allow_clipping)?;
    if !allow_clipping && applied_gain + f32::EPSILON < gain {
        eprintln!(
            "requested gain {:.2} was limited to {:.2} to avoid clipping",
            gain, applied_gain
        );
    }

    let temp_file = tempfile::Builder::new()
        .prefix("kitten-tts-play-")
        .suffix(".wav")
        .tempfile()
        .context("failed creating temporary wav file for playback")?;
    let temp_path = temp_file.path();

    write_wav(temp_path, sample_rate, WavEncoding::Pcm16, &playback_audio)?;
    let status = match player {
        "ffplay" => Command::new("ffplay")
            .args(["-autoexit", "-nodisp", "-loglevel", "error"])
            .arg(temp_path)
            .status()
            .context("failed to launch ffplay")?,
        "pw-play" => Command::new("pw-play")
            .arg(temp_path)
            .status()
            .context("failed to launch pw-play")?,
        _ => bail!("unsupported player selected"),
    };
    if !status.success() {
        bail!("audio player exited with failure status: {status}");
    }
    Ok(())
}

fn parse_npy_f32(bytes: &[u8]) -> Result<(usize, usize, Vec<f32>)> {
    if bytes.len() < 12 {
        bail!("npy payload too small");
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        bail!("invalid npy magic header");
    }

    let major = bytes[6];
    let (header_len, header_offset) = match major {
        1 => {
            let h = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (h, 10_usize)
        }
        2 | 3 => {
            let h = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (h, 12_usize)
        }
        other => bail!("unsupported npy version {other}"),
    };

    let header_end = header_offset
        .checked_add(header_len)
        .ok_or_else(|| anyhow!("npy header length overflow"))?;
    if bytes.len() < header_end {
        bail!("npy header length exceeds payload size");
    }
    let header = std::str::from_utf8(&bytes[header_offset..header_end])
        .context("npy header is not valid utf-8")?;

    let descr = parse_header_field(header, "descr")
        .ok_or_else(|| anyhow!("npy header missing 'descr' field"))?;
    let fortran = parse_header_field(header, "fortran_order")
        .ok_or_else(|| anyhow!("npy header missing 'fortran_order' field"))?;
    let shape = parse_shape(header).ok_or_else(|| anyhow!("npy header missing 'shape' field"))?;

    if descr != "<f4" {
        bail!("unsupported npy dtype '{descr}', expected '<f4'");
    }
    if fortran != "False" {
        bail!("unsupported npy order '{fortran}', expected 'False'");
    }
    if shape.len() != 2 {
        bail!("expected 2D style tensor shape, got {shape:?}");
    }

    let rows = shape[0];
    let cols = shape[1];
    let item_count = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("npy shape overflow for {rows}x{cols}"))?;
    let expected_data_bytes = item_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("npy byte size overflow for {rows}x{cols}"))?;

    let data_bytes = &bytes[header_end..];
    if data_bytes.len() != expected_data_bytes {
        bail!(
            "npy data size mismatch: expected {} bytes, got {}",
            expected_data_bytes,
            data_bytes.len()
        );
    }

    let mut data = Vec::with_capacity(item_count);
    for chunk in data_bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok((rows, cols, data))
}

fn parse_header_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("'{key}':");
    let start = header.find(&pattern)? + pattern.len();
    let rest = header[start..].trim_start();

    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(&stripped[..end]);
    }
    if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        return Some(&stripped[..end]);
    }

    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let marker = "'shape':";
    let start = header.find(marker)? + marker.len();
    let rest = header[start..].trim_start();
    let tuple_start = rest.find('(')?;
    let tuple_rest = &rest[tuple_start + 1..];
    let tuple_end = tuple_rest.find(')')?;
    let shape_text = &tuple_rest[..tuple_end];

    let mut shape = Vec::new();
    for part in shape_text.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        shape.push(trimmed.parse::<usize>().ok()?);
    }

    Some(shape)
}

fn resolve_cache_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    let dirs = ProjectDirs::from("io", "KittenML", "kitten-tts")
        .ok_or_else(|| anyhow!("could not determine platform cache directory"))?;
    Ok(dirs.cache_dir().to_path_buf())
}

fn build_http_client() -> Result<Client> {
    Client::builder()
        .user_agent("kitten-tts-rs/0.1.0")
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .build()
        .context("failed building HTTP client")
}

enum SynthesisBackend {
    Onnx {
        synthesizer: Synthesizer,
        phonemizer: Box<dyn Phonemizer>,
    },
    Native(NativeSynth),
}

impl SynthesisBackend {
    fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        config: SynthesisConfig,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Onnx {
                synthesizer,
                phonemizer,
            } => synthesizer.synthesize(text, voice, config, phonemizer.as_ref()),
            Self::Native(synthesizer) => synthesizer.synthesize(text, voice, config),
        }
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        config: SynthesisConfig,
        on_chunk: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Onnx {
                synthesizer,
                phonemizer,
            } => {
                synthesizer.synthesize_streaming(text, voice, config, phonemizer.as_ref(), on_chunk)
            }
            Self::Native(synthesizer) => {
                synthesizer.synthesize_streaming(text, voice, config, on_chunk)
            }
        }
    }
}

fn prepare_synthesis_runtime(args: &SynthesisArgs) -> Result<(String, SynthesisBackend)> {
    let text = read_text_input(&args.text)?;

    match args.engine {
        EngineChoice::Onnx => Ok((text, prepare_onnx_backend(args)?)),
        EngineChoice::Native => Ok((text, prepare_native_backend(args)?)),
        EngineChoice::Auto => match auto_native_intent(args) {
            AutoNativeIntent::Required => Ok((text, prepare_native_backend(args)?)),
            AutoNativeIntent::Opportunistic => match prepare_native_backend(args) {
                Ok(backend) => Ok((text, backend)),
                Err(err) => {
                    eprintln!(
                        "note: native engine unavailable ({err:#}); falling back to ONNX Runtime"
                    );
                    Ok((text, prepare_onnx_backend(args)?))
                }
            },
            AutoNativeIntent::None => Ok((text, prepare_onnx_backend(args)?)),
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoNativeIntent {
    Required,
    Opportunistic,
    None,
}

fn auto_native_intent(args: &SynthesisArgs) -> AutoNativeIntent {
    if args.native_model.is_some()
        || args.native_lib.is_some()
        || args.native_family == NativeFamily::KittenTts2
        || args.native_espeak_lib.is_some()
        || args.native_espeak_data.is_some()
    {
        AutoNativeIntent::Required
    } else if args.model.model == Some(ModelPreset::Mini08) {
        AutoNativeIntent::Opportunistic
    } else {
        AutoNativeIntent::None
    }
}

fn prepare_onnx_backend(args: &SynthesisArgs) -> Result<SynthesisBackend> {
    let client = build_http_client()?;
    let cache_root = args.model.resolve_cache_dir()?;
    let repo_id = args.model.resolve_repo_id();
    let ort_runtime = args.model.resolve_ort_runtime_config();
    let artifacts = ensure_model_cached(&client, &cache_root, &repo_id, false)?;
    let synthesizer = Synthesizer::new(&artifacts, &ort_runtime)?;
    let phonemizer = detect_phonemizer(args.phonemizer)?;
    Ok(SynthesisBackend::Onnx {
        synthesizer,
        phonemizer,
    })
}

fn prepare_native_backend(args: &SynthesisArgs) -> Result<SynthesisBackend> {
    validate_native_args(args)?;

    if args.phonemizer != PhonemizerMode::EspeakNg {
        eprintln!(
            "note: the native engine uses its built-in eSpeak-ng frontend; --phonemizer is ignored"
        );
    }
    if args.trim_tail != DEFAULT_TRIM_TAIL {
        eprintln!("note: --trim-tail is ONNX-only and is ignored by the native engine");
    }
    if args.native_family == NativeFamily::KittenTts2 && (args.speed - 1.0).abs() > 1e-6 {
        eprintln!("note: the kitten_tts2 port does not implement --speed; using its default rate");
    }

    if args.native_model.is_none() && args.model.model.is_none() {
        eprintln!(
            "[Native] using KittenTTS Mini 0.8, the only packaged native model (the ONNX default preset does not apply)"
        );
    }

    let library_path = resolve_native_lib_path(args.native_lib.as_deref());
    let cache_root = args.model.resolve_cache_dir()?;
    let client = build_http_client()?;
    let model_path = ensure_native_model_cached(&client, &cache_root, args)?;
    let (espeak_lib, espeak_data) = if args.native_family == NativeFamily::KittenTts {
        (
            resolve_espeak_path(
                args.native_espeak_lib.as_deref(),
                detect_espeak_library,
                "libespeak-ng",
            )?,
            resolve_espeak_path(
                args.native_espeak_data.as_deref(),
                detect_espeak_data,
                "espeak-ng-data",
            )?,
        )
    } else {
        if args.native_espeak_lib.is_some() || args.native_espeak_data.is_some() {
            eprintln!(
                "note: --native-espeak-lib/--native-espeak-data only apply to kitten_tts; ignored"
            );
        }
        (None, None)
    };
    let load_started = Instant::now();
    let synthesizer = NativeSynth::open(NativeOpenConfig {
        library_path: &library_path,
        family: args.native_family,
        backend: args.native_backend,
        threads: args.native_threads,
        model_path: &model_path,
        cache_root: &cache_root,
        espeak_lib: espeak_lib.as_deref(),
        espeak_data: espeak_data.as_deref(),
    })?;
    eprintln!(
        "[Native] session ready in {} ms",
        load_started.elapsed().as_millis()
    );

    Ok(SynthesisBackend::Native(synthesizer))
}

fn validate_native_args(args: &SynthesisArgs) -> Result<()> {
    if !(1..=256).contains(&args.native_threads) {
        bail!("--native-threads must be between 1 and 256");
    }
    if args.style_index.is_some() {
        bail!("--style-index is not supported by the native engine; omit it or pass --engine onnx");
    }
    if args.model.vram_profile.is_some() {
        bail!("--vram-profile is ONNX-only; omit it or pass --engine onnx");
    }
    if args.model.repo_id.is_some() {
        bail!(
            "--repo-id selects an ONNX repository and cannot be combined with the native engine; pass --native-model PATH instead"
        );
    }
    if args.native_model.is_none()
        && let Some(preset) = args.model.model
        && preset != ModelPreset::Mini08
    {
        bail!(
            "the native engine currently provides only the KittenTTS Mini 0.8 package; pass --model mini-0.8 or --native-model PATH"
        );
    }
    if args.native_family == NativeFamily::KittenTts2 {
        if !args.accept_kitten2_license {
            bail!(
                "KittenTTS 2 weights use the Stellon Labs Community License (commercial registration and revenue limits). Pass --accept-kitten2-license to confirm, and see docs/native-backend-evaluation.md"
            );
        }
        if args.native_model.is_none() {
            bail!(
                "KittenTTS 2 weights are not downloaded automatically; pass --native-model PATH to a local GGUF"
            );
        }
    }
    Ok(())
}

fn resolve_espeak_path(
    explicit: Option<&Path>,
    detect: fn() -> Option<PathBuf>,
    label: &str,
) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        if !path.exists() {
            bail!("{label} path '{}' does not exist", path.display());
        }
        return Ok(Some(path.to_path_buf()));
    }
    Ok(detect())
}

fn detect_espeak_library() -> Option<PathBuf> {
    const CANDIDATES: [&str; 6] = [
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so",
    ];
    CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
}

fn detect_espeak_data() -> Option<PathBuf> {
    const CANDIDATES: [&str; 3] = [
        "/opt/homebrew/share/espeak-ng-data",
        "/usr/local/share/espeak-ng-data",
        "/usr/share/espeak-ng-data",
    ];
    CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_dir())
}

fn ensure_native_model_cached(
    client: &Client,
    cache_root: &Path,
    args: &SynthesisArgs,
) -> Result<PathBuf> {
    if let Some(path) = &args.native_model {
        if !path.is_file() {
            bail!(
                "native model '{}' does not exist or is not a file",
                path.display()
            );
        }
        return Ok(path.clone());
    }

    ensure_native_kitten_tts_model(client, cache_root, false)
}

fn ensure_native_kitten_tts_model(
    client: &Client,
    cache_root: &Path,
    force: bool,
) -> Result<PathBuf> {
    let native_dir = cache_root.join("native");
    fs::create_dir_all(&native_dir)
        .with_context(|| format!("failed creating native cache dir {}", native_dir.display()))?;
    let destination = native_dir.join(NATIVE_KITTEN_TTS_FILE);
    if force || !destination.is_file() {
        let remote = format!("{NATIVE_KITTEN_TTS_DIR}/{NATIVE_KITTEN_TTS_FILE}");
        eprintln!("[Native] downloading {NATIVE_MODEL_REPO}/{remote} (about 288 MiB)");
        download_repo_file(client, NATIVE_MODEL_REPO, &remote, &destination)
            .with_context(|| format!("failed downloading native model {remote}"))?;
    }
    Ok(destination)
}

fn fetch_native_model(args: &NativeFetchArgs) -> Result<()> {
    if args.family == NativeFamily::KittenTts2 {
        bail!(
            "KittenTTS 2 weights are not downloaded automatically because of their license (commercial registration and revenue limits). Download a GGUF from https://huggingface.co/{NATIVE_MODEL_REPO}/tree/main/KittenTTS2-GGUF and pass it with --native-model, plus --accept-kitten2-license when synthesizing"
        );
    }

    let cache_root = resolve_cache_root(args.cache_dir.as_deref())?;
    let client = build_http_client()?;
    let path = ensure_native_kitten_tts_model(&client, &cache_root, args.force)?;
    let size = fs::metadata(&path)
        .with_context(|| format!("failed reading model metadata at {}", path.display()))?
        .len();

    println!("Native package: {}", args.family.as_str());
    println!("Model: {}", path.display());
    println!("Size: {size} bytes");
    println!(
        "Source: https://huggingface.co/{NATIVE_MODEL_REPO}/resolve/main/{NATIVE_KITTEN_TTS_DIR}/{NATIVE_KITTEN_TTS_FILE}"
    );
    Ok(())
}

type NativeAbiVersionFn = unsafe extern "C" fn() -> u32;
type NativeStringFn = unsafe extern "C" fn() -> *const c_char;
type NativeStatusStringFn = unsafe extern "C" fn(c_int) -> *const c_char;
type NativeRegistryCreateFn = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int;
type NativeRegistryFreeFn = unsafe extern "C" fn(*mut c_void);
type NativeFamilyCountFn = unsafe extern "C" fn(*const c_void) -> usize;
type NativeFamilyFn = unsafe extern "C" fn(*const c_void, usize, *mut *const c_char) -> c_int;
type NativeTaskCountFn = unsafe extern "C" fn() -> usize;
type NativeTaskNameFn = unsafe extern "C" fn(usize) -> *const c_char;

struct NativeRuntime {
    _library: Library,
    abi_version: NativeAbiVersionFn,
    build_version: NativeStringFn,
    last_error: NativeStringFn,
    status_string: NativeStatusStringFn,
    registry_create: NativeRegistryCreateFn,
    registry_free: NativeRegistryFreeFn,
    family_count: NativeFamilyCountFn,
    family: NativeFamilyFn,
    task_count: NativeTaskCountFn,
    task_name: NativeTaskNameFn,
}

impl NativeRuntime {
    fn load(path: &Path) -> Result<Self> {
        let library = open_native_library(path)?;
        let abi_version: NativeAbiVersionFn = load_native_symbol(&library, "audiocpp_abi_version")?;

        Ok(Self {
            abi_version,
            build_version: load_native_symbol(&library, "audiocpp_build_version")?,
            last_error: load_native_symbol(&library, "audiocpp_last_error")?,
            status_string: load_native_symbol(&library, "audiocpp_status_string")?,
            registry_create: load_native_symbol(&library, "audiocpp_registry_create")?,
            registry_free: load_native_symbol(&library, "audiocpp_registry_free")?,
            family_count: load_native_symbol(&library, "audiocpp_registry_family_count")?,
            family: load_native_symbol(&library, "audiocpp_registry_family")?,
            task_count: load_native_symbol(&library, "audiocpp_task_count")?,
            task_name: load_native_symbol(&library, "audiocpp_task_name")?,
            _library: library,
        })
    }

    fn describe_status(&self, status: c_int) -> String {
        let message = borrowed_string(unsafe { (self.last_error)() });
        let label = borrowed_string(unsafe { (self.status_string)(status) });
        if message.is_empty() {
            format!("{label} ({status})")
        } else {
            format!("{message} ({label})")
        }
    }

    fn families(&self) -> Result<Vec<String>> {
        let mut registry: *mut c_void = std::ptr::null_mut();
        let status = unsafe { (self.registry_create)(std::ptr::null(), &mut registry) };
        if status != AUDIOCPP_OK {
            bail!(
                "failed to create native registry: {}",
                self.describe_status(status)
            );
        }

        let collected = (|| -> Result<Vec<String>> {
            let count = unsafe { (self.family_count)(registry) };
            if count > MAX_NATIVE_FAMILIES {
                bail!("native library reported {count} families; refusing to enumerate");
            }

            let mut families = Vec::with_capacity(count);
            for index in 0..count {
                let mut name: *const c_char = std::ptr::null();
                let status = unsafe { (self.family)(registry, index, &mut name) };
                if status != AUDIOCPP_OK {
                    bail!(
                        "failed to read native family at index {index}: {}",
                        self.describe_status(status)
                    );
                }
                families.push(borrowed_string(name));
            }
            Ok(families)
        })();

        unsafe { (self.registry_free)(registry) };
        collected
    }

    fn tasks(&self) -> Vec<String> {
        let count = unsafe { (self.task_count)() }.min(MAX_NATIVE_TASKS);
        (0..count)
            .map(|index| borrowed_string(unsafe { (self.task_name)(index) }))
            .filter(|name| !name.is_empty())
            .collect()
    }
}

fn open_native_library(path: &Path) -> Result<Library> {
    // SAFETY: the library path is user-supplied. Symbols are loaded with the exact
    // C signatures from audiocpp.h, and the ABI major is checked here before any
    // other entry point is resolved or called.
    let library = unsafe { Library::new(path) }.with_context(|| {
        format!(
            "failed to load native audio.cpp library at '{}'; pass --native-lib PATH or set {AUDIOCPP_LIB_ENV}",
            path.display()
        )
    })?;

    let abi_version: NativeAbiVersionFn = load_native_symbol(&library, "audiocpp_abi_version")?;
    ensure_compatible_abi(abi_parts(unsafe { abi_version() }).0)?;
    Ok(library)
}

fn load_native_symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T> {
    // SAFETY: the caller supplies the matching C signature for this symbol.
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .with_context(|| format!("native audio.cpp library is missing symbol '{name}'"))
}

fn borrowed_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: audiocpp.h guarantees returned strings are NUL-terminated and borrowed
    // until the producing handle is mutated or freed; this copies immediately.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn abi_parts(version: u32) -> (u32, u32, u32) {
    (
        (version >> 16) & 0xff,
        (version >> 8) & 0xff,
        version & 0xff,
    )
}

fn ensure_compatible_abi(major: u32) -> Result<()> {
    if major != AUDIOCPP_ABI_MAJOR {
        bail!(
            "native library ABI major {major} is incompatible with this build (expected {AUDIOCPP_ABI_MAJOR}); install a matching audio.cpp build"
        );
    }
    Ok(())
}

fn default_native_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "audiocpp.dll"
    } else if cfg!(target_os = "macos") {
        "libaudiocpp.dylib"
    } else {
        "libaudiocpp.so"
    }
}

fn resolve_native_lib_path(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Some(path) = std::env::var_os(AUDIOCPP_LIB_ENV) {
        return PathBuf::from(path);
    }
    PathBuf::from(default_native_lib_name())
}

fn probe_native_runtime(args: &NativeProbeArgs) -> Result<()> {
    let library_path = resolve_native_lib_path(args.native_lib.as_deref());
    let runtime = NativeRuntime::load(&library_path)?;

    let (major, minor, patch) = abi_parts(unsafe { (runtime.abi_version)() });

    let mut report = vec![
        format!("Library: {}", library_path.display()),
        format!(
            "audio.cpp build: {}",
            borrowed_string(unsafe { (runtime.build_version)() })
        ),
        format!("ABI: {major}.{minor}.{patch}"),
    ];

    let mut families = runtime.families()?;
    families.sort();
    report.push(format!(
        "Families ({}): {}",
        families.len(),
        families.join(", ")
    ));
    for family in ["kitten_tts", "kitten_tts2"] {
        report.push(format!(
            "{family}: {}",
            if families.iter().any(|name| name == family) {
                "available"
            } else {
                "not available"
            }
        ));
    }

    let tasks = runtime.tasks();
    report.push(format!("Tasks ({}): {}", tasks.len(), tasks.join(", ")));
    write_stdout(&(report.join("\n") + "\n"))?;

    if let Some(requested) = &args.family
        && !families.iter().any(|name| name == requested)
    {
        let available = if families.is_empty() {
            "none".to_string()
        } else {
            families.join(", ")
        };
        bail!(
            "family '{requested}' is not available in this native library (available: {available})"
        );
    }

    Ok(())
}

#[derive(Default)]
struct NativeDoctorReport {
    lines: Vec<String>,
    failures: Vec<String>,
}

impl NativeDoctorReport {
    fn ok(&mut self, section: &str, detail: String) {
        self.lines.push(format!("{section}: ok - {detail}"));
    }

    fn failed(&mut self, section: &str, detail: String) {
        self.lines.push(format!("{section}: FAILED - {detail}"));
        self.failures.push(format!("{section}: {detail}"));
    }
}

struct NativeDoctorLibrary {
    build: String,
    abi: (u32, u32, u32),
    families: Vec<String>,
}

fn check_native_doctor_library(library_path: &Path) -> Result<NativeDoctorLibrary> {
    let runtime = NativeRuntime::load(library_path)?;
    let abi = abi_parts(unsafe { (runtime.abi_version)() });
    let build = borrowed_string(unsafe { (runtime.build_version)() });
    let families = runtime.families()?;
    Ok(NativeDoctorLibrary {
        build,
        abi,
        families,
    })
}

fn check_native_doctor_model(
    family: NativeFamily,
    explicit: Option<&Path>,
    cache_root: &Path,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.is_file() {
            bail!(
                "native model '{}' does not exist or is not a file",
                path.display()
            );
        }
        return Ok(path.to_path_buf());
    }

    match family {
        NativeFamily::KittenTts2 => bail!(
            "KittenTTS 2 weights are not downloaded automatically; pass --native-model PATH to a local GGUF"
        ),
        NativeFamily::KittenTts => {
            let path = cache_root.join("native").join(NATIVE_KITTEN_TTS_FILE);
            if !path.is_file() {
                bail!("cached model is missing at {}", path.display());
            }
            Ok(path)
        }
    }
}

fn doctor_native_runtime(args: &NativeDoctorArgs) -> Result<()> {
    let family = args.family.as_str();
    let cache_root = resolve_cache_root(args.cache_dir.as_deref())?;
    let library_path = resolve_native_lib_path(args.native_lib.as_deref());
    let mut report = NativeDoctorReport::default();

    match check_native_doctor_library(&library_path) {
        Ok(library) => {
            let (major, minor, patch) = library.abi;
            let detail = format!(
                "audio.cpp {}, ABI {major}.{minor}.{patch}, {}",
                library.build,
                library_path.display()
            );
            if library.families.iter().any(|name| name == family) {
                report.ok("library", format!("family {family} available; {detail}"));
            } else {
                report.failed(
                    "library",
                    format!("family {family} is not available; {detail}"),
                );
            }
        }
        Err(err) => report.failed("library", format!("{err:#}")),
    }

    if args.family == NativeFamily::KittenTts {
        match resolve_espeak_path(
            args.native_espeak_lib.as_deref(),
            detect_espeak_library,
            "libespeak-ng",
        ) {
            Ok(Some(path)) => report.ok("espeak lib", path.display().to_string()),
            Ok(None) => report.failed(
                "espeak lib",
                "libespeak-ng was not found in the usual locations".to_string(),
            ),
            Err(err) => report.failed("espeak lib", format!("{err:#}")),
        }
        match resolve_espeak_path(
            args.native_espeak_data.as_deref(),
            detect_espeak_data,
            "espeak-ng-data",
        ) {
            Ok(Some(path)) => report.ok("espeak data", path.display().to_string()),
            Ok(None) => report.failed(
                "espeak data",
                "espeak-ng-data was not found in the usual locations".to_string(),
            ),
            Err(err) => report.failed("espeak data", format!("{err:#}")),
        }
    } else {
        report.ok("espeak", "not required by this family".to_string());
    }

    match check_native_doctor_model(args.family, args.native_model.as_deref(), &cache_root) {
        Ok(path) => {
            let size = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            report.ok("model", format!("{} ({size} bytes)", path.display()));
        }
        Err(err) => report.failed("model", format!("{err:#}")),
    }

    let mut output = String::from("kitten-tts native doctor\n");
    for line in &report.lines {
        output.push_str(&format!("  {line}\n"));
    }
    if report.failures.is_empty() {
        output.push_str("Result: ready for --engine native\n");
    } else {
        let mut hints: Vec<&str> = Vec::new();
        for failure in &report.failures {
            let hint = if failure.starts_with("library:") {
                "pass --native-lib PATH or set AUDIOCPP_LIB"
            } else if failure.starts_with("espeak") {
                "install espeak-ng or pass --native-espeak-lib/--native-espeak-data PATH"
            } else if args.family == NativeFamily::KittenTts {
                "run kitten-tts native fetch or pass --native-model PATH"
            } else {
                "pass --native-model PATH to a local GGUF"
            };
            if !hints.contains(&hint) {
                hints.push(hint);
            }
        }
        output.push_str(&format!(
            "Result: {} check(s) failed; {}\n",
            report.failures.len(),
            hints.join("; ")
        ));
    }
    write_stdout(&output)?;

    if !report.failures.is_empty() {
        bail!("native environment is not ready; fix the FAILED checks above");
    }
    Ok(())
}

#[repr(C)]
struct NativeModelConfig {
    family_hint: *const c_char,
    config_id: *const c_char,
    weight_id: *const c_char,
    model_spec_override: *const c_char,
}

#[repr(C)]
struct NativeBackendConfig {
    backend: *const c_char,
    device: c_int,
    threads: c_int,
}

type NativeModelLoadFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const NativeModelConfig,
    *const c_void,
    *mut *mut c_void,
) -> c_int;
type NativeSessionCreateFn = unsafe extern "C" fn(
    *const c_void,
    *const c_char,
    *const c_char,
    *const NativeBackendConfig,
    *const c_void,
    *mut *mut c_void,
) -> c_int;
type NativeRunFn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> c_int;
type NativeResultAudioFn = unsafe extern "C" fn(
    *const c_void,
    *mut *const f32,
    *mut usize,
    *mut c_int,
    *mut c_int,
) -> c_int;
type NativeVoidHandleFn = unsafe extern "C" fn(*mut c_void);
type NativeHandleFn = unsafe extern "C" fn() -> *mut c_void;
type NativeRequestSetTextFn =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int;
type NativeRequestSetStrFn = unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int;
type NativeRequestSetRateFn = unsafe extern "C" fn(*mut c_void, f32) -> c_int;
type NativeRequestSetOptionFn =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int;

struct NativeSynth {
    _library: Library,
    last_error: NativeStringFn,
    status_string: NativeStatusStringFn,
    registry_create: NativeRegistryCreateFn,
    registry_free: NativeRegistryFreeFn,
    model_load: NativeModelLoadFn,
    model_free: NativeVoidHandleFn,
    session_create: NativeSessionCreateFn,
    session_free: NativeVoidHandleFn,
    session_run: NativeRunFn,
    request_create: NativeHandleFn,
    request_free: NativeVoidHandleFn,
    request_set_text: NativeRequestSetTextFn,
    request_set_voice: NativeRequestSetStrFn,
    request_set_rate: NativeRequestSetRateFn,
    request_set_option: NativeRequestSetOptionFn,
    result_audio: NativeResultAudioFn,
    result_free: NativeVoidHandleFn,
    options_create: NativeHandleFn,
    options_set: NativeRequestSetOptionFn,
    options_free: NativeVoidHandleFn,
    registry: *mut c_void,
    options: *mut c_void,
    model: *mut c_void,
    session: *mut c_void,
    family: NativeFamily,
    random_voice_state_path: PathBuf,
    backend: &'static str,
    auto_backend: bool,
    threads: i32,
}

#[derive(Debug, Clone, Copy)]
enum NativeAudioError {
    NonFinite(usize),
    Silent,
}

impl fmt::Display for NativeAudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(index) => write!(
                f,
                "native backend produced non-finite audio at sample {index}; this backend may not support the model family. Retry with --native-backend cpu"
            ),
            Self::Silent => write!(
                f,
                "native backend produced silent audio; this backend may not support the model family. Retry with --native-backend cpu"
            ),
        }
    }
}

impl std::error::Error for NativeAudioError {}

fn should_retry_on_cpu(auto_backend: bool, backend: &str, err: &anyhow::Error) -> bool {
    auto_backend && backend != "cpu" && err.downcast_ref::<NativeAudioError>().is_some()
}

impl Drop for NativeSynth {
    fn drop(&mut self) {
        // SAFETY: each handle was produced by this library and is freed at most
        // once here. audiocpp treats a NULL handle free as a no-op.
        unsafe {
            if !self.session.is_null() {
                (self.session_free)(self.session);
            }
            if !self.options.is_null() {
                (self.options_free)(self.options);
            }
            if !self.model.is_null() {
                (self.model_free)(self.model);
            }
            if !self.registry.is_null() {
                (self.registry_free)(self.registry);
            }
        }
    }
}

struct NativeOpenConfig<'a> {
    library_path: &'a Path,
    family: NativeFamily,
    backend: NativeBackend,
    threads: i32,
    model_path: &'a Path,
    cache_root: &'a Path,
    espeak_lib: Option<&'a Path>,
    espeak_data: Option<&'a Path>,
}

impl NativeSynth {
    fn open(config: NativeOpenConfig<'_>) -> Result<Self> {
        let NativeOpenConfig {
            library_path,
            family,
            backend,
            threads,
            model_path,
            cache_root,
            espeak_lib,
            espeak_data,
        } = config;
        let library = open_native_library(library_path)?;
        let auto_backend = backend == NativeBackend::Auto;
        let mut synthesizer = Self {
            last_error: load_native_symbol(&library, "audiocpp_last_error")?,
            status_string: load_native_symbol(&library, "audiocpp_status_string")?,
            registry_create: load_native_symbol(&library, "audiocpp_registry_create")?,
            registry_free: load_native_symbol(&library, "audiocpp_registry_free")?,
            model_load: load_native_symbol(&library, "audiocpp_model_load")?,
            model_free: load_native_symbol(&library, "audiocpp_model_free")?,
            session_create: load_native_symbol(&library, "audiocpp_session_create")?,
            session_free: load_native_symbol(&library, "audiocpp_session_free")?,
            session_run: load_native_symbol(&library, "audiocpp_session_run")?,
            request_create: load_native_symbol(&library, "audiocpp_request_create")?,
            request_free: load_native_symbol(&library, "audiocpp_request_free")?,
            request_set_text: load_native_symbol(&library, "audiocpp_request_set_text")?,
            request_set_voice: load_native_symbol(&library, "audiocpp_request_set_voice_id")?,
            request_set_rate: load_native_symbol(&library, "audiocpp_request_set_speaking_rate")?,
            request_set_option: load_native_symbol(&library, "audiocpp_request_set_option")?,
            result_audio: load_native_symbol(&library, "audiocpp_result_audio")?,
            result_free: load_native_symbol(&library, "audiocpp_result_free")?,
            options_create: load_native_symbol(&library, "audiocpp_options_create")?,
            options_set: load_native_symbol(&library, "audiocpp_options_set")?,
            options_free: load_native_symbol(&library, "audiocpp_options_free")?,
            registry: std::ptr::null_mut(),
            options: std::ptr::null_mut(),
            model: std::ptr::null_mut(),
            session: std::ptr::null_mut(),
            family,
            random_voice_state_path: cache_root.join("last_random_voice.txt"),
            backend: "cpu",
            auto_backend,
            threads,
            _library: library,
        };

        let mut registry: *mut c_void = std::ptr::null_mut();
        synthesizer.check_status(
            unsafe { (synthesizer.registry_create)(std::ptr::null(), &mut registry) },
            "failed to create native registry",
        )?;
        synthesizer.registry = registry;

        let family_name = c_string(family.as_str())?;
        let model_path = model_path.to_str().ok_or_else(|| {
            anyhow!(
                "native model path is not valid UTF-8: {}",
                model_path.display()
            )
        })?;
        let model_path = c_string(model_path)?;
        let model_config = NativeModelConfig {
            family_hint: family_name.as_ptr(),
            config_id: std::ptr::null(),
            weight_id: std::ptr::null(),
            model_spec_override: std::ptr::null(),
        };
        let mut model: *mut c_void = std::ptr::null_mut();
        synthesizer.check_status(
            unsafe {
                (synthesizer.model_load)(
                    synthesizer.registry,
                    model_path.as_ptr(),
                    &model_config,
                    std::ptr::null(),
                    &mut model,
                )
            },
            "failed to load native model",
        )?;
        synthesizer.model = model;

        let espeak_lib = espeak_lib.map(path_to_c_string).transpose()?;
        let espeak_data = espeak_data.map(path_to_c_string).transpose()?;
        if espeak_lib.is_some() || espeak_data.is_some() {
            let options = unsafe { (synthesizer.options_create)() };
            if options.is_null() {
                bail!("native engine failed to create session options");
            }
            synthesizer.options = options;
            let set_option = |key: &str, value: &CString| -> Result<()> {
                let key = c_string(key)?;
                synthesizer.check_status(
                    unsafe { (synthesizer.options_set)(options, key.as_ptr(), value.as_ptr()) },
                    "failed to set native session option",
                )
            };
            if let Some(value) = &espeak_lib {
                set_option("kitten_tts.espeak_library_path", value)?;
            }
            if let Some(value) = &espeak_data {
                set_option("kitten_tts.espeak_data_path", value)?;
            }
        }

        let task = c_string("tts")?;
        let mode = c_string("offline")?;
        let candidates = backend.candidates();
        let attempt_count = candidates.len();
        let mut session_error = None;
        for backend_name in candidates {
            let backend_name_c = c_string(backend_name)?;
            let backend_config = NativeBackendConfig {
                backend: backend_name_c.as_ptr(),
                device: 0,
                threads,
            };
            let mut session: *mut c_void = std::ptr::null_mut();
            let status = unsafe {
                (synthesizer.session_create)(
                    synthesizer.model,
                    task.as_ptr(),
                    mode.as_ptr(),
                    &backend_config,
                    synthesizer.options,
                    &mut session,
                )
            };
            if status == AUDIOCPP_OK {
                eprintln!("[Native] backend={backend_name}");
                synthesizer.session = session;
                synthesizer.backend = backend_name;
                break;
            }
            let message = synthesizer.describe_status(status);
            if attempt_count > 1 {
                eprintln!("[Native] backend={backend_name} unavailable: {message}");
            }
            session_error = Some(message);
        }
        if synthesizer.session.is_null() {
            bail!(
                "failed to create native session: {}",
                session_error.unwrap_or_else(|| "no backend available".to_string())
            );
        }

        if let Some(path) = espeak_lib {
            eprintln!("[Native] espeak-ng library={}", path.to_string_lossy());
        }
        if let Some(path) = espeak_data {
            eprintln!("[Native] espeak-ng data={}", path.to_string_lossy());
        }

        Ok(synthesizer)
    }

    fn describe_status(&self, status: c_int) -> String {
        let message = borrowed_string(unsafe { (self.last_error)() });
        let label = borrowed_string(unsafe { (self.status_string)(status) });
        if message.is_empty() {
            format!("{label} ({status})")
        } else {
            format!("{message} ({label})")
        }
    }

    fn check_status(&self, status: c_int, action: &str) -> Result<()> {
        if status == AUDIOCPP_OK {
            return Ok(());
        }
        bail!("{action}: {}", self.describe_status(status))
    }

    fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        config: SynthesisConfig,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.synthesize_streaming(text, voice, config, |chunk| {
            output.extend_from_slice(chunk);
            Ok(())
        })?;
        Ok(output)
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        config: SynthesisConfig,
        mut on_chunk: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let chunks = prepare_native_chunks(text, config.speed, config.max_chars)?;
        let resolved_voice = self.resolve_voice(voice, config.seed)?;

        let started = Instant::now();
        let mut total_samples = 0usize;
        let mut first_audio = None;
        for chunk in chunks {
            let audio = match self.run_chunk(
                &chunk,
                resolved_voice.as_deref(),
                config.speed,
                config.seed,
            ) {
                Ok(audio) => audio,
                Err(err) if should_retry_on_cpu(self.auto_backend, self.backend, &err) => {
                    eprintln!(
                        "[Native] backend={} returned invalid audio; retrying with cpu",
                        self.backend
                    );
                    self.switch_backend("cpu")?;
                    self.run_chunk(&chunk, resolved_voice.as_deref(), config.speed, config.seed)?
                }
                Err(err) => return Err(err),
            };
            if first_audio.is_none() {
                first_audio = Some(started.elapsed());
            }
            total_samples += audio.len();
            on_chunk(&audio)?;
        }
        eprintln!(
            "[Native] synthesized {total_samples} samples in {:.2}s (first audio {} ms)",
            started.elapsed().as_secs_f64(),
            first_audio.map(|d| d.as_millis()).unwrap_or(0)
        );
        Ok(())
    }

    fn switch_backend(&mut self, backend: &'static str) -> Result<()> {
        if self.backend == backend {
            return Ok(());
        }

        let task = c_string("tts")?;
        let mode = c_string("offline")?;
        let backend_name = c_string(backend)?;
        let backend_config = NativeBackendConfig {
            backend: backend_name.as_ptr(),
            device: 0,
            threads: self.threads,
        };
        let mut session: *mut c_void = std::ptr::null_mut();
        self.check_status(
            unsafe {
                (self.session_create)(
                    self.model,
                    task.as_ptr(),
                    mode.as_ptr(),
                    &backend_config,
                    self.options,
                    &mut session,
                )
            },
            &format!("failed to switch native backend to {backend}"),
        )?;
        if session.is_null() {
            bail!("native engine returned no session for backend {backend}");
        }

        unsafe { (self.session_free)(self.session) };
        self.session = session;
        self.backend = backend;
        eprintln!("[Native] switched backend={backend}");
        Ok(())
    }

    fn resolve_voice(&self, voice: Option<&str>, seed: Option<u64>) -> Result<Option<String>> {
        if let Some(voice) = voice {
            let trimmed = voice.trim();
            if trimmed.is_empty() {
                bail!("voice name cannot be empty");
            }
            return Ok(Some(trimmed.to_string()));
        }

        let canonical = self.family.canonical_voices();
        if canonical.is_empty() {
            eprintln!("[Native] using the model's default voice");
            return Ok(None);
        }

        let mut rng = SimpleRng::from_seed(seed.unwrap_or_else(runtime_entropy_seed));
        let previous = load_last_random_voice(&self.random_voice_state_path);
        let mut candidates = canonical.to_vec();
        if let Some(previous) = previous
            && candidates.len() > 1
        {
            candidates.retain(|candidate| *candidate != previous.as_str());
        }
        let selected = candidates[rng.gen_index(candidates.len())].to_string();
        if let Err(err) = store_last_random_voice(&self.random_voice_state_path, &selected) {
            eprintln!(
                "warning: could not persist last random voice to {}: {err}",
                self.random_voice_state_path.display()
            );
        }
        eprintln!("using random voice: {selected}");
        Ok(Some(selected))
    }

    fn run_chunk(
        &mut self,
        text: &str,
        voice: Option<&str>,
        speed: f32,
        seed: Option<u64>,
    ) -> Result<Vec<f32>> {
        let request = unsafe { (self.request_create)() };
        if request.is_null() {
            bail!("native engine failed to create a request");
        }

        let mut result: *mut c_void = std::ptr::null_mut();
        let audio = (|| -> Result<Vec<f32>> {
            let text = c_string(text)?;
            self.check_status(
                unsafe { (self.request_set_text)(request, text.as_ptr(), std::ptr::null()) },
                "failed to set native request text",
            )?;

            if let Some(voice) = voice {
                let voice = c_string(voice)?;
                if let Err(err) = self.check_status(
                    unsafe { (self.request_set_voice)(request, voice.as_ptr()) },
                    "failed to set native request voice",
                ) {
                    if self.family == NativeFamily::KittenTts {
                        return Err(err.context(format!(
                            "available kitten_tts voices: {}",
                            NATIVE_KITTEN_TTS_VOICES.join(", ")
                        )));
                    }
                    return Err(err);
                }
            }

            if native_applies_speed(self.family, speed) {
                self.check_status(
                    unsafe { (self.request_set_rate)(request, speed) },
                    "failed to set native speaking rate",
                )?;
            }

            if let Some(seed) = seed {
                let key = c_string("seed")?;
                let value = c_string(&seed.to_string())?;
                self.check_status(
                    unsafe { (self.request_set_option)(request, key.as_ptr(), value.as_ptr()) },
                    "failed to set native request seed",
                )?;
            }

            if let Err(err) = self.check_status(
                unsafe { (self.session_run)(self.session, request, &mut result) },
                "native synthesis failed",
            ) {
                if self.family == NativeFamily::KittenTts {
                    return Err(err.context(format!(
                        "available kitten_tts voices: {}",
                        NATIVE_KITTEN_TTS_VOICES.join(", ")
                    )));
                }
                return Err(err);
            }
            if result.is_null() {
                bail!("native engine returned no result");
            }

            let mut samples: *const f32 = std::ptr::null();
            let mut frames = 0usize;
            let mut sample_rate: c_int = 0;
            let mut channels: c_int = 0;
            self.check_status(
                unsafe {
                    (self.result_audio)(
                        result,
                        &mut samples,
                        &mut frames,
                        &mut sample_rate,
                        &mut channels,
                    )
                },
                "failed to read native result audio",
            )?;
            if samples.is_null() || frames == 0 {
                bail!("native engine produced no audio");
            }
            if channels <= 0 {
                bail!("native engine reported an invalid channel count: {channels}");
            }
            if sample_rate <= 0 {
                bail!("native engine reported an invalid sample rate: {sample_rate}");
            }

            let channels = channels as usize;
            let total = frames
                .checked_mul(channels)
                .ok_or_else(|| anyhow!("native audio size overflow"))?;
            // SAFETY: the ABI returns `frames * channels` interleaved f32 samples that
            // stay valid until the result is freed, which happens after this copy.
            let raw = unsafe { std::slice::from_raw_parts(samples, total) };
            let audio: Vec<f32> = if channels == 1 {
                raw.to_vec()
            } else {
                raw.chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                    .collect()
            };

            if let Some(index) = audio.iter().position(|sample| !sample.is_finite()) {
                return Err(NativeAudioError::NonFinite(index).into());
            }
            if audio.iter().all(|sample| sample.abs() < 1e-6) {
                return Err(NativeAudioError::Silent.into());
            }

            Ok(audio)
        })();

        // SAFETY: both handles are either NULL or owned here, and freeing NULL is a no-op.
        unsafe {
            (self.result_free)(result);
            (self.request_free)(request);
        }
        audio
    }
}

fn write_stdout(text: &str) -> Result<()> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    match stdout
        .write_all(text.as_bytes())
        .and_then(|_| stdout.flush())
    {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(err) => Err(err).context("failed writing to stdout"),
    }
}

fn native_applies_speed(family: NativeFamily, speed: f32) -> bool {
    family == NativeFamily::KittenTts && (speed - 1.0).abs() > 1e-6
}

fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| anyhow!("value contains a NUL byte: {value}"))
}

fn path_to_c_string(path: &Path) -> Result<CString> {
    let path = path
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))?;
    c_string(path)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Models(models) => match models.command {
            ModelsCommands::List => {
                println!(
                    "Default preset: {} ({DEFAULT_REPO_ID})",
                    default_preset_label()
                );
                for preset in ModelPreset::all() {
                    let marker = if preset.repo_id() == DEFAULT_REPO_ID {
                        " (default)"
                    } else {
                        ""
                    };
                    println!(
                        "- {} -> {}{}",
                        format_model_preset(preset),
                        preset.repo_id(),
                        marker
                    );
                }
                println!();
                println!("Native engine (optional, no ONNX Runtime):");
                println!("- {NATIVE_KITTEN_TTS_DIR}/{NATIVE_KITTEN_TTS_FILE} (about 288 MiB)");
                println!("  download with: kitten-tts native fetch [--cache-dir DIR] [--force]");
                println!(
                    "  used by: kitten-tts synthesize --engine native --model mini-0.8 --native-lib PATH"
                );
                println!("  check the library first: kitten-tts native probe --native-lib PATH");
                println!("  check library + eSpeak + model at once: kitten-tts native doctor");
            }
        },
        Commands::Native(native_cmd) => match native_cmd.command {
            NativeCommands::Probe(args) => probe_native_runtime(&args)?,
            NativeCommands::Fetch(args) => fetch_native_model(&args)?,
            NativeCommands::Doctor(args) => doctor_native_runtime(&args)?,
        },
        Commands::Model(model_cmd) => {
            let client = build_http_client()?;
            match model_cmd.command {
                ModelCommands::Fetch(args) => {
                    let cache_root = args.model.resolve_cache_dir()?;
                    let repo_id = args.model.resolve_repo_id();
                    let artifacts =
                        ensure_model_cached(&client, &cache_root, &repo_id, args.force)?;
                    println!("Fetched model into {}", artifacts.cache_dir.display());
                    println!("Repo: {}", artifacts.repo_id);
                    println!("Model: {}", artifacts.model_path.display());
                    println!("Voices: {}", artifacts.voices_path.display());
                }
                ModelCommands::Info(args) => {
                    let cache_root = args.model.resolve_cache_dir()?;
                    let repo_id = args.model.resolve_repo_id();
                    let artifacts = ensure_model_cached(&client, &cache_root, &repo_id, false)?;
                    println!("Repo: {}", artifacts.repo_id);
                    println!("Cache dir: {}", artifacts.cache_dir.display());
                    println!("Config: {}", artifacts.config_path.display());
                    println!("Model: {}", artifacts.model_path.display());
                    println!("Voices: {}", artifacts.voices_path.display());
                    println!("Aliases: {}", artifacts.config.voice_aliases.len());
                }
            }
        }
        Commands::Voices(args) => {
            let client = build_http_client()?;
            let cache_root = args.model.resolve_cache_dir()?;
            let repo_id = args.model.resolve_repo_id();
            let artifacts = ensure_model_cached(&client, &cache_root, &repo_id, false)?;
            let voice_table = VoiceTable::load(&artifacts.voices_path)?;

            println!("Repo: {}", artifacts.repo_id);
            println!("Canonical voices:");
            for voice in voice_table.canonical_voices() {
                println!("- {voice}");
            }
            println!("Aliases:");
            let mut aliases: Vec<_> = artifacts.config.voice_aliases.iter().collect();
            aliases.sort_by(|a, b| a.0.cmp(b.0));
            for (alias, canonical) in aliases {
                println!("- {alias} -> {canonical}");
            }
        }
        Commands::Play(args) => {
            let (text, mut backend) = prepare_synthesis_runtime(&args.synthesis)?;
            let config = SynthesisConfig::from(&args.synthesis);

            let audio = backend.synthesize(&text, args.synthesis.voice.as_deref(), config)?;
            play_audio(
                &audio,
                args.synthesis.sample_rate,
                args.player,
                args.gain,
                args.allow_clipping,
            )?;
        }
        Commands::Synthesize(args) => {
            let (text, mut backend) = prepare_synthesis_runtime(&args.synthesis)?;
            let config = SynthesisConfig::from(&args.synthesis);

            let audio = backend.synthesize(&text, args.synthesis.voice.as_deref(), config)?;

            write_wav(
                &args.output,
                args.synthesis.sample_rate,
                args.wav_encoding,
                &audio,
            )?;
            eprintln!("wrote {} samples to {}", audio.len(), args.output.display());
        }
        Commands::Stream(args) => {
            let (text, mut backend) = prepare_synthesis_runtime(&args.synthesis)?;
            let config = SynthesisConfig::from(&args.synthesis);

            let stdout = io::stdout();
            let mut stdout = stdout.lock();
            backend.synthesize_streaming(
                &text,
                args.synthesis.voice.as_deref(),
                config,
                |chunk| {
                    for sample in chunk {
                        stdout.write_all(&sample.to_le_bytes())?;
                    }
                    stdout.flush()?;
                    Ok(())
                },
            )?;
        }
    }

    Ok(())
}

fn format_model_preset(preset: ModelPreset) -> &'static str {
    match preset {
        ModelPreset::Nano08Int8 => "nano-0.8-int8",
        ModelPreset::Nano08Fp32 => "nano-0.8-fp32",
        ModelPreset::Micro08 => "micro-0.8",
        ModelPreset::Mini08 => "mini-0.8",
    }
}

fn default_preset_label() -> &'static str {
    for preset in ModelPreset::all() {
        if preset.repo_id() == DEFAULT_REPO_ID {
            return format_model_preset(preset);
        }
    }
    "custom"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_selection_defaults_to_nano_fp32() {
        let selection = ModelSelection {
            model: None,
            repo_id: None,
            cache_dir: None,
            ort_lib: None,
            cuda_lib_dir: None,
            cudnn_lib_dir: None,
            cuda_device_id: None,
            cuda_memory_limit: None,
            arena_strategy: ArenaStrategy::SameAsRequested,
            conv_algo: ConvAlgo::Heuristic,
            graph_opt: GraphOpt::Level1,
            memory_pattern: false,
            parallel_execution: false,
            vram_profile: None,
        };
        assert_eq!(selection.resolve_repo_id(), DEFAULT_REPO_ID);
    }

    #[test]
    fn repo_id_overrides_model_preset() {
        let selection = ModelSelection {
            model: Some(ModelPreset::Micro08),
            repo_id: Some("KittenML/custom".to_string()),
            cache_dir: None,
            ort_lib: None,
            cuda_lib_dir: None,
            cudnn_lib_dir: None,
            cuda_device_id: None,
            cuda_memory_limit: None,
            arena_strategy: ArenaStrategy::SameAsRequested,
            conv_algo: ConvAlgo::Heuristic,
            graph_opt: GraphOpt::Level1,
            memory_pattern: false,
            parallel_execution: false,
            vram_profile: None,
        };
        assert_eq!(selection.resolve_repo_id(), "KittenML/custom");
    }

    #[test]
    fn test_tokenize_phonemes_spacing() {
        let cleaner = TextCleaner::new();
        let phonemes = "hɛloʊ, wɜːld.";
        let tokens = tokenize_phonemes(&cleaner, phonemes);

        let mut decoded = String::new();
        let reverse_map: std::collections::HashMap<i64, char> = cleaner
            .word_index_dictionary
            .iter()
            .map(|(&k, &v)| (v, k))
            .collect();
        for t in tokens {
            if let Some(c) = reverse_map.get(&t) {
                decoded.push(*c);
            }
        }
        assert_eq!(decoded, "hɛloʊ, wɜːld."); // This will fail if it injects spaces
    }

    #[test]
    fn chunking_retains_punctuation() {
        let text = "Hello world! How are you?";
        let chunks = chunk_text(text, 50);
        // This test will fail if punctuation is lost and replaced with commas.
        assert_eq!(chunks, vec!["Hello world!", "How are you?"]);
    }

    #[test]
    fn chunking_splits_long_sentence_and_adds_punctuation() {
        let text = "hello world this sentence is long enough to split";
        let chunks = chunk_text(text, 10);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| {
            matches!(
                c.chars().last(),
                Some('.') | Some('!') | Some('?') | Some(',') | Some(';') | Some(':')
            )
        }));
    }

    #[test]
    fn cache_dir_is_stable_for_same_repo() {
        let root = Path::new("/tmp/kitten");
        let a = cache_model_dir(root, "KittenML/kitten-tts-nano-0.8-int8");
        let b = cache_model_dir(root, "KittenML/kitten-tts-nano-0.8-int8");
        assert_eq!(a, b);
    }

    #[test]
    fn parse_shape_extracts_expected_values() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (400, 256), }";
        assert_eq!(parse_shape(header), Some(vec![400, 256]));
    }

    #[test]
    fn parse_npy_rejects_shape_that_overflows_byte_count() {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 1), }}",
            usize::MAX
        );
        let mut payload = Vec::new();
        payload.extend_from_slice(b"\x93NUMPY");
        payload.push(2);
        payload.push(0);
        payload.extend_from_slice(&(header.len() as u32).to_le_bytes());
        payload.extend_from_slice(header.as_bytes());

        let err = parse_npy_f32(&payload).expect_err("overflow shape should be rejected");
        assert!(format!("{err:#}").contains("npy byte size overflow"));
    }

    #[test]
    fn clean_text_collapses_whitespace() {
        let text = "  hello   world\n\tfoo  ";
        assert_eq!(clean_text_basic(text), "hello world foo");
    }

    #[test]
    fn resolve_style_index_defaults_to_seeded_random_in_range() {
        let mut a = SimpleRng::from_seed(1234);
        let mut b = SimpleRng::from_seed(1234);
        let idx_a = resolve_style_index(400, None, &mut a);
        let idx_b = resolve_style_index(400, None, &mut b);
        assert_eq!(idx_a, idx_b);
        assert!(idx_a < 400);
    }

    #[test]
    fn resolve_style_index_honors_override_and_clamps() {
        let mut rng = SimpleRng::from_seed(42);
        assert_eq!(resolve_style_index(400, Some(4), &mut rng), 4);
        assert_eq!(resolve_style_index(400, Some(9999), &mut rng), 399);
    }

    #[test]
    fn resolve_style_index_handles_empty_rows() {
        let mut rng = SimpleRng::from_seed(1);
        assert_eq!(resolve_style_index(0, None, &mut rng), 0);
    }

    #[test]
    fn apply_playback_gain_limits_when_clipping_not_allowed() {
        let audio = vec![0.8_f32, -0.8];
        let (scaled, applied) = apply_playback_gain(&audio, 2.5, false).expect("gain should work");
        assert!(applied < 2.5);
        let peak = scaled
            .iter()
            .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
        assert!(peak <= SAFE_PLAYBACK_PEAK + 1e-6);
    }

    #[test]
    fn apply_playback_gain_honors_gain_when_clipping_allowed() {
        let audio = vec![0.5_f32, -0.25];
        let (scaled, applied) = apply_playback_gain(&audio, 2.0, true).expect("gain should work");
        assert_eq!(applied, 2.0);
        assert!((scaled[0] - 1.0).abs() < 1e-6);
        assert!((scaled[1] + 0.5).abs() < 1e-6);
    }

    fn default_ort_config() -> OrtRuntimeConfig {
        OrtRuntimeConfig {
            ort_lib: None,
            cuda_lib_dir: None,
            cudnn_lib_dir: None,
            cuda_device_id: DEFAULT_CUDA_DEVICE_ID,
            cuda_memory_limit_bytes: DEFAULT_CUDA_MEMORY_LIMIT_BYTES,
            arena_strategy: ArenaStrategy::SameAsRequested,
            conv_algo: ConvAlgo::Heuristic,
            graph_opt: GraphOpt::Level1,
            memory_pattern: false,
            parallel_execution: false,
            vram_profile: None,
        }
    }

    #[test]
    fn resolve_session_config_performance_profile_overrides_individual_flags() {
        let mut cfg = default_ort_config();
        cfg.vram_profile = Some(VramProfile::Performance);
        // Individual flags are still at conservative defaults, but profile wins
        let resolved = resolve_session_config(&cfg);
        assert_eq!(resolved.arena_strategy, ArenaStrategy::NextPowerOfTwo);
        assert_eq!(resolved.conv_algo, ConvAlgo::Exhaustive);
        assert_eq!(resolved.graph_opt, GraphOpt::Level3);
        assert!(resolved.memory_pattern);
        assert!(resolved.parallel_execution);
    }

    #[test]
    fn resolve_session_config_minimal_profile_forces_conservative() {
        let mut cfg = default_ort_config();
        cfg.vram_profile = Some(VramProfile::Minimal);
        // Even if individual flags are aggressive, profile overrides
        cfg.arena_strategy = ArenaStrategy::NextPowerOfTwo;
        cfg.conv_algo = ConvAlgo::Exhaustive;
        cfg.graph_opt = GraphOpt::Level3;
        cfg.memory_pattern = true;
        cfg.parallel_execution = true;
        let resolved = resolve_session_config(&cfg);
        assert_eq!(resolved.arena_strategy, ArenaStrategy::SameAsRequested);
        assert_eq!(resolved.conv_algo, ConvAlgo::Heuristic);
        assert_eq!(resolved.graph_opt, GraphOpt::Level1);
        assert!(!resolved.memory_pattern);
        assert!(!resolved.parallel_execution);
        assert_eq!(resolved.inter_threads, SAFE_INTER_THREADS);
    }

    #[test]
    fn resolve_session_config_no_profile_passes_individual_flags() {
        let mut cfg = default_ort_config();
        cfg.arena_strategy = ArenaStrategy::NextPowerOfTwo;
        cfg.conv_algo = ConvAlgo::Exhaustive;
        cfg.graph_opt = GraphOpt::Level3;
        cfg.memory_pattern = true;
        let resolved = resolve_session_config(&cfg);
        assert_eq!(resolved.arena_strategy, ArenaStrategy::NextPowerOfTwo);
        assert_eq!(resolved.conv_algo, ConvAlgo::Exhaustive);
        assert_eq!(resolved.graph_opt, GraphOpt::Level3);
        assert!(resolved.memory_pattern);
    }

    #[test]
    fn resolve_session_config_parallel_enables_inter_threads() {
        let mut cfg = default_ort_config();
        cfg.parallel_execution = true;
        let resolved = resolve_session_config(&cfg);
        assert!(resolved.parallel_execution);
        // inter_threads must be > 1 (or 0 = auto) for parallel to work
        assert_ne!(resolved.inter_threads, SAFE_INTER_THREADS);
    }

    #[test]
    fn resolve_session_config_no_parallel_keeps_safe_inter_threads() {
        let cfg = default_ort_config();
        let resolved = resolve_session_config(&cfg);
        assert!(!resolved.parallel_execution);
        assert_eq!(resolved.inter_threads, SAFE_INTER_THREADS);
    }

    #[test]
    fn native_abi_unpacks_major_minor_patch() {
        assert_eq!(abi_parts((1 << 16) | (2 << 8) | 3), (1, 2, 3));
        assert_eq!(abi_parts(0), (0, 0, 0));
    }

    #[test]
    fn native_abi_major_mismatch_is_rejected() {
        let err = ensure_compatible_abi(AUDIOCPP_ABI_MAJOR + 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("incompatible"), "{err}");
    }

    #[test]
    fn native_abi_current_major_is_accepted() {
        assert!(ensure_compatible_abi(AUDIOCPP_ABI_MAJOR).is_ok());
    }

    #[test]
    fn native_lib_path_prefers_explicit_argument() {
        let path = resolve_native_lib_path(Some(Path::new("/custom/libaudiocpp.so")));
        assert_eq!(path, PathBuf::from("/custom/libaudiocpp.so"));
    }

    #[test]
    fn native_loader_reports_missing_library() {
        let err = NativeRuntime::load(Path::new("/nonexistent/libaudiocpp-missing.so"))
            .err()
            .expect("missing library should fail")
            .to_string();
        assert!(
            err.contains("failed to load native audio.cpp library"),
            "{err}"
        );
    }

    fn parse_synthesis_args(extra: &[&str]) -> SynthesisArgs {
        let mut argv = vec!["kitten-tts", "synthesize", "--text", "hello"];
        argv.extend_from_slice(extra);
        let cli = Cli::try_parse_from(argv).expect("valid cli");
        match cli.command {
            Commands::Synthesize(args) => args.synthesis,
            _ => panic!("expected synthesize command"),
        }
    }

    #[test]
    fn engine_defaults_to_auto() {
        assert_eq!(parse_synthesis_args(&[]).engine, EngineChoice::Auto);
    }

    #[test]
    fn auto_engine_is_opportunistic_only_for_mini() {
        assert_eq!(
            auto_native_intent(&parse_synthesis_args(&[])),
            AutoNativeIntent::None
        );
        assert_eq!(
            auto_native_intent(&parse_synthesis_args(&["--model", "mini-0.8"])),
            AutoNativeIntent::Opportunistic
        );
    }

    #[test]
    fn auto_engine_honours_explicit_native_intent() {
        assert_eq!(
            auto_native_intent(&parse_synthesis_args(&[
                "--native-model",
                "/tmp/model.gguf"
            ])),
            AutoNativeIntent::Required
        );
        assert_eq!(
            auto_native_intent(&parse_synthesis_args(&["--native-family", "kitten_tts2"])),
            AutoNativeIntent::Required
        );
    }

    #[test]
    fn native_threads_must_be_positive() {
        let args = parse_synthesis_args(&["--engine", "native", "--native-threads", "0"]);
        let err = validate_native_args(&args).unwrap_err().to_string();
        assert!(err.contains("between 1 and 256"), "{err}");
    }

    #[test]
    fn native_rejects_style_index() {
        let args = parse_synthesis_args(&["--engine", "native", "--style-index", "3"]);
        let err = validate_native_args(&args).unwrap_err().to_string();
        assert!(err.contains("--style-index"), "{err}");
    }

    #[test]
    fn native_rejects_non_mini_preset_without_override() {
        let args = parse_synthesis_args(&["--engine", "native", "--model", "nano-0.8-fp32"]);
        let err = validate_native_args(&args).unwrap_err().to_string();
        assert!(err.contains("Mini 0.8"), "{err}");
    }

    #[test]
    fn native_kitten_tts2_requires_license_acknowledgement() {
        let args = parse_synthesis_args(&[
            "--engine",
            "native",
            "--native-family",
            "kitten_tts2",
            "--native-model",
            "/tmp/kitten-tts2.gguf",
        ]);
        let err = validate_native_args(&args).unwrap_err().to_string();
        assert!(err.contains("Stellon Labs"), "{err}");
    }

    #[test]
    fn native_kitten_tts2_accepts_license_and_local_model() {
        let args = parse_synthesis_args(&[
            "--engine",
            "native",
            "--native-family",
            "kitten_tts2",
            "--native-model",
            "/tmp/kitten-tts2.gguf",
            "--accept-kitten2-license",
        ]);
        assert!(validate_native_args(&args).is_ok());
    }

    #[test]
    fn synthesis_guardrails_apply_to_shared_chunking() {
        let err = prepare_synthesis_chunks("hello", 9.0, 400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("speed must be between"), "{err}");
    }

    #[test]
    fn native_auto_backend_prefers_hardware_then_cpu() {
        let candidates = NativeBackend::Auto.candidates();
        assert_eq!(candidates.last().copied(), Some("cpu"));
        if cfg!(target_os = "macos") {
            assert_eq!(candidates.first().copied(), Some("metal"));
        }
        if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
            assert_eq!(candidates.first().copied(), Some("cuda"));
        }
        assert_eq!(NativeBackend::Cpu.candidates(), vec!["cpu"]);
    }

    #[test]
    fn native_invalid_audio_retries_on_cpu_only_in_auto_mode() {
        let audio_error: anyhow::Error = NativeAudioError::Silent.into();
        assert!(should_retry_on_cpu(true, "metal", &audio_error));
        assert!(!should_retry_on_cpu(false, "metal", &audio_error));
        assert!(!should_retry_on_cpu(true, "cpu", &audio_error));

        let other_error = anyhow!("session failed");
        assert!(!should_retry_on_cpu(true, "metal", &other_error));
    }

    #[test]
    fn native_fetch_refuses_v2_without_network() {
        let args = NativeFetchArgs {
            family: NativeFamily::KittenTts2,
            cache_dir: None,
            force: false,
        };
        let err = fetch_native_model(&args).unwrap_err().to_string();
        assert!(err.contains("not downloaded automatically"), "{err}");
        assert!(err.contains("--accept-kitten2-license"), "{err}");
    }

    #[test]
    fn native_fetch_uses_cached_model_without_force() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let existing = dir.path().join("native").join(NATIVE_KITTEN_TTS_FILE);
        fs::create_dir_all(existing.parent().expect("parent"))?;
        fs::write(&existing, b"cached")?;

        let client = build_http_client()?;
        let path = ensure_native_kitten_tts_model(&client, dir.path(), false)?;
        assert_eq!(path, existing);
        assert_eq!(fs::read(&path)?, b"cached");
        Ok(())
    }

    #[test]
    fn native_chunking_keeps_short_text_in_one_request() {
        let chunks = prepare_native_chunks("First sentence. Second sentence.", 1.0, 400)
            .expect("valid chunks");
        assert_eq!(chunks, vec!["First sentence. Second sentence."]);
    }

    #[test]
    fn native_chunking_splits_text_above_max_chars() {
        let text = "This is a long first sentence for the test. This is a long second sentence for the test.";
        let chunks = prepare_native_chunks(text, 1.0, 30).expect("valid chunks");
        assert!(chunks.len() > 1, "{chunks:?}");
    }

    #[test]
    fn native_speed_only_applies_to_kitten_tts() {
        assert!(native_applies_speed(NativeFamily::KittenTts, 1.5));
        assert!(!native_applies_speed(NativeFamily::KittenTts, 1.0));
        assert!(!native_applies_speed(NativeFamily::KittenTts2, 1.5));
    }

    #[test]
    fn explicit_native_espeak_path_must_exist() {
        let err = resolve_espeak_path(
            Some(Path::new("/nonexistent/libespeak-ng.dylib")),
            detect_espeak_library,
            "libespeak-ng",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("does not exist"), "{err}");
    }

    #[test]
    fn doctor_reports_missing_cached_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let err = check_native_doctor_model(NativeFamily::KittenTts, None, dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("cached model is missing"), "{err}");
        Ok(())
    }

    #[test]
    fn doctor_accepts_cached_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let existing = dir.path().join("native").join(NATIVE_KITTEN_TTS_FILE);
        fs::create_dir_all(existing.parent().expect("parent"))?;
        fs::write(&existing, b"cached")?;

        let path = check_native_doctor_model(NativeFamily::KittenTts, None, dir.path())?;
        assert_eq!(path, existing);
        Ok(())
    }

    #[test]
    fn doctor_v2_requires_explicit_model() {
        let err = check_native_doctor_model(
            NativeFamily::KittenTts2,
            None,
            Path::new("/nonexistent-cache"),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("not downloaded automatically"), "{err}");
        assert!(err.contains("--native-model"), "{err}");
    }

    #[test]
    fn doctor_rejects_missing_explicit_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let missing = dir.path().join("weights.gguf");
        let err = check_native_doctor_model(NativeFamily::KittenTts, Some(&missing), dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not exist"), "{err}");
        Ok(())
    }
}
