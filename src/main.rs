use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use directories::ProjectDirs;
use libloading::Library;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::blocking::Client;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, IsTerminal, Read, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_MAX_CHARS: usize = 400;
const DEFAULT_PLAY_GAIN: f32 = 2.5;
const MAX_TOTAL_INPUT_CHARS: usize = 20_000;
const MAX_SYNTHESIS_CHUNKS: usize = 256;
const MAX_CHARS_HARD_LIMIT: usize = 2_000;
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 2.0;
const MIN_OUTPUT_GAIN: f32 = 0.1;
const MAX_OUTPUT_GAIN: f32 = 8.0;
const SAFE_PLAYBACK_PEAK: f32 = 0.98;
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

#[derive(Debug, Parser)]
#[command(name = "kitten-tts")]
#[command(about = "Native KittenTTS text-to-speech with audio.cpp")]
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
    #[arg(
        long,
        help = "Random seed for reproducible voice selection and decoder noise"
    )]
    seed: Option<u64>,
    #[arg(
        long,
        value_enum,
        default_value_t = EngineChoice::Native,
        help = "Native engine (auto is a compatibility alias for native)"
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
    #[arg(long, value_enum, default_value_t = NativeFamily::KittenTts)]
    native_family: NativeFamily,
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
    long_about = "Native audio.cpp engine control. `doctor` checks everything `--engine native` needs in one pass: the library and its ABI/families, the eSpeak runtime, and the model (cached or explicit). `probe` reports library details only; `fetch` downloads the Mini 0.8 GGUF so `--engine native` can run offline. All synthesis uses this engine."
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
        long_about = "Download the KittenTTS Mini 0.8 GGUF used by `--model mini-0.8` into the cache directory. Re-running is cheap: an existing file is reused unless --force is passed. KittenTTS 2 weights are licensed differently and are never downloaded by this command."
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
    #[arg(long, value_enum, default_value_t = ModelPreset::Mini08)]
    model: ModelPreset,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PlayerMode {
    Auto,
    Ffplay,
    PwPlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EngineChoice {
    Auto,
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
    #[value(name = "mini-0.8")]
    Mini08,
}

impl ModelSelection {
    fn resolve_cache_dir(&self) -> Result<PathBuf> {
        resolve_cache_root(self.cache_dir.as_deref())
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

#[derive(Debug, Clone, Copy)]
struct SynthesisConfig {
    speed: f32,
    max_chars: usize,
    seed: Option<u64>,
}

impl From<&SynthesisArgs> for SynthesisConfig {
    fn from(args: &SynthesisArgs) -> Self {
        Self {
            speed: args.speed,
            max_chars: args.max_chars,
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

        let expected_len = response.content_length();
        let mut file = File::create(&temp_path)
            .with_context(|| format!("failed creating temp file {}", temp_path.display()))?;
        let written = io::copy(&mut response, &mut file)
            .with_context(|| format!("failed writing to temp file {}", temp_path.display()))?;
        file.flush()?;
        verify_download_size(expected_len, written, &url)?;
        validate_native_gguf(&temp_path)?;

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

fn verify_download_size(expected: Option<u64>, written: u64, url: &str) -> Result<()> {
    if written == 0 {
        bail!("download from {url} returned no data");
    }
    if let Some(expected) = expected
        && expected != written
    {
        bail!("download from {url} was truncated: expected {expected} bytes, received {written}");
    }
    Ok(())
}

fn validate_native_gguf(path: &Path) -> Result<()> {
    let mut file = File::open(path)
        .with_context(|| format!("failed opening native model {}", path.display()))?;
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)
        .with_context(|| format!("failed reading native model header from {}", path.display()))?;
    if &magic != b"GGUF" {
        bail!(
            "native model '{}' is not a GGUF file (unexpected header)",
            path.display()
        );
    }
    Ok(())
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

fn prepare_synthesis_runtime(args: &SynthesisArgs) -> Result<(String, NativeSynth)> {
    let text = read_text_input(&args.text)?;
    prepare_native_chunks(&text, args.speed, args.max_chars)?;
    Ok((text, prepare_native_backend(args)?))
}

fn prepare_native_backend(args: &SynthesisArgs) -> Result<NativeSynth> {
    validate_native_args(args)?;

    if args.native_family == NativeFamily::KittenTts2 && (args.speed - 1.0).abs() > 1e-6 {
        eprintln!("note: the kitten_tts2 port does not implement --speed; using its default rate");
    }

    let library_path = resolve_native_lib_path(args.native_lib.as_deref());
    // Check local prerequisites before downloading weights.
    drop(open_native_library(&library_path)?);
    let (espeak_lib, espeak_data) = if args.native_family == NativeFamily::KittenTts {
        (
            resolve_espeak_path(
                args.native_espeak_lib.as_deref(),
                detect_espeak_library,
                "libespeak-ng",
                false,
            )?,
            resolve_espeak_path(
                args.native_espeak_data.as_deref(),
                detect_espeak_data,
                "espeak-ng-data",
                true,
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
    let cache_root = args.model.resolve_cache_dir()?;
    let client = build_http_client()?;
    let model_path = ensure_native_model_cached(&client, &cache_root, args)?;
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

    Ok(synthesizer)
}

fn validate_native_args(args: &SynthesisArgs) -> Result<()> {
    if args.sample_rate != DEFAULT_SAMPLE_RATE {
        bail!("native output is fixed at {DEFAULT_SAMPLE_RATE} Hz; omit --sample-rate");
    }
    if !(1..=256).contains(&args.native_threads) {
        bail!("--native-threads must be between 1 and 256");
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
        if let Some(seed) = args.seed
            && seed > i64::MAX as u64
        {
            bail!(
                "the kitten_tts2 port reads --seed as a signed 64-bit integer; use a value up to {}",
                i64::MAX
            );
        }
    }
    Ok(())
}

fn resolve_espeak_path(
    explicit: Option<&Path>,
    detect: fn() -> Option<PathBuf>,
    label: &str,
    directory: bool,
) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        if !path.exists() {
            bail!("{label} path '{}' does not exist", path.display());
        }
        if directory && !path.is_dir() {
            bail!("{label} path '{}' must be a directory", path.display());
        }
        if !directory && !path.is_file() {
            bail!("{label} path '{}' must be a file", path.display());
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
        validate_native_gguf(path)?;
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
    if let Err(err) = validate_native_gguf(&destination) {
        bail!(
            "cached native model at {} is unusable: {err:#}; run `kitten-tts native fetch --force` to download it again",
            destination.display()
        );
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
        validate_native_gguf(path)?;
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
            if let Err(err) = validate_native_gguf(&path) {
                bail!(
                    "cached model at {} is unusable: {err:#}; run `kitten-tts native fetch --force`",
                    path.display()
                );
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
            false,
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
            true,
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

fn validate_native_audio(audio: &[f32]) -> Result<()> {
    if let Some(index) = audio.iter().position(|sample| !sample.is_finite()) {
        return Err(NativeAudioError::NonFinite(index).into());
    }
    if audio.iter().all(|sample| sample.abs() < 1e-6) {
        return Err(NativeAudioError::Silent.into());
    }
    Ok(())
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
        let previous = if seed.is_none() {
            load_last_random_voice(&self.random_voice_state_path)
        } else {
            None
        };
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
            if sample_rate != DEFAULT_SAMPLE_RATE as c_int {
                bail!("native engine reported {sample_rate} Hz; expected {DEFAULT_SAMPLE_RATE} Hz");
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

            validate_native_audio(&audio)?;

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
                println!("Default: mini-0.8 (native audio.cpp, about 288 MiB)");
                println!("- {NATIVE_MODEL_REPO}/{NATIVE_KITTEN_TTS_DIR}/{NATIVE_KITTEN_TTS_FILE}");
                println!("  download: kitten-tts native fetch");
                println!("- kitten_tts2: local GGUF + --accept-kitten2-license required");
                println!("  check: kitten-tts native doctor");
            }
        },
        Commands::Native(native_cmd) => match native_cmd.command {
            NativeCommands::Probe(args) => probe_native_runtime(&args)?,
            NativeCommands::Fetch(args) => fetch_native_model(&args)?,
            NativeCommands::Doctor(args) => doctor_native_runtime(&args)?,
        },
        Commands::Voices(args) => {
            for voice in args.native_family.canonical_voices() {
                println!("{voice}");
            }
            if args.native_family == NativeFamily::KittenTts2 {
                println!("KittenTTS 2 uses the model default voice when --voice is omitted.");
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn clean_text_collapses_whitespace() {
        let text = "  hello   world\n\tfoo  ";
        assert_eq!(clean_text_basic(text), "hello world foo");
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
    fn native_mini_is_the_default_on_every_platform() {
        let args = parse_synthesis_args(&[]);
        assert_eq!(args.engine, EngineChoice::Native);
        assert_eq!(args.model.model, ModelPreset::Mini08);
        assert_eq!(args.native_family, NativeFamily::KittenTts);
        assert!(validate_native_args(&args).is_ok());
        assert_eq!(
            parse_synthesis_args(&["--engine", "auto"]).engine,
            EngineChoice::Auto
        );
    }

    #[test]
    fn removed_onnx_options_are_cli_errors() {
        for extra in [
            vec!["--engine", "onnx"],
            vec!["--model", "nano-0.8-fp32"],
            vec!["--model", "nano-0.8-int8"],
            vec!["--model", "micro-0.8"],
            vec!["--repo-id", "someone/model"],
            vec!["--ort-lib", "runtime"],
            vec!["--cuda-lib-dir", "cuda"],
            vec!["--cudnn-lib-dir", "cudnn"],
            vec!["--cuda-device-id", "0"],
            vec!["--cuda-memory-limit", "2048"],
            vec!["--arena-strategy", "same-as-requested"],
            vec!["--conv-algo", "heuristic"],
            vec!["--graph-opt", "1"],
            vec!["--vram-profile", "minimal"],
            vec!["--memory-pattern"],
            vec!["--parallel-execution"],
            vec!["--style-index", "3"],
            vec!["--trim-tail", "5000"],
            vec!["--phonemizer", "espeak-ng"],
        ] {
            for command in ["synthesize", "stream", "play"] {
                let mut argv = vec!["kitten-tts", command, "--text", "hello"];
                argv.extend_from_slice(&extra);
                let err = Cli::try_parse_from(argv).unwrap_err();
                assert!(
                    matches!(
                        err.kind(),
                        clap::error::ErrorKind::UnknownArgument
                            | clap::error::ErrorKind::InvalidValue
                    ),
                    "{command} {extra:?}: {err}"
                );
            }
        }
        assert!(Cli::try_parse_from(["kitten-tts", "model", "fetch"]).is_err());
    }

    #[test]
    fn missing_native_library_never_falls_back() {
        for engine in ["native", "auto"] {
            let args =
                parse_synthesis_args(&["--engine", engine, "--native-lib", "/missing/libaudiocpp"]);
            let err = match prepare_synthesis_runtime(&args) {
                Err(err) => err,
                Ok(_) => panic!("missing library should fail"),
            };
            assert!(
                format!("{err:#}").contains("failed to load native audio.cpp"),
                "{err:#}"
            );
        }
    }

    #[test]
    fn native_kitten_tts2_requires_local_model_even_with_license() {
        let args =
            parse_synthesis_args(&["--native-family", "kitten_tts2", "--accept-kitten2-license"]);
        assert!(
            validate_native_args(&args)
                .unwrap_err()
                .to_string()
                .contains("--native-model")
        );
    }

    #[test]
    fn native_rejects_mislabelled_sample_rate() {
        let args = parse_synthesis_args(&["--sample-rate", "48000"]);
        assert!(
            validate_native_args(&args)
                .unwrap_err()
                .to_string()
                .contains("24000")
        );
    }

    #[test]
    fn native_audio_must_be_finite_and_non_silent() {
        for audio in [
            vec![],
            vec![0.0; 10],
            vec![1e-8; 10],
            vec![0.1, f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            let err = validate_native_audio(&audio).unwrap_err();
            assert!(err.downcast_ref::<NativeAudioError>().is_some());
        }
        assert!(validate_native_audio(&[0.0, 0.1, -0.2]).is_ok());
    }

    #[test]
    fn native_guardrails_reject_unsafe_input_before_library_load() {
        for extra in [
            vec!["--speed", "NaN"],
            vec!["--speed", "0.4"],
            vec!["--max-chars", "0"],
            vec!["--max-chars", "2001"],
        ] {
            let err = match prepare_synthesis_runtime(&parse_synthesis_args(&extra)) {
                Err(err) => err,
                Ok(_) => panic!("unsafe input should fail"),
            };
            let expected = if extra[0] == "--speed" {
                "speed must"
            } else {
                "max_chars must"
            };
            assert!(err.to_string().contains(expected), "{err:#}");
        }
        assert!(prepare_native_chunks(&"x".repeat(MAX_TOTAL_INPUT_CHARS + 1), 1.0, 400).is_err());
        assert!(prepare_native_chunks(&"x. ".repeat(MAX_SYNTHESIS_CHUNKS + 1), 1.0, 1).is_err());
    }

    #[test]
    fn native_threads_must_be_positive() {
        let args = parse_synthesis_args(&["--engine", "native", "--native-threads", "0"]);
        let err = validate_native_args(&args).unwrap_err().to_string();
        assert!(err.contains("between 1 and 256"), "{err}");
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
    fn native_kitten_tts2_rejects_seed_above_signed_range() {
        let args = parse_synthesis_args(&[
            "--engine",
            "native",
            "--native-family",
            "kitten_tts2",
            "--native-model",
            "/tmp/kitten-tts2.gguf",
            "--accept-kitten2-license",
            "--seed",
            "9223372036854775808",
        ]);
        let err = validate_native_args(&args).unwrap_err().to_string();
        assert!(err.contains("signed 64-bit"), "{err}");
        assert!(err.contains("9223372036854775807"), "{err}");
    }

    #[test]
    fn native_kitten_tts_accepts_full_u64_seed_range() {
        let args = parse_synthesis_args(&[
            "--engine",
            "native",
            "--model",
            "mini-0.8",
            "--seed",
            "18446744073709551615",
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
        fs::write(&existing, b"GGUFcached")?;

        let client = build_http_client()?;
        let path = ensure_native_kitten_tts_model(&client, dir.path(), false)?;
        assert_eq!(path, existing);
        assert_eq!(fs::read(&path)?, b"GGUFcached");
        Ok(())
    }

    #[test]
    fn native_fetch_rejects_corrupt_cached_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let existing = dir.path().join("native").join(NATIVE_KITTEN_TTS_FILE);
        fs::create_dir_all(existing.parent().expect("parent"))?;
        fs::write(&existing, b"<html>not a model</html>")?;

        let client = build_http_client()?;
        let err = ensure_native_kitten_tts_model(&client, dir.path(), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unusable"), "{err}");
        assert!(err.contains("--force"), "{err}");
        Ok(())
    }

    #[test]
    fn download_size_check_rejects_empty_and_truncated() {
        assert!(verify_download_size(Some(10), 10, "url").is_ok());
        assert!(verify_download_size(None, 10, "url").is_ok());

        let err = verify_download_size(Some(10), 4, "url")
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "{err}");
        assert!(err.contains("expected 10 bytes, received 4"), "{err}");

        let err = verify_download_size(Some(0), 0, "url")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no data"), "{err}");
    }

    #[test]
    fn gguf_validation_checks_magic() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let good = dir.path().join("good.gguf");
        fs::write(&good, b"GGUFpayload")?;
        validate_native_gguf(&good)?;

        let bad = dir.path().join("bad.gguf");
        fs::write(&bad, b"<html>")?;
        let err = validate_native_gguf(&bad).unwrap_err().to_string();
        assert!(err.contains("not a GGUF"), "{err}");

        let empty = dir.path().join("empty.gguf");
        fs::write(&empty, b"")?;
        assert!(validate_native_gguf(&empty).is_err());
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
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("does not exist"), "{err}");
    }

    #[test]
    fn explicit_native_espeak_paths_require_correct_types() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("libespeak-ng");
        fs::write(&file, b"fixture")?;
        assert!(
            resolve_espeak_path(Some(&file), detect_espeak_library, "libespeak-ng", false)?
                .is_some()
        );
        assert!(
            resolve_espeak_path(Some(dir.path()), detect_espeak_data, "espeak-ng-data", true)?
                .is_some()
        );
        let err = resolve_espeak_path(
            Some(dir.path()),
            detect_espeak_library,
            "libespeak-ng",
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("must be a file"), "{err}");
        let err = resolve_espeak_path(Some(&file), detect_espeak_data, "espeak-ng-data", true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be a directory"), "{err}");
        Ok(())
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
        fs::write(&existing, b"GGUFcached")?;

        let path = check_native_doctor_model(NativeFamily::KittenTts, None, dir.path())?;
        assert_eq!(path, existing);
        Ok(())
    }

    #[test]
    fn doctor_rejects_corrupt_cached_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let existing = dir.path().join("native").join(NATIVE_KITTEN_TTS_FILE);
        fs::create_dir_all(existing.parent().expect("parent"))?;
        fs::write(&existing, b"<html>nope</html>")?;

        let err = check_native_doctor_model(NativeFamily::KittenTts, None, dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("unusable"), "{err}");
        assert!(err.contains("--force"), "{err}");
        Ok(())
    }

    #[test]
    fn doctor_rejects_non_gguf_explicit_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let weights = dir.path().join("weights.gguf");
        fs::write(&weights, b"not a model")?;

        let err = check_native_doctor_model(NativeFamily::KittenTts2, Some(&weights), dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a GGUF"), "{err}");
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
