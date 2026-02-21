use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use directories::ProjectDirs;
use ndarray::{Array1, Array2};
use once_cell::sync::{Lazy, OnceCell};
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use regex::Regex;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::ZipArchive;

const DEFAULT_REPO_ID: &str = "KittenML/kitten-tts-nano-0.8-fp32";
const DEFAULT_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_MAX_CHARS: usize = 400;
const DEFAULT_TRIM_TAIL: usize = 5_000;

static TOKEN_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w+|[^\w\s]").expect("valid regex"));
static SENTENCE_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[.!?]+").expect("valid regex"));
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
}

#[derive(Debug, Args)]
struct SynthesizeArgs {
    #[command(flatten)]
    model: ModelSelection,
    #[command(flatten)]
    text: TextInput,
    #[arg(long, default_value = "Leo")]
    voice: String,
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
    #[arg(long, value_enum, default_value_t = PhonemizerMode::Auto)]
    phonemizer: PhonemizerMode,
    #[arg(long, default_value = "output.wav")]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = WavEncoding::Pcm16)]
    wav_encoding: WavEncoding,
}

#[derive(Debug, Args)]
struct StreamArgs {
    #[command(flatten)]
    model: ModelSelection,
    #[command(flatten)]
    text: TextInput,
    #[arg(long, default_value = "Leo")]
    voice: String,
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
    #[arg(long, value_enum, default_value_t = PhonemizerMode::Auto)]
    phonemizer: PhonemizerMode,
}

#[derive(Debug, Args)]
struct PlayArgs {
    #[command(flatten)]
    model: ModelSelection,
    #[command(flatten)]
    text: TextInput,
    #[arg(long, default_value = "Leo")]
    voice: String,
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
    #[arg(long, value_enum, default_value_t = PhonemizerMode::Auto)]
    phonemizer: PhonemizerMode,
    #[arg(long, value_enum, default_value_t = PlayerMode::Auto)]
    player: PlayerMode,
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

#[derive(Debug, Clone, Args)]
struct ModelSelection {
    #[arg(long, value_enum)]
    model: Option<ModelPreset>,
    #[arg(long)]
    repo_id: Option<String>,
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
enum PhonemizerMode {
    Auto,
    EspeakNg,
    Espeak,
    Basic,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PlayerMode {
    Auto,
    Ffplay,
    PwPlay,
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
        if let Some(cache_dir) = &self.cache_dir {
            return Ok(cache_dir.clone());
        }
        let dirs = ProjectDirs::from("io", "KittenML", "kitten-tts")
            .ok_or_else(|| anyhow!("could not determine platform cache directory"))?;
        Ok(dirs.cache_dir().to_path_buf())
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
            entry.read_to_end(&mut raw)?;
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
        let punctuation = ";:,.!?¡¿—…\"«»\"\" ";
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

    fn encode(&self, text: &str) -> Vec<i64> {
        let mut indexes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            if let Some(index) = self.word_index_dictionary.get(&ch) {
                indexes.push(*index);
            }
        }
        indexes
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

#[derive(Debug)]
struct BasicPhonemizer;

impl Phonemizer for BasicPhonemizer {
    fn name(&self) -> &str {
        "basic"
    }

    fn phonemize(&self, text: &str) -> Result<String> {
        let normalized = SPACES_RE.replace_all(text.trim(), " ").to_string();
        if normalized.is_empty() {
            bail!("basic phonemizer received empty text")
        }
        Ok(normalized)
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
            eprintln!(
                "warning: no espeak binary found; falling back to basic grapheme mode (lower quality)"
            );
            Ok(Box::new(BasicPhonemizer))
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
        PhonemizerMode::Basic => Ok(Box::new(BasicPhonemizer)),
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
}

impl Synthesizer {
    fn new(artifacts: &ModelArtifacts) -> Result<Self> {
        init_ort()?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
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
        })
    }

    fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        max_chars: usize,
        trim_tail: usize,
        style_index: Option<usize>,
        phonemizer: &dyn Phonemizer,
    ) -> Result<Vec<f32>> {
        let cleaned = clean_text_basic(text);
        if cleaned.is_empty() {
            bail!("input text is empty after basic normalization");
        }

        let chunks = chunk_text(&cleaned, max_chars);
        if chunks.is_empty() {
            bail!("input text produced zero chunks");
        }

        let canonical_voice = self.resolve_voice(voice)?;

        let mut output = Vec::new();
        for chunk in chunks {
            let mut chunk_audio = self.synthesize_single_chunk(
                &chunk,
                &canonical_voice,
                speed,
                trim_tail,
                style_index,
                phonemizer,
            )?;
            output.append(&mut chunk_audio);
        }

        Ok(output)
    }

    fn resolve_voice(&self, voice: &str) -> Result<String> {
        let resolved = self
            .voice_aliases
            .get(voice)
            .cloned()
            .unwrap_or_else(|| voice.to_string());

        if self.voice_table.voices.contains_key(&resolved) {
            return Ok(resolved);
        }

        bail!("voice '{voice}' is not available. Run `kitten-tts voices` to list voices")
    }

    fn synthesize_single_chunk(
        &mut self,
        chunk: &str,
        voice: &str,
        speed: f32,
        trim_tail: usize,
        style_index: Option<usize>,
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

        tokens.insert(0, 0);
        tokens.push(0);

        let max_rows = self.voice_table.max_style_rows(voice)?;
        let ref_id = resolve_style_index(chunk.len(), max_rows, style_index);
        let style = self.voice_table.style_row(voice, ref_id)?.to_vec();
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

        let (_, data) = outputs[0]
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
    text_len: usize,
    max_rows: usize,
    explicit_style_index: Option<usize>,
) -> usize {
    if max_rows == 0 {
        return 0;
    }
    let max_index = max_rows.saturating_sub(1);
    match explicit_style_index {
        Some(i) => usize::min(i, max_index),
        None => usize::min(text_len, max_index),
    }
}

fn init_ort() -> Result<()> {
    ORT_INIT.get_or_init(|| {
        let _ = ort::init().with_name("kitten-tts").commit();
    });
    Ok(())
}

fn tokenize_phonemes(cleaner: &TextCleaner, phonemes: &str) -> Vec<i64> {
    let mut tokens = Vec::new();
    let pieces: Vec<&str> = TOKEN_SPLIT_RE
        .find_iter(phonemes)
        .map(|m| m.as_str())
        .collect();
    if pieces.is_empty() {
        return tokens;
    }

    let joined = pieces.join(" ");
    tokens.extend(cleaner.encode(&joined));
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

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    for part in SENTENCE_SPLIT_RE.split(text) {
        let sentence = part.trim();
        if sentence.is_empty() {
            continue;
        }

        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(sentence));
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

    let config_path = model_dir.join("config.json");
    if force || !config_path.exists() {
        download_repo_file(client, repo_id, "config.json", &config_path)
            .with_context(|| format!("failed downloading config.json for {repo_id}"))?;
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
    }

    if force || !voices_path.exists() {
        download_repo_file(client, repo_id, voices_file, &voices_path)
            .with_context(|| format!("failed downloading voices file {voices_file}"))?;
    }

    let manifest = CacheManifest {
        repo_id: repo_id.to_string(),
        fetched_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model_file: model_file.to_string(),
        voices_file: voices_file.to_string(),
    };
    let manifest_path = model_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("failed writing manifest at {}", manifest_path.display()))?;

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

fn write_pcm_f32_stdout(audio: &[f32]) -> Result<()> {
    let mut stdout = io::stdout().lock();
    for sample in audio {
        stdout.write_all(&sample.to_le_bytes())?;
    }
    stdout.flush()?;
    Ok(())
}

fn play_audio(audio: &[f32], sample_rate: u32, player_mode: PlayerMode) -> Result<()> {
    let player = detect_player(player_mode)?;
    let temp_path = std::env::temp_dir().join(format!(
        "kitten-tts-play-{}-{}.wav",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));

    write_wav(&temp_path, sample_rate, WavEncoding::Pcm16, audio)?;
    let status = match player {
        "ffplay" => Command::new("ffplay")
            .args(["-autoexit", "-nodisp", "-loglevel", "error"])
            .arg(&temp_path)
            .status()
            .context("failed to launch ffplay")?,
        "pw-play" => Command::new("pw-play")
            .arg(&temp_path)
            .status()
            .context("failed to launch pw-play")?,
        _ => bail!("unsupported player selected"),
    };

    let _ = fs::remove_file(&temp_path);
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
            (h, 10)
        }
        2 | 3 => {
            let h = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (h, 12)
        }
        other => bail!("unsupported npy version {other}"),
    };

    let header_end = header_offset + header_len;
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

    let data_bytes = &bytes[header_end..];
    if data_bytes.len() != item_count * 4 {
        bail!(
            "npy data size mismatch: expected {} bytes, got {}",
            item_count * 4,
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

fn build_http_client() -> Result<Client> {
    Client::builder()
        .user_agent("kitten-tts-rs/0.1.0")
        .build()
        .context("failed building HTTP client")
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
            }
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
            let text = read_text_input(&args.text)?;
            let client = build_http_client()?;
            let cache_root = args.model.resolve_cache_dir()?;
            let repo_id = args.model.resolve_repo_id();
            let artifacts = ensure_model_cached(&client, &cache_root, &repo_id, false)?;
            let mut synthesizer = Synthesizer::new(&artifacts)?;
            let phonemizer = detect_phonemizer(args.phonemizer)?;

            let audio = synthesizer.synthesize(
                &text,
                &args.voice,
                args.speed,
                args.max_chars,
                args.trim_tail,
                args.style_index,
                phonemizer.as_ref(),
            )?;
            play_audio(&audio, args.sample_rate, args.player)?;
        }
        Commands::Synthesize(args) => {
            let text = read_text_input(&args.text)?;
            let client = build_http_client()?;
            let cache_root = args.model.resolve_cache_dir()?;
            let repo_id = args.model.resolve_repo_id();
            let artifacts = ensure_model_cached(&client, &cache_root, &repo_id, false)?;
            let mut synthesizer = Synthesizer::new(&artifacts)?;
            let phonemizer = detect_phonemizer(args.phonemizer)?;

            let audio = synthesizer.synthesize(
                &text,
                &args.voice,
                args.speed,
                args.max_chars,
                args.trim_tail,
                args.style_index,
                phonemizer.as_ref(),
            )?;

            write_wav(&args.output, args.sample_rate, args.wav_encoding, &audio)?;
            eprintln!("wrote {} samples to {}", audio.len(), args.output.display());
        }
        Commands::Stream(args) => {
            let text = read_text_input(&args.text)?;
            let client = build_http_client()?;
            let cache_root = args.model.resolve_cache_dir()?;
            let repo_id = args.model.resolve_repo_id();
            let artifacts = ensure_model_cached(&client, &cache_root, &repo_id, false)?;
            let mut synthesizer = Synthesizer::new(&artifacts)?;
            let phonemizer = detect_phonemizer(args.phonemizer)?;

            let audio = synthesizer.synthesize(
                &text,
                &args.voice,
                args.speed,
                args.max_chars,
                args.trim_tail,
                args.style_index,
                phonemizer.as_ref(),
            )?;
            write_pcm_f32_stdout(&audio)?;
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
        };
        assert_eq!(selection.resolve_repo_id(), DEFAULT_REPO_ID);
    }

    #[test]
    fn repo_id_overrides_model_preset() {
        let selection = ModelSelection {
            model: Some(ModelPreset::Micro08),
            repo_id: Some("KittenML/custom".to_string()),
            cache_dir: None,
        };
        assert_eq!(selection.resolve_repo_id(), "KittenML/custom");
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
    fn clean_text_collapses_whitespace() {
        let text = "  hello   world\n\tfoo  ";
        assert_eq!(clean_text_basic(text), "hello world foo");
    }

    #[test]
    fn resolve_style_index_uses_text_len_by_default() {
        assert_eq!(resolve_style_index(25, 400, None), 25);
        assert_eq!(resolve_style_index(999, 400, None), 399);
    }

    #[test]
    fn resolve_style_index_honors_override_and_clamps() {
        assert_eq!(resolve_style_index(10, 400, Some(4)), 4);
        assert_eq!(resolve_style_index(10, 400, Some(9999)), 399);
    }
}
