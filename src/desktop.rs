use serde::de::DeserializeOwned;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{plugin::PluginApi, AppHandle, Emitter, Manager, Runtime};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use md5::{Digest, Md5};
use vosk::{Model, Recognizer};

use crate::models::*;

/// A downloadable Vosk model.
///
/// `md5` and `size` are the values alphacephei.com publishes for the archive in
/// `model-list.json`. They are pinned here on purpose: they let us tell a
/// finished download from a truncated one *before* anything is unpacked, and
/// they detect a swapped or corrupted archive.
pub(crate) struct ModelSpec {
    lang: &'static str,
    name: &'static str,
    url: &'static str,
    /// MD5 of the .zip archive, as published upstream.
    md5: &'static str,
    /// Size of the .zip archive in bytes, as published upstream.
    size: u64,
}

impl ModelSpec {
    /// True if this model is completely installed under `models_dir`.
    pub(crate) fn is_installed_in(&self, models_dir: &Path) -> bool {
        model_is_complete(&models_dir.join(self.name), self.md5)
    }
}

/// Written inside a model directory once its archive was fully downloaded,
/// checksum-verified and extracted. The file holds the verified MD5, so an
/// interrupted install leaves no marker and is never mistaken for a usable
/// model. This is the ONLY thing that makes a model count as installed.
const MARKER_FILE: &str = ".verified-md5";

/// Holds partial downloads and extraction staging, next to the models.
const STAGING_SUBDIR: &str = ".incomplete";

/// Percentage the download itself accounts for; checksum and extraction fill
/// the rest so the UI keeps moving through all three phases.
const DOWNLOAD_PROGRESS_SHARE: u8 = 45;
const EXTRACT_PROGRESS_START: u8 = 50;

/// Available Vosk models with their download URLs
/// Using high-accuracy models for better transcription quality
const AVAILABLE_MODELS: &[ModelSpec] = &[
    ModelSpec {
        lang: "en-US",
        name: "vosk-model-en-us-0.42-gigaspeech",
        url: "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
        md5: "db1202c15b40ea4b1ec27b85a90dffbe",
        size: 2_423_807_363,
    },
    ModelSpec {
        lang: "pt-BR",
        name: "vosk-model-pt-fb-v0.1.1-20220516_2113",
        url: "https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip",
        md5: "5d259cb674fd52a61c97c1e53282d4b3",
        size: 1_693_530_883,
    },
    ModelSpec {
        lang: "es-ES",
        name: "vosk-model-es-0.42",
        url: "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
        md5: "83f83e045a4537c53ed5ed42f959bf6d",
        size: 1_484_681_703,
    },
    ModelSpec {
        lang: "fr-FR",
        name: "vosk-model-fr-0.22",
        url: "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip",
        md5: "12662b25b4d35059ec05e3a75a27841e",
        size: 1_523_026_348,
    },
    ModelSpec {
        lang: "de-DE",
        name: "vosk-model-de-0.21",
        url: "https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip",
        md5: "23298ddeb602739016956144ac4c74de",
        size: 2_031_717_803,
    },
    ModelSpec {
        lang: "ru-RU",
        name: "vosk-model-ru-0.42",
        url: "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
        md5: "ae356c0fc8879deed1982d879f40880a",
        size: 1_937_602_113,
    },
    ModelSpec {
        lang: "zh-CN",
        name: "vosk-model-cn-0.22",
        url: "https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip",
        md5: "c050f6849398ceecfa723cca69b8c67d",
        size: 1_358_736_686,
    },
    ModelSpec {
        lang: "ja-JP",
        name: "vosk-model-ja-0.22",
        url: "https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip",
        md5: "e7ab21ff213aff2edf1f04724487846c",
        size: 1_045_975_323,
    },
    ModelSpec {
        lang: "it-IT",
        name: "vosk-model-it-0.22",
        url: "https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip",
        md5: "973cf0adc17ea0acd042d079bba67a94",
        size: 1_306_765_292,
    },
];

/// The model used when a language has no entry of its own.
const DEFAULT_MODEL: &ModelSpec = &AVAILABLE_MODELS[0];

/// True if `model_path` holds a completely installed model for `expected_md5`.
pub(crate) fn model_is_complete(model_path: &Path, expected_md5: &str) -> bool {
    fs::read_to_string(model_path.join(MARKER_FILE))
        .map(|recorded| recorded.trim().eq_ignore_ascii_case(expected_md5))
        .unwrap_or(false)
}

/// Streams `path` through MD5 without holding it in memory.
fn md5_of_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 1024 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Finds the model directory inside a freshly extracted archive. Vosk archives
/// carry a single top-level directory named after the model.
fn locate_model_root(extract_dir: &Path, model_name: &str) -> crate::Result<PathBuf> {
    let direct = extract_dir.join(model_name);
    if direct.is_dir() {
        return Ok(direct);
    }

    let mut dirs: Vec<PathBuf> = fs::read_dir(extract_dir)
        .map_err(|e| crate::Error::Recording(format!("Failed to read extracted archive: {}", e)))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();

    match dirs.len() {
        1 => Ok(dirs.remove(0)),
        // No directory at all: the archive put the model files at its root.
        0 => Ok(extract_dir.to_path_buf()),
        _ => Err(crate::Error::Recording(format!(
            "Archive for '{}' has an unexpected layout ({} top-level directories)",
            model_name,
            dirs.len()
        ))),
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Session counter - incremented each time a new listening session starts.
/// Audio callbacks capture their session ID and only process audio if it matches
/// the current session. This prevents old audio data from bleeding into new sessions.
static CURRENT_SESSION_ID: AtomicU64 = AtomicU64::new(0);

/// Shared audio processing state that can be reused across sessions.
/// This avoids creating new audio streams for each PTT press.
struct AudioProcessor {
    /// The audio buffer accumulating samples
    buffer: Vec<i16>,
    /// The Vosk recognizer
    recognizer: Recognizer,
    /// Last emitted partial result (to avoid duplicates)
    last_partial: String,
    /// Whether to emit interim results
    interim_results: bool,
    /// Resampling step (1 = no resampling)
    resample_step: usize,
}

struct SttState {
    model: Option<Arc<Model>>,
    current_model_name: Option<String>,
    is_listening: bool,
    listen_start_time: Option<Instant>,
    max_duration_ms: Option<u64>,
    /// The session ID of the current listening session (0 = not listening)
    active_session_id: u64,
    /// Shared audio processor - reused across sessions
    audio_processor: Option<Arc<Mutex<AudioProcessor>>>,
    /// Whether the audio stream has been created
    stream_created: bool,
}

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
) -> crate::Result<Stt<R>> {
    let state = Arc::new(Mutex::new(SttState {
        model: None,
        current_model_name: None,
        is_listening: false,
        listen_start_time: None,
        max_duration_ms: None,
        active_session_id: 0,
        audio_processor: None,
        stream_created: false,
    }));

    Ok(Stt {
        app: app.clone(),
        state,
    })
}

pub struct Stt<R: Runtime> {
    app: AppHandle<R>,
    state: Arc<Mutex<SttState>>,
}

impl<R: Runtime> Clone for Stt<R> {
    fn clone(&self) -> Self {
        Self {
            app: self.app.clone(),
            state: self.state.clone(),
        }
    }
}

impl<R: Runtime> Stt<R> {
    fn complete_result_text(result: vosk::CompleteResult) -> String {
        match result {
            vosk::CompleteResult::Single(single) => single.text.to_string(),
            vosk::CompleteResult::Multiple(multiple) => multiple
                .alternatives
                .first()
                .map(|alt| alt.text.to_string())
                .unwrap_or_default(),
        }
    }

    fn emit_result(&self, result: &RecognitionResult) {
        let _ = self.app.emit("stt://result", result);
        let _ = self.app.emit("plugin:stt:result", result);
    }

    pub(crate) fn get_models_dir(&self) -> PathBuf {
        self.app
            .path()
            .app_data_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("vosk-models")
    }

    pub(crate) fn get_model_info_for_language(&self, language: &str) -> Option<&'static ModelSpec> {
        // First try exact match
        if let Some(spec) = AVAILABLE_MODELS.iter().find(|spec| spec.lang == language) {
            return Some(spec);
        }

        // If not found, try to match by language prefix (e.g., "pt" matches "pt-BR")
        if let Some(prefix) = language.split('-').next() {
            if let Some(spec) = AVAILABLE_MODELS
                .iter()
                .find(|spec| spec.lang.split('-').next() == Some(prefix))
            {
                return Some(spec);
            }
        }

        None
    }

    fn emit_progress(&self, status: &str, model: &str, progress: u8) {
        let _ = self.app.emit(
            "stt://download-progress",
            serde_json::json!({
                "status": status,
                "model": model,
                "progress": progress
            }),
        );
    }

    /// Fetches the archive into `archive_path`, resuming a partial file if one
    /// is there. Runs the blocking HTTP in its own thread to avoid tokio
    /// runtime conflicts.
    fn fetch_archive(&self, spec: &'static ModelSpec, archive_path: &Path) -> crate::Result<()> {
        let mut have = fs::metadata(archive_path).map(|m| m.len()).unwrap_or(0);

        if have > spec.size {
            // Longer than the real archive: not a prefix of it, so nothing to resume.
            println!(
                "Discarding oversized partial download for '{}' ({} > {} bytes)",
                spec.name, have, spec.size
            );
            fs::remove_file(archive_path).ok();
            have = 0;
        }

        if have == spec.size {
            println!("Archive for '{}' already complete, skipping download", spec.name);
            return Ok(());
        }

        if have > 0 {
            println!(
                "Resuming download of '{}' at {:.2} / {:.2} MB",
                spec.name,
                have as f64 / 1_048_576.0,
                spec.size as f64 / 1_048_576.0
            );
        } else {
            println!("Downloading model '{}' from {}", spec.name, spec.url);
        }

        self.emit_progress(
            "downloading",
            spec.name,
            ((have as f64 / spec.size as f64) * DOWNLOAD_PROGRESS_SHARE as f64) as u8,
        );

        let app_handle = self.app.clone();
        let target = archive_path.to_path_buf();

        let handle = std::thread::spawn(move || -> Result<(), String> {
            let client = reqwest::blocking::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                // No overall timeout: these archives are gigabytes, and an
                // interrupted transfer resumes on the next attempt anyway.
                .timeout(None)
                .build()
                .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

            let mut request = client.get(spec.url);
            if have > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={}-", have));
            }

            let response = request
                .send()
                .map_err(|e| format!("Failed to download model from {}: {}", spec.url, e))?;

            let status = response.status();
            if !status.is_success() {
                return Err(format!(
                    "Failed to download model: HTTP {} - {}",
                    status,
                    response
                        .text()
                        .unwrap_or_else(|_| "Failed to get error details".to_string())
                ));
            }

            // A server that ignores Range answers 200 with the whole file — then
            // the partial file has to go, or we would splice two copies together.
            let resuming = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
            if have > 0 && !resuming {
                println!("Server ignored the range request, restarting from zero");
            }

            let file = if resuming {
                OpenOptions::new()
                    .append(true)
                    .open(&target)
                    .map_err(|e| format!("Failed to open partial download: {}", e))?
            } else {
                File::create(&target)
                    .map_err(|e| format!("Failed to create download file: {}", e))?
            };
            let mut writer = BufWriter::with_capacity(1024 * 1024, file);

            let mut reader = response;
            let mut written: u64 = if resuming { have } else { 0 };
            let mut chunk = vec![0u8; 64 * 1024];
            let mut last_progress_mb = written / (5 * 1024 * 1024);

            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        writer
                            .write_all(&chunk[..n])
                            .map_err(|e| format!("Failed to write download: {}", e))?;
                        written += n as u64;

                        // Report progress every 5MB
                        let current_mb = written / (5 * 1024 * 1024);
                        if current_mb > last_progress_mb {
                            last_progress_mb = current_mb;
                            print!(
                                "\rProgress: {:.2} / {:.2} MB   ",
                                written as f64 / 1_048_576.0,
                                spec.size as f64 / 1_048_576.0
                            );
                            std::io::Write::flush(&mut std::io::stdout()).ok();

                            let progress = ((written as f64 / spec.size as f64)
                                * DOWNLOAD_PROGRESS_SHARE as f64)
                                as u8;
                            let _ = app_handle.emit(
                                "stt://download-progress",
                                serde_json::json!({
                                    "status": "downloading",
                                    "model": spec.name,
                                    "progress": progress
                                }),
                            );
                        }
                    }
                    Err(e) => {
                        println!(); // New line after progress
                        writer.flush().ok();
                        return Err(format!("Failed to read chunk: {}", e));
                    }
                }
            }

            writer
                .flush()
                .map_err(|e| format!("Failed to flush download: {}", e))?;

            println!(); // New line after progress bar
            println!("Download complete: {:.2} MB", written as f64 / 1_048_576.0);

            if written != spec.size {
                // Kept on disk: the next attempt resumes where this one stopped.
                return Err(format!(
                    "Download stopped short: got {} of {} bytes — retry to resume",
                    written, spec.size
                ));
            }

            Ok(())
        });

        handle
            .join()
            .map_err(|_| crate::Error::Recording("Download thread panicked".to_string()))?
            .map_err(crate::Error::Recording)?;

        Ok(())
    }

    /// Downloads (resuming if possible), verifies and extracts a model.
    ///
    /// Nothing lands under the model's own name until the archive matched its
    /// published MD5 and was fully unpacked — the last step writes the marker
    /// file that makes the model count as installed.
    fn download_model(&self, spec: &'static ModelSpec) -> crate::Result<PathBuf> {
        let models_dir = self.get_models_dir();
        fs::create_dir_all(&models_dir).map_err(|e| {
            crate::Error::Recording(format!("Failed to create models directory: {}", e))
        })?;

        let model_path = models_dir.join(spec.name);

        if model_is_complete(&model_path, spec.md5) {
            return Ok(model_path);
        }

        if model_path.exists() {
            // Left over from an interrupted install (or from before the marker
            // existed). It may be missing files, so it cannot be trusted.
            println!(
                "Discarding unverified model directory {}",
                model_path.display()
            );
            fs::remove_dir_all(&model_path).map_err(|e| {
                crate::Error::Recording(format!("Failed to remove incomplete model: {}", e))
            })?;
        }

        let staging = models_dir.join(STAGING_SUBDIR);
        fs::create_dir_all(&staging).map_err(|e| {
            crate::Error::Recording(format!("Failed to create staging directory: {}", e))
        })?;
        let archive_path = staging.join(format!("{}.zip.part", spec.name));

        self.fetch_archive(spec, &archive_path)?;

        println!("Verifying checksum...");
        self.emit_progress("verifying", spec.name, DOWNLOAD_PROGRESS_SHARE);
        let actual = md5_of_file(&archive_path).map_err(|e| {
            crate::Error::Recording(format!("Failed to read downloaded archive: {}", e))
        })?;
        if !actual.eq_ignore_ascii_case(spec.md5) {
            // A resumed transfer can only mismatch if the bytes are bad or the
            // upstream file changed — either way, resuming it further is futile.
            fs::remove_file(&archive_path).ok();
            return Err(crate::Error::Recording(format!(
                "Checksum mismatch for '{}': expected {}, got {}. Archive discarded, please retry.",
                spec.name, spec.md5, actual
            )));
        }

        println!("Extracting model...");
        self.emit_progress("extracting", spec.name, EXTRACT_PROGRESS_START);

        let extract_dir = staging.join(format!("{}-extract", spec.name));
        fs::remove_dir_all(&extract_dir).ok();
        fs::create_dir_all(&extract_dir).map_err(|e| {
            crate::Error::Recording(format!("Failed to create extraction directory: {}", e))
        })?;

        self.extract_archive(&archive_path, &extract_dir, spec)?;

        let extracted_root = locate_model_root(&extract_dir, spec.name)?;
        fs::rename(&extracted_root, &model_path).map_err(|e| {
            crate::Error::Recording(format!("Failed to move model into place: {}", e))
        })?;

        // Only now does the model count as installed.
        fs::write(model_path.join(MARKER_FILE), spec.md5).map_err(|e| {
            crate::Error::Recording(format!("Failed to write model marker: {}", e))
        })?;

        fs::remove_file(&archive_path).ok();
        fs::remove_dir_all(&extract_dir).ok();

        self.emit_progress("complete", spec.name, 100);

        Ok(model_path)
    }

    /// Unpacks the archive from disk (never held in memory — these are GBs).
    fn extract_archive(
        &self,
        archive_path: &Path,
        extract_dir: &Path,
        spec: &ModelSpec,
    ) -> crate::Result<()> {
        let file = File::open(archive_path)
            .map_err(|e| crate::Error::Recording(format!("Failed to open archive: {}", e)))?;
        let mut archive = zip::ZipArchive::new(BufReader::with_capacity(1024 * 1024, file))
            .map_err(|e| crate::Error::Recording(format!("Failed to open zip: {}", e)))?;

        let total = archive.len().max(1);
        let mut last_reported = EXTRACT_PROGRESS_START;

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| crate::Error::Recording(format!("Failed to read zip entry: {}", e)))?;

            // `enclosed_name` rejects paths that would escape the target directory.
            let outpath = match file.enclosed_name() {
                Some(path) => extract_dir.join(path),
                None => continue,
            };

            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath).ok();
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        fs::create_dir_all(p).ok();
                    }
                }
                let outfile = File::create(&outpath).map_err(|e| {
                    crate::Error::Recording(format!("Failed to create file: {}", e))
                })?;
                let mut writer = BufWriter::with_capacity(1024 * 1024, outfile);
                io::copy(&mut file, &mut writer).map_err(|e| {
                    crate::Error::Recording(format!("Failed to extract file: {}", e))
                })?;
                writer.flush().map_err(|e| {
                    crate::Error::Recording(format!("Failed to extract file: {}", e))
                })?;
            }

            let span = 99 - EXTRACT_PROGRESS_START;
            let progress =
                EXTRACT_PROGRESS_START + ((i + 1) as f64 / total as f64 * span as f64) as u8;
            if progress > last_reported {
                last_reported = progress;
                self.emit_progress("extracting", spec.name, progress);
            }
        }

        Ok(())
    }

    pub(crate) fn ensure_model(&self, language: Option<&str>) -> crate::Result<Arc<Model>> {
        let spec = language
            .and_then(|lang| self.get_model_info_for_language(lang))
            .unwrap_or(DEFAULT_MODEL);
        let model_name = spec.name;

        let mut state = self.state.lock().unwrap();

        // Check if we already have this model loaded
        if let Some(current) = &state.current_model_name {
            if current == model_name {
                if let Some(model) = &state.model {
                    return Ok(model.clone());
                }
            }
        }

        // Drop existing model if switching
        state.model = None;
        state.current_model_name = None;
        drop(state);

        // Download model if needed
        let model_path = self.download_model(spec)?;

        if !model_path.exists() {
            return Err(crate::Error::NotAvailable(format!(
                "Vosk model not found at {:?}",
                model_path
            )));
        }

        let model = Model::new(model_path.to_str().unwrap())
            .ok_or_else(|| crate::Error::Recording("Failed to load Vosk model".to_string()))?;

        let model = Arc::new(model);

        let mut state = self.state.lock().unwrap();
        state.model = Some(model.clone());
        state.current_model_name = Some(model_name.to_string());
        if let Some(processor) = &state.audio_processor {
            if let Ok(mut proc) = processor.lock() {
                let target_sample_rate = 16000.0;
                if let Some(mut recognizer) = Recognizer::new(&model, target_sample_rate) {
                    recognizer.set_max_alternatives(1);
                    recognizer.set_partial_words(proc.interim_results);
                    proc.buffer.clear();
                    proc.last_partial.clear();
                    proc.recognizer = recognizer;
                }
            }
        }

        Ok(model)
    }

    pub fn start_listening(&self, config: ListenConfig) -> crate::Result<()> {
        let model = self.ensure_model(config.language.as_deref())?;

        let mut state = self.state.lock().unwrap();

        if state.is_listening {
            return Err(crate::Error::Recording("Already listening".to_string()));
        }

        // Generate a new session ID
        let session_id = CURRENT_SESSION_ID.fetch_add(1, Ordering::SeqCst) + 1;
        state.active_session_id = session_id;

        // Store maxDuration config (in milliseconds)
        state.listen_start_time = Some(Instant::now());
        state.max_duration_ms = if config.max_duration > 0 {
            Some(config.max_duration as u64)
        } else {
            None
        };

        let interim_results = config.interim_results;

        // Check if we need to create a new stream or can reuse existing one
        let need_new_stream = !state.stream_created || state.audio_processor.is_none();

        if need_new_stream {
            // Create new audio processor and stream
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .ok_or_else(|| crate::Error::Recording("No input device available".to_string()))?;

            let stream_config = device.default_input_config().map_err(|e| {
                crate::Error::Recording(format!("Failed to get input config: {}", e))
            })?;

            let channels = stream_config.channels() as usize;
            let sample_format = stream_config.sample_format();
            let device_sample_rate = stream_config.sample_rate().0 as f32;

            // Vosk expects 16kHz
            let target_sample_rate = 16000.0;
            let mut recognizer = Recognizer::new(&model, target_sample_rate).ok_or_else(|| {
                crate::Error::Recording("Failed to create recognizer".to_string())
            })?;

            recognizer.set_max_alternatives(config.max_alternatives.unwrap_or(1) as u16);
            recognizer.set_partial_words(interim_results);

            // Simple resampling: skip samples if device rate > 16kHz
            let resample_step = (device_sample_rate / target_sample_rate) as usize;
            let resample_step = resample_step.max(1);

            let audio_processor = Arc::new(Mutex::new(AudioProcessor {
                buffer: Vec::new(),
                recognizer,
                last_partial: String::new(),
                interim_results,
                resample_step,
            }));

            state.audio_processor = Some(audio_processor.clone());

            let stt = self.app.clone();
            let processor_for_callback = audio_processor.clone();

            let process_audio = move |samples_i16: Vec<i16>| {
                // Check if this callback's session is still the active one
                let current_session = CURRENT_SESSION_ID.load(Ordering::SeqCst);
                if current_session == 0 {
                    // Session ID 0 means not listening - skip processing
                    return;
                }

                let mut processor = processor_for_callback.lock().unwrap();

                // Accumulate samples in buffer
                processor.buffer.extend_from_slice(&samples_i16);

                // Process when we have at least 0.1 seconds of audio after resampling
                let required_samples = (1600 * processor.resample_step).max(3200);

                if processor.buffer.len() < required_samples {
                    return;
                }

                // Take all accumulated samples
                let samples_to_process: Vec<i16> = processor.buffer.drain(..).collect();

                // Resample if needed
                let resampled: Vec<i16> = if processor.resample_step > 1 {
                    samples_to_process
                        .iter()
                        .step_by(processor.resample_step)
                        .copied()
                        .collect()
                } else {
                    samples_to_process
                };

                // Accept waveform returns Result<DecodingState, _>
                let result = processor.recognizer.accept_waveform(&resampled);
                let is_final = matches!(result, Ok(vosk::DecodingState::Finalized));

                if is_final {
                    let text = Self::complete_result_text(processor.recognizer.result());

                    if !text.is_empty() {
                        processor.last_partial = String::new();

                        let result = RecognitionResult {
                            transcript: text,
                            is_final: true,
                            confidence: Some(1.0),
                        };
                        let _ = stt.emit("stt://result", &result);
                        let _ = stt.emit("plugin:stt:result", &result);
                    }
                } else if processor.interim_results {
                    let partial = processor.recognizer.partial_result();
                    let partial_text = partial.partial.to_string();
                    if !partial_text.is_empty() && processor.last_partial != partial_text {
                        processor.last_partial = partial_text.clone();

                        let result = RecognitionResult {
                            transcript: partial_text,
                            is_final: false,
                            confidence: None,
                        };
                        let _ = stt.emit("stt://result", &result);
                        let _ = stt.emit("plugin:stt:result", &result);
                    }
                }
            };

            let stream = match sample_format {
                cpal::SampleFormat::F32 => device.build_input_stream(
                    &stream_config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let mono_i16: Vec<i16> = if channels == 1 {
                            data.iter()
                                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                                .collect()
                        } else {
                            data.chunks(channels)
                                .map(|frame| {
                                    let avg = frame.iter().sum::<f32>() / channels as f32;
                                    (avg.clamp(-1.0, 1.0) * 32767.0) as i16
                                })
                                .collect()
                        };
                        process_audio(mono_i16);
                    },
                    move |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                ),
                cpal::SampleFormat::I16 => device.build_input_stream(
                    &stream_config.into(),
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let mono_i16: Vec<i16> = if channels == 1 {
                            data.to_vec()
                        } else {
                            data.chunks(channels)
                                .map(|frame| {
                                    let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                                    (sum / channels as i32) as i16
                                })
                                .collect()
                        };
                        process_audio(mono_i16);
                    },
                    move |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                ),
                cpal::SampleFormat::U16 => device.build_input_stream(
                    &stream_config.into(),
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        let mono_i16: Vec<i16> = if channels == 1 {
                            data.iter().map(|&s| (s as i32 - 32768) as i16).collect()
                        } else {
                            data.chunks(channels)
                                .map(|frame| {
                                    let avg = frame.iter().map(|&s| s as i32).sum::<i32>()
                                        / channels as i32;
                                    (avg - 32768) as i16
                                })
                                .collect()
                        };
                        process_audio(mono_i16);
                    },
                    move |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                ),
                _ => {
                    return Err(crate::Error::Recording(format!(
                        "Unsupported sample format: {:?}",
                        sample_format
                    )));
                }
            }
            .map_err(|e| crate::Error::Recording(format!("Failed to build stream: {}", e)))?;

            stream
                .play()
                .map_err(|e| crate::Error::Recording(format!("Failed to start stream: {}", e)))?;

            state.stream_created = true;

            // Keep the stream alive for the process lifetime.
            // The callback consults CURRENT_SESSION_ID and only processes audio
            // when a session is active. Model switches reuse the existing processor
            // instead of spawning a new stream.
            std::mem::forget(stream);
        } else {
            // Reuse existing stream - just reset the audio processor state
            if let Some(processor) = &state.audio_processor {
                let mut proc = processor.lock().unwrap();
                // Clear accumulated audio buffer from previous session
                proc.buffer.clear();
                // Clear last partial to avoid duplicate detection issues
                proc.last_partial.clear();
                // Reset the recognizer to clear any accumulated state
                // Note: Vosk doesn't have a reset method, so we create a new one
                let target_sample_rate = 16000.0;
                if let Some(ref model) = state.model {
                    if let Some(mut new_recognizer) = Recognizer::new(model, target_sample_rate) {
                        new_recognizer
                            .set_max_alternatives(config.max_alternatives.unwrap_or(1) as u16);
                        new_recognizer.set_partial_words(interim_results);
                        proc.recognizer = new_recognizer;
                    }
                }
                proc.interim_results = interim_results;
            }
        }

        state.is_listening = true;

        // Emit stateChange event with RecognitionStatus
        let _ = self.app.emit(
            "plugin:stt:stateChange",
            RecognitionStatus {
                state: RecognitionState::Listening,
                is_available: true,
                language: config.language.clone(),
            },
        );

        // Start maxDuration timer thread if configured
        if config.max_duration > 0 {
            let max_ms = config.max_duration as u64;
            let app_handle_timer = self.app.clone();
            let state_clone = self.state.clone();
            let timer_session_id = session_id;
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(max_ms));

                // Check if this timer's session is still active
                let mut state = state_clone.lock().unwrap();
                if state.is_listening && state.active_session_id == timer_session_id {
                    // Set session to 0 to stop audio processing
                    CURRENT_SESSION_ID.store(0, Ordering::SeqCst);
                    state.is_listening = false;
                    state.listen_start_time = None;
                    state.max_duration_ms = None;
                    state.active_session_id = 0;

                    // Emit events
                    let _ = app_handle_timer.emit(
                        "plugin:stt:stateChange",
                        RecognitionStatus {
                            state: RecognitionState::Idle,
                            is_available: true,
                            language: None,
                        },
                    );
                    let _ = app_handle_timer.emit(
                        "stt://error",
                        serde_json::json!({
                            "error": "Maximum duration reached",
                            "code": -2
                        }),
                    );
                }
            });
        }

        Ok(())
    }

    pub fn stop_listening(
        &self,
        config: StopListeningConfig,
    ) -> crate::Result<Option<RecognitionResult>> {
        if config.post_roll_ms > 0 {
            std::thread::sleep(Duration::from_millis(config.post_roll_ms as u64));
        }

        let mut state = self.state.lock().unwrap();

        if !state.is_listening {
            return Ok(None);
        }

        let final_result = state.audio_processor.as_ref().and_then(|processor| {
            let mut proc = processor.lock().ok()?;

            if !proc.buffer.is_empty() {
                let samples_to_process: Vec<i16> = proc.buffer.drain(..).collect();
                let resampled: Vec<i16> = if proc.resample_step > 1 {
                    samples_to_process
                        .iter()
                        .step_by(proc.resample_step)
                        .copied()
                        .collect()
                } else {
                    samples_to_process
                };
                let _ = proc.recognizer.accept_waveform(&resampled);
            }

            let text = Self::complete_result_text(proc.recognizer.final_result());
            proc.last_partial.clear();

            if text.is_empty() {
                None
            } else {
                Some(RecognitionResult {
                    transcript: text,
                    is_final: true,
                    confidence: Some(1.0),
                })
            }
        });

        // Set session to 0 to signal audio callback to stop processing
        // (but the stream itself keeps running for reuse)
        CURRENT_SESSION_ID.store(0, Ordering::SeqCst);

        state.is_listening = false;
        state.listen_start_time = None;
        state.max_duration_ms = None;
        state.active_session_id = 0;
        drop(state);

        if let Some(result) = final_result.as_ref() {
            self.emit_result(result);
        }

        // Emit stateChange event
        let _ = self.app.emit(
            "plugin:stt:stateChange",
            RecognitionStatus {
                state: RecognitionState::Idle,
                is_available: true,
                language: None,
            },
        );

        Ok(final_result)
    }

    pub fn is_available(&self) -> crate::Result<AvailabilityResponse> {
        let available = cpal::default_host().default_input_device().is_some();
        Ok(AvailabilityResponse {
            available,
            reason: if available {
                None
            } else {
                Some("No input audio device available".to_string())
            },
        })
    }

    pub fn get_supported_languages(&self) -> crate::Result<SupportedLanguagesResponse> {
        let models_dir = self.get_models_dir();

        let languages: Vec<SupportedLanguage> = AVAILABLE_MODELS
            .iter()
            .map(|spec| {
                let installed = spec.is_installed_in(&models_dir);
                SupportedLanguage {
                    code: spec.lang.to_string(),
                    name: get_language_display_name(spec.lang),
                    installed: Some(installed),
                }
            })
            .collect();

        Ok(SupportedLanguagesResponse { languages })
    }

    pub fn check_permission(&self) -> crate::Result<PermissionResponse> {
        Ok(PermissionResponse {
            microphone: PermissionStatus::Granted,
            speech_recognition: PermissionStatus::Granted,
        })
    }

    pub fn request_permission(&self) -> crate::Result<PermissionResponse> {
        Ok(PermissionResponse {
            microphone: PermissionStatus::Granted,
            speech_recognition: PermissionStatus::Granted,
        })
    }
}

fn get_language_display_name(code: &str) -> String {
    match code {
        "en-US" => "English (United States)".to_string(),
        "pt-BR" => "Portuguese (Brazil)".to_string(),
        "es-ES" => "Spanish (Spain)".to_string(),
        "fr-FR" => "French (France)".to_string(),
        "de-DE" => "German (Germany)".to_string(),
        "ru-RU" => "Russian (Russia)".to_string(),
        "zh-CN" => "Chinese (Simplified)".to_string(),
        "ja-JP" => "Japanese (Japan)".to_string(),
        "it-IT" => "Italian (Italy)".to_string(),
        _ => code.to_string(),
    }
}
