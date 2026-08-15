//! Buffer-fed speech recognition, callable from Rust.
//!
//! The command layer of this plugin (`startListening` & co.) lets a webview
//! drive a recognizer that owns its own microphone capture. Hosts that already
//! own the microphone — an app with a Rust audio pipeline — want the opposite
//! shape: *they* push PCM into the recognizer and get hypotheses back, without
//! any webview or IPC bridge in between. That is what [`Recognizer`] is.
//!
//! - iOS: `SFSpeechAudioBufferRecognitionRequest` fed over a C ABI implemented
//!   in `ios/Sources/SttStream.swift` (see [`crate::ios_stream`]).
//! - Desktop: an in-process Vosk recognizer over the plugin's model store.
//! - Android: not implemented (`SpeechRecognizer` cannot be fed audio before
//!   API 33, and this plugin has no host that needs it yet).
//!
//! The host obtains an instance through [`crate::SttExt::recognizer`].

use serde::Serialize;

/// Sample format every [`RecognizerSession`] accepts: mono, signed 16-bit PCM
/// at the sample rate given in [`RecognizerConfig`].
pub type Sample = i16;

/// What the recognizer heard, delivered through the callback handed to
/// [`Recognizer::open`]. Callbacks may run on any thread and must not block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecognizerEvent {
    /// Refined hypothesis for the utterance in progress. Replaces the previous
    /// `Partial`; not yet committed.
    Partial(String),
    /// A committed utterance segment. Segments are cumulative *within a
    /// platform's own semantics*: Vosk emits one per detected utterance and
    /// hosts append them; Apple's recognizer emits the whole session's
    /// transcript once, on `finish`.
    Final(String),
    /// The recognizer failed. `Ended` follows.
    Error(String),
    /// No further events will be delivered for this session.
    Ended,
}

pub type RecognizerCallback = Box<dyn Fn(RecognizerEvent) + Send + Sync + 'static>;

/// How to open a session.
#[derive(Debug, Clone)]
pub struct RecognizerConfig {
    /// BCP-47 language tag ("de-DE"). Bare codes ("de") are widened by the
    /// platform binding where it can.
    pub language: String,
    /// Sample rate of the PCM the host will feed.
    pub sample_rate: u32,
    /// Whether to report `Partial` hypotheses.
    pub interim_results: bool,
    /// iOS: require on-device recognition (offline; may be less accurate).
    /// Ignored on desktop, where recognition is always local.
    pub on_device: bool,
}

/// An open, buffer-fed recognition session.
///
/// Feed audio with [`feed`](Self::feed); events arrive on the callback given
/// to [`Recognizer::open`]. End the session with either [`finish`](Self::finish)
/// (flush and wait for the final transcript) or [`cancel`](Self::cancel).
/// Dropping a session without either behaves like `cancel`.
pub trait RecognizerSession: Send {
    /// Push more audio. Non-blocking; safe to call from a realtime-ish thread.
    fn feed(&mut self, samples: &[Sample]);
    /// No more audio: finalize. The final transcript arrives on the callback
    /// as `Final` (possibly preceded by more `Partial`s), then `Ended`.
    fn finish(self: Box<Self>);
    /// Abort; the callback receives `Ended` and no transcript.
    fn cancel(self: Box<Self>);
}

/// Availability of the recognizer for a language, as the host's settings UI
/// wants to show it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognizerStatus {
    /// Whether recognition can work on this platform at all.
    pub available: bool,
    /// Why not, when `available` is false.
    pub reason: Option<String>,
    /// Whether the language's model is present. Always `true` on platforms
    /// where the OS ships the recognizer (iOS).
    pub model_installed: bool,
    /// Whether the platform needs a model download step at all.
    pub needs_model_download: bool,
}

/// Factory for buffer-fed recognition sessions.
///
/// Registered in Tauri state by [`crate::init`]; hosts reach it through
/// [`crate::SttExt::recognizer`].
pub trait Recognizer: Send + Sync {
    /// Open a session. **Blocking**: on desktop this may download and load a
    /// model (progress is emitted as `stt://download-progress`), on iOS it may
    /// wait for the speech-recognition authorization prompt. Call it from a
    /// blocking-capable thread.
    fn open(
        &self,
        config: RecognizerConfig,
        on_event: RecognizerCallback,
    ) -> crate::Result<Box<dyn RecognizerSession>>;

    /// Availability and model status for `language`. Non-blocking.
    fn status(&self, language: &str) -> RecognizerStatus;

    /// Make sure the model for `language` is present, downloading it if
    /// needed. **Blocking**; progress is emitted as `stt://download-progress`.
    /// A no-op on platforms without a model download step.
    fn install_model(&self, language: &str) -> crate::Result<()>;

    /// Ask the OS for speech-recognition permission if it has not been decided
    /// yet. **Blocking** until the user answers. Returns whether recognition
    /// is authorized. Always `Ok(true)` on desktop.
    fn request_authorization(&self) -> crate::Result<bool>;
}
