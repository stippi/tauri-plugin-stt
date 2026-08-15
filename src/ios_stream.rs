//! iOS [`Recognizer`]: `SFSpeechAudioBufferRecognitionRequest` fed over the
//! C ABI implemented in `ios/Sources/SttStream.swift`.
//!
//! Rust mints session ids and registers the host's callback under them; the
//! Swift side reports back through the `rust_stt_stream_on_*` exports below.
//! Callbacks for ids that are no longer registered (finished, cancelled,
//! dropped) are ignored, so a late delegate callback can never reach a dead
//! session.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tauri::Runtime;

use crate::mobile::Stt;
use crate::recognizer::{
    Recognizer, RecognizerCallback, RecognizerConfig, RecognizerEvent, RecognizerSession,
    RecognizerStatus, Sample,
};

extern "C" {
    fn stt_stream_authorization_status() -> i32;
    fn stt_stream_request_authorization() -> i32;
    fn stt_stream_start(
        id: u64,
        locale: *const c_char,
        sample_rate: f64,
        interim_results: u32,
        on_device: u32,
    ) -> i32;
    fn stt_stream_feed(id: u64, samples: *const i16, count: u32);
    fn stt_stream_finish(id: u64);
    fn stt_stream_cancel(id: u64);
}

// Authorization codes (mirror `stt_stream_authorization_status`).
const AUTH_UNDETERMINED: i32 = 0;
const AUTH_AUTHORIZED: i32 = 1;

// Start error codes (mirror Swift `StreamStartError`).
fn describe_start_error(code: i32) -> String {
    match code {
        1 => "Locale not supported by the speech recognizer".to_string(),
        2 => "Speech recognizer unavailable (offline or restricted)".to_string(),
        3 => "Speech recognition not authorized".to_string(),
        4 => "Unsupported audio format".to_string(),
        5 => "Session id already running".to_string(),
        other => format!("Speech recognizer failed to start (code {other})"),
    }
}

type SharedCallback = Arc<RecognizerCallback>;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn sessions() -> &'static Mutex<HashMap<u64, SharedCallback>> {
    static SESSIONS: OnceLock<Mutex<HashMap<u64, SharedCallback>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Look the callback up without holding the lock while calling it — the
/// host's callback may itself touch this plugin.
fn callback_for(id: u64) -> Option<SharedCallback> {
    sessions().lock().ok()?.get(&id).cloned()
}

fn deliver(id: u64, event: RecognizerEvent) {
    if let Some(cb) = callback_for(id) {
        cb(event);
    }
}

fn c_string_arg(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: Swift passes a NUL-terminated C string that lives for the
    // duration of the call.
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

#[no_mangle]
pub extern "C" fn rust_stt_stream_on_partial(id: u64, text: *const c_char) {
    deliver(id, RecognizerEvent::Partial(c_string_arg(text)));
}

#[no_mangle]
pub extern "C" fn rust_stt_stream_on_final(id: u64, text: *const c_char) {
    deliver(id, RecognizerEvent::Final(c_string_arg(text)));
}

#[no_mangle]
pub extern "C" fn rust_stt_stream_on_error(id: u64, message: *const c_char) {
    deliver(id, RecognizerEvent::Error(c_string_arg(message)));
}

#[no_mangle]
pub extern "C" fn rust_stt_stream_on_ended(id: u64) {
    // Unregister first so nothing after `Ended` can reach the host.
    let cb = sessions().lock().ok().and_then(|mut s| s.remove(&id));
    if let Some(cb) = cb {
        cb(RecognizerEvent::Ended);
    }
}

struct IosSession {
    id: u64,
    closed: bool,
}

impl IosSession {
    fn close(&mut self, finish: bool) {
        if self.closed {
            return;
        }
        self.closed = true;
        // SAFETY: plain FFI call with a value argument.
        unsafe {
            if finish {
                stt_stream_finish(self.id);
            } else {
                stt_stream_cancel(self.id);
            }
        }
    }
}

impl RecognizerSession for IosSession {
    fn feed(&mut self, samples: &[Sample]) {
        if self.closed || samples.is_empty() {
            return;
        }
        // SAFETY: the slice is valid for the duration of the call; Swift
        // copies it into an AVAudioPCMBuffer before returning.
        unsafe { stt_stream_feed(self.id, samples.as_ptr(), samples.len() as u32) };
    }

    fn finish(mut self: Box<Self>) {
        self.close(true);
    }

    fn cancel(mut self: Box<Self>) {
        self.close(false);
    }
}

impl Drop for IosSession {
    fn drop(&mut self) {
        // Dropping an open session aborts it; if Swift never reports `Ended`
        // (it always does after cancel), the callback entry is removed here.
        if !self.closed {
            self.close(false);
        }
    }
}

/// Widen a bare language code to the region variant Apple's recognizer
/// expects ("de" → "de-DE"). Tags with a region pass through.
fn widen_language(code: &str) -> String {
    if code.contains('-') || code.contains('_') {
        return code.to_string();
    }
    let region = match code.to_ascii_lowercase().as_str() {
        "de" => "de-DE",
        "en" => "en-US",
        "fr" => "fr-FR",
        "es" => "es-ES",
        "it" => "it-IT",
        "pt" => "pt-BR",
        "nl" => "nl-NL",
        "pl" => "pl-PL",
        "ru" => "ru-RU",
        "ja" => "ja-JP",
        "ko" => "ko-KR",
        "zh" => "zh-CN",
        "ar" => "ar-SA",
        "tr" => "tr-TR",
        _ => return format!("{}-{}", code, code.to_ascii_uppercase()),
    };
    region.to_string()
}

impl<R: Runtime> Recognizer for Stt<R> {
    fn open(
        &self,
        config: RecognizerConfig,
        on_event: RecognizerCallback,
    ) -> crate::Result<Box<dyn RecognizerSession>> {
        if !self.request_authorization()? {
            return Err(crate::Error::PermissionDenied(
                "Speech recognition not authorized".to_string(),
            ));
        }
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let locale = CString::new(widen_language(&config.language))
            .map_err(|_| crate::Error::ConfigError("Invalid language tag".to_string()))?;
        sessions()
            .lock()
            .map_err(|_| crate::Error::Recording("Session registry poisoned".to_string()))?
            .insert(id, Arc::new(on_event));
        // SAFETY: `locale` outlives the call; Swift copies it.
        let code = unsafe {
            stt_stream_start(
                id,
                locale.as_ptr(),
                config.sample_rate as f64,
                config.interim_results as u32,
                config.on_device as u32,
            )
        };
        if code != 0 {
            if let Ok(mut s) = sessions().lock() {
                s.remove(&id);
            }
            return Err(crate::Error::NotAvailable(describe_start_error(code)));
        }
        Ok(Box::new(IosSession { id, closed: false }))
    }

    fn status(&self, _language: &str) -> RecognizerStatus {
        // SAFETY: plain FFI query.
        let auth = unsafe { stt_stream_authorization_status() };
        let denied = auth != AUTH_AUTHORIZED && auth != AUTH_UNDETERMINED;
        RecognizerStatus {
            available: !denied,
            reason: denied.then(|| "Speech recognition permission denied".to_string()),
            model_installed: true,
            needs_model_download: false,
        }
    }

    fn install_model(&self, _language: &str) -> crate::Result<()> {
        Ok(())
    }

    fn request_authorization(&self) -> crate::Result<bool> {
        // SAFETY: plain FFI call; blocks until the user answers the prompt.
        let status = unsafe { stt_stream_request_authorization() };
        Ok(status == AUTH_AUTHORIZED)
    }
}
