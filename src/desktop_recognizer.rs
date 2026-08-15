//! Desktop [`Recognizer`]: an in-process Vosk recognizer over the plugin's
//! model store. Shares model download/loading with the command layer
//! (`Stt::ensure_model`), so a model fetched for either path serves both.

use std::sync::Arc;

use tauri::Runtime;
use vosk::{DecodingState, Model};

use crate::desktop::Stt;
use crate::recognizer::{
    Recognizer, RecognizerCallback, RecognizerConfig, RecognizerEvent, RecognizerSession,
    RecognizerStatus, Sample,
};

struct VoskSession {
    recognizer: vosk::Recognizer,
    // Held so the recognizer's model outlives it even if the plugin swaps
    // models under a concurrent `ensure_model`.
    _model: Arc<Model>,
    on_event: RecognizerCallback,
    interim_results: bool,
    last_partial: String,
    ended: bool,
}

impl VoskSession {
    fn text_of(result: vosk::CompleteResult) -> String {
        match result {
            vosk::CompleteResult::Single(single) => single.text.to_string(),
            vosk::CompleteResult::Multiple(multiple) => multiple
                .alternatives
                .first()
                .map(|a| a.text.to_string())
                .unwrap_or_default(),
        }
    }

    fn end(&mut self) {
        if !self.ended {
            self.ended = true;
            (self.on_event)(RecognizerEvent::Ended);
        }
    }
}

impl RecognizerSession for VoskSession {
    fn feed(&mut self, samples: &[Sample]) {
        if self.ended || samples.is_empty() {
            return;
        }
        match self.recognizer.accept_waveform(samples) {
            Ok(DecodingState::Finalized) => {
                let text = Self::text_of(self.recognizer.result());
                self.last_partial.clear();
                if !text.is_empty() {
                    (self.on_event)(RecognizerEvent::Final(text));
                }
            }
            Ok(DecodingState::Running) if self.interim_results => {
                let partial = self.recognizer.partial_result().partial.to_string();
                if !partial.is_empty() && partial != self.last_partial {
                    self.last_partial = partial.clone();
                    (self.on_event)(RecognizerEvent::Partial(partial));
                }
            }
            Ok(DecodingState::Running) => {}
            Ok(DecodingState::Failed) => {
                (self.on_event)(RecognizerEvent::Error("Vosk decoding failed".to_string()));
                self.end();
            }
            Err(e) => {
                (self.on_event)(RecognizerEvent::Error(format!("Vosk rejected audio: {e:?}")));
                self.end();
            }
        }
    }

    fn finish(mut self: Box<Self>) {
        if self.ended {
            return;
        }
        let text = Self::text_of(self.recognizer.final_result());
        if !text.is_empty() {
            (self.on_event)(RecognizerEvent::Final(text));
        }
        self.end();
    }

    fn cancel(mut self: Box<Self>) {
        self.end();
    }
}

impl Drop for VoskSession {
    fn drop(&mut self) {
        self.end();
    }
}

impl<R: Runtime> Recognizer for Stt<R> {
    fn open(
        &self,
        config: RecognizerConfig,
        on_event: RecognizerCallback,
    ) -> crate::Result<Box<dyn RecognizerSession>> {
        let model = self.ensure_model(Some(&config.language))?;
        let mut recognizer = vosk::Recognizer::new(&model, config.sample_rate as f32)
            .ok_or_else(|| {
                crate::Error::Recording("Failed to create Vosk recognizer".to_string())
            })?;
        recognizer.set_max_alternatives(0);
        recognizer.set_partial_words(false);
        Ok(Box::new(VoskSession {
            recognizer,
            _model: model,
            on_event,
            interim_results: config.interim_results,
            last_partial: String::new(),
            ended: false,
        }))
    }

    fn status(&self, language: &str) -> RecognizerStatus {
        let model_installed = self
            .get_model_info_for_language(language)
            .map(|(name, _)| self.get_models_dir().join(name).exists())
            .unwrap_or(false);
        RecognizerStatus {
            available: true,
            reason: None,
            model_installed,
            needs_model_download: true,
        }
    }

    fn install_model(&self, language: &str) -> crate::Result<()> {
        self.ensure_model(Some(language)).map(|_| ())
    }

    fn request_authorization(&self) -> crate::Result<bool> {
        Ok(true)
    }
}
