use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

pub use models::*;

#[cfg(desktop)]
mod desktop;
#[cfg(mobile)]
mod mobile;

mod commands;
#[cfg(desktop)]
mod desktop_recognizer;
mod error;
#[cfg(target_os = "ios")]
mod ios_stream;
mod models;
mod paths;
pub mod recognizer;

pub use error::{Error, Result};
pub use recognizer::{
    Recognizer, RecognizerCallback, RecognizerConfig, RecognizerEvent, RecognizerSession,
    RecognizerStatus,
};
pub use paths::{
    get_model_path, get_models_dir, list_available_models, model_exists, validate_path,
};

#[cfg(desktop)]
use desktop::Stt;
#[cfg(mobile)]
use mobile::Stt;

/// Shared handle to the platform's buffer-fed [`Recognizer`].
#[derive(Clone)]
pub struct SharedRecognizer(std::sync::Arc<dyn Recognizer>);

impl std::ops::Deref for SharedRecognizer {
    type Target = dyn Recognizer;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl SharedRecognizer {
    pub fn into_arc(self) -> std::sync::Arc<dyn Recognizer> {
        self.0
    }
}

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access the stt APIs.
pub trait SttExt<R: Runtime> {
    fn stt(&self) -> &Stt<R>;
    /// The buffer-fed recognizer for hosts that own the microphone themselves.
    fn recognizer(&self) -> SharedRecognizer;
}

impl<R: Runtime, T: Manager<R>> crate::SttExt<R> for T {
    fn stt(&self) -> &Stt<R> {
        self.state::<Stt<R>>().inner()
    }
    fn recognizer(&self) -> SharedRecognizer {
        self.state::<SharedRecognizer>().inner().clone()
    }
}

/// Platforms without a buffer-fed binding (Android) still register a
/// recognizer so hosts can branch on `status().available`.
#[cfg(target_os = "android")]
struct UnsupportedRecognizer;

#[cfg(target_os = "android")]
impl Recognizer for UnsupportedRecognizer {
    fn open(
        &self,
        _config: RecognizerConfig,
        _on_event: RecognizerCallback,
    ) -> Result<Box<dyn RecognizerSession>> {
        Err(Error::NotAvailable(
            "Buffer-fed recognition is not implemented on this platform".to_string(),
        ))
    }
    fn status(&self, _language: &str) -> RecognizerStatus {
        RecognizerStatus {
            available: false,
            reason: Some("Buffer-fed recognition is not implemented on this platform".to_string()),
            model_installed: false,
            needs_model_download: false,
        }
    }
    fn install_model(&self, _language: &str) -> Result<()> {
        Ok(())
    }
    fn request_authorization(&self) -> Result<bool> {
        Ok(false)
    }
}

/// Initializes the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    let mut builder = Builder::new("stt");

    #[cfg(desktop)]
    {
        builder = builder.invoke_handler(tauri::generate_handler![
            commands::start_listening,
            commands::stop_listening,
            commands::is_available,
            commands::get_supported_languages,
            commands::check_permission,
            commands::request_permission,
            commands::register_listener,
            commands::remove_listener,
        ]);
    }

    #[cfg(mobile)]
    {
        builder = builder.invoke_handler(tauri::generate_handler![
            commands::start_listening,
            commands::stop_listening,
            commands::is_available,
            commands::get_supported_languages,
            commands::check_permission,
            commands::request_permission,
        ]);
    }

    builder
        .setup(|app, api| {
            #[cfg(mobile)]
            let stt = mobile::init(app, api)?;
            #[cfg(desktop)]
            let stt = desktop::init(app, api)?;
            #[cfg(target_os = "android")]
            let recognizer = SharedRecognizer(std::sync::Arc::new(UnsupportedRecognizer));
            #[cfg(not(target_os = "android"))]
            let recognizer = SharedRecognizer(std::sync::Arc::new(stt.clone()));
            app.manage(recognizer);
            app.manage(stt);
            Ok(())
        })
        .build()
}
