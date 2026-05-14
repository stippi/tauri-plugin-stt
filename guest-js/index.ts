import {
  invoke,
  PluginListener,
  addPluginListener,
} from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

export interface ListenConfig {
  /** Language code for recognition (e.g., "en-US", "pt-BR") */
  language?: string;
  /** Whether to return interim (partial) results */
  interimResults?: boolean;
  /** Whether to continue listening after getting a result */
  continuous?: boolean;
  /** Maximum duration to listen in milliseconds (0 = no limit) */
  maxDuration?: number;
  /** Use on-device recognition only (iOS 13+, no network required)
   * When true, recognition works offline but may be less accurate.
   * Falls back to server if on-device not available for the language.
   */
  onDevice?: boolean;
}

export interface StopListeningConfig {
  /**
   * Additional microphone audio to accept after stop was requested.
   * Useful for push-to-talk so the final syllables are not clipped.
   */
  postRollMs?: number;
  /**
   * Maximum time mobile platform recognizers may wait for a final result.
   * Desktop Vosk finalizes synchronously.
   */
  finalizeTimeoutMs?: number;
}

export type RecognitionState = "idle" | "listening" | "processing";

export interface RecognitionResult {
  transcript: string;
  isFinal: boolean;
  confidence?: number;
}

/**
 * Unified error codes for cross-platform consistency
 */
export type SttErrorCode =
  | "NONE"
  | "NOT_AVAILABLE"
  | "PERMISSION_DENIED"
  | "SPEECH_PERMISSION_DENIED"
  | "NETWORK_ERROR"
  | "AUDIO_ERROR"
  | "TIMEOUT"
  | "NO_SPEECH"
  | "LANGUAGE_NOT_SUPPORTED"
  | "CANCELLED"
  | "ALREADY_LISTENING"
  | "NOT_LISTENING"
  | "BUSY"
  | "UNKNOWN";

/**
 * Structured error event with code and message
 */
export interface SttError {
  /** Error code for programmatic handling */
  code: SttErrorCode;
  /** Human-readable error message */
  message: string;
  /** Platform-specific error details */
  details?: string;
}

/**
 * @deprecated Use SttError instead
 */
export interface RecognitionError {
  error: string;
  code?: string;
}

export interface StateChangeEvent {
  state: RecognitionState;
}

export interface SupportedLanguage {
  code: string;
  name: string;
  installed?: boolean;
}

export interface AvailabilityResponse {
  available: boolean;
  reason?: string;
}

export interface SupportedLanguagesResponse {
  languages: SupportedLanguage[];
}

export type PermissionStatus = "granted" | "denied" | "unknown";

export interface PermissionResponse {
  microphone: PermissionStatus;
  speechRecognition: PermissionStatus;
}

export async function startListening(config?: ListenConfig): Promise<void> {
  await invoke("plugin:stt|start_listening", { config: config || {} });
}

export async function stopListening(
  config?: StopListeningConfig
): Promise<RecognitionResult | null> {
  return await invoke<RecognitionResult | null>("plugin:stt|stop_listening", {
    config: config || {},
  });
}

export async function isAvailable(): Promise<AvailabilityResponse> {
  return await invoke("plugin:stt|is_available");
}

export async function getSupportedLanguages(): Promise<SupportedLanguagesResponse> {
  return await invoke("plugin:stt|get_supported_languages");
}

export async function checkPermission(): Promise<PermissionResponse> {
  return await invoke("plugin:stt|check_permission");
}

export async function requestPermission(): Promise<PermissionResponse> {
  return await invoke("plugin:stt|request_permission");
}

/**
 * Listen for speech recognition results.
 * Uses channel-based communication on mobile, event system on desktop.
 */
export async function onResult(
  handler: (result: RecognitionResult) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<RecognitionResult>("stt", "result", handler);
  }

  const unlisten = await listen<RecognitionResult>(
    "plugin:stt:result",
    event => {
      handler(event.payload);
    }
  );
  return unlisten;
}

/**
 * Listen for state changes in the speech recognizer.
 */
export async function onStateChange(
  handler: (event: StateChangeEvent) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<StateChangeEvent>(
      "stt",
      "stateChange",
      handler
    );
  }

  const unlisten = await listen<StateChangeEvent>(
    "plugin:stt:stateChange",
    event => {
      handler(event.payload);
    }
  );
  return unlisten;
}

/**
 * Listen for speech recognition errors.
 */
export async function onError(
  handler: (error: SttError) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<SttError>("stt", "error", handler);
  }

  return await listen<SttError>("plugin:stt:error", event => {
    handler(event.payload);
  });
}

/**
 * Debug event from native platform implementations
 */
export interface DebugEvent {
  /** Source of the debug message (e.g. "ios", "android") */
  source: string;
  /** Debug message */
  message: string;
}

/**
 * Listen for debug messages from native STT implementations.
 * Useful for diagnosing issues on iOS/Android.
 */
export async function onDebug(
  handler: (event: DebugEvent) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<DebugEvent>("stt", "debug", handler);
  }

  return await listen<DebugEvent>("plugin:stt:debug", event => {
    handler(event.payload);
  });
}

/**
 * Detect if we're running on mobile platform.
 * Mobile uses channel-based listeners, desktop uses event system.
 *
 * iPadOS 13+ reports a macOS user agent by default ("Request Desktop Website"),
 * so we cannot rely on navigator.userAgent for iPad detection.
 * Instead we use Tauri's internal platform detection and iOS-specific APIs.
 */
function isMobilePlatform(): boolean {
  const w = window as any;

  // Check Tauri's internal platform detection first (most reliable)
  const platform = w.__TAURI_INTERNALS__?.plugins?.os?.platform;
  if (platform === "android" || platform === "ios") {
    return true;
  }

  // Check for Android WebView
  if (w.Android) {
    return true;
  }

  // Detect iOS/iPadOS via webkit.messageHandlers + touch support.
  // On iPadOS, navigator.userAgent reports macOS, so we can't rely on UA parsing.
  // Instead: webkit.messageHandlers exists on both macOS and iOS in WKWebView,
  // but we combine it with touch support and navigator.maxTouchPoints to
  // distinguish iPad from Mac.
  if (w.webkit?.messageHandlers) {
    // navigator.maxTouchPoints > 1 is true on iPad/iPhone but 0 on macOS
    if (navigator.maxTouchPoints > 1) {
      return true;
    }
    // Fallback: check user-agent for iPhone/iPod (these don't fake desktop UA)
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("iphone") || ua.includes("ipod")) {
      return true;
    }
  }

  // Default to desktop
  return false;
}
