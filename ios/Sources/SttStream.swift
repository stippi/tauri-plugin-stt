import AVFoundation
import Foundation
import Speech

// MARK: - Buffer-fed recognition over a C ABI
//
// The host app owns the microphone (a Rust audio pipeline) and pushes PCM
// into an `SFSpeechAudioBufferRecognitionRequest` through the `stt_stream_*`
// functions below. Results travel back through the `rust_stt_stream_*`
// callbacks, which the plugin's Rust side exports. No WKWebView, no Tauri
// invoke bridge, no second AVAudioEngine — recognition becomes a pure
// function of the audio the app already captures.
//
// Threading: `stt_stream_*` may be called from any thread. Each session
// serializes its own state on `queue`; Speech-framework delegate callbacks
// arrive on arbitrary threads and hop onto that queue too. Rust callbacks
// are invoked from `queue` — they must not block (they don't: they push
// into a channel).

@_silgen_name("rust_stt_stream_on_partial")
func rust_stt_stream_on_partial(_ id: UInt64, _ text: UnsafePointer<CChar>)

@_silgen_name("rust_stt_stream_on_final")
func rust_stt_stream_on_final(_ id: UInt64, _ text: UnsafePointer<CChar>)

@_silgen_name("rust_stt_stream_on_error")
func rust_stt_stream_on_error(_ id: UInt64, _ message: UnsafePointer<CChar>)

@_silgen_name("rust_stt_stream_on_ended")
func rust_stt_stream_on_ended(_ id: UInt64)

/// Error codes returned by `stt_stream_start`. Mirrored in Rust.
private enum StreamStartError: Int32 {
    case ok = 0
    case unsupportedLocale = 1
    case recognizerUnavailable = 2
    case notAuthorized = 3
    case badFormat = 4
    case alreadyRunning = 5
}

/// One recognition session fed from Rust.
private final class SpeechStream: NSObject, SFSpeechRecognitionTaskDelegate {
    let id: UInt64
    private let queue: DispatchQueue
    private let recognizer: SFSpeechRecognizer
    private let request: SFSpeechAudioBufferRecognitionRequest
    private let format: AVAudioFormat
    private var task: SFSpeechRecognitionTask?
    private var ended = false
    private var finishing = false
    private var lastHypothesis: String = ""

    init?(id: UInt64, locale: Locale, sampleRate: Double, interimResults: Bool, onDevice: Bool) {
        guard let recognizer = SFSpeechRecognizer(locale: locale) else { return nil }
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }
        self.id = id
        self.queue = DispatchQueue(label: "com.yellowbites.stt-stream.\(id)", qos: .userInteractive)
        self.recognizer = recognizer
        self.format = format
        self.request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = interimResults
        // PTT turns are single utterances; let the recognizer punctuate.
        request.taskHint = .dictation
        if #available(iOS 16, *) {
            request.addsPunctuation = true
        }
        if onDevice, recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        super.init()
    }

    var isAvailable: Bool { recognizer.isAvailable }

    func start() {
        queue.async {
            self.task = self.recognizer.recognitionTask(with: self.request, delegate: self)
        }
    }

    /// Append PCM (mono Int16 at the session's sample rate).
    func feed(_ samples: UnsafePointer<Int16>, count: UInt32) {
        guard count > 0 else { return }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(count)
        if let channel = buffer.int16ChannelData?[0] {
            channel.update(from: samples, count: Int(count))
        }
        // `append` is thread-safe; no need to hop queues for the hot path.
        request.append(buffer)
    }

    /// No more audio: let the recognizer produce its final transcript.
    func finish() {
        queue.async {
            guard !self.ended, !self.finishing else { return }
            self.finishing = true
            self.request.endAudio()
            // Some recognizers never deliver `didFinishRecognition` after
            // endAudio (short/empty audio, server hiccup). Bound the wait and
            // fall back to the last hypothesis so the host is never stuck.
            self.queue.asyncAfter(deadline: .now() + 3.0) { [weak self] in
                guard let self = self, !self.ended else { return }
                NSLog("[SttStream %llu] finalize timeout — using last hypothesis", self.id)
                self.emitFinalAndEnd(self.lastHypothesis)
                self.task?.cancel()
            }
        }
    }

    func cancel() {
        queue.async {
            guard !self.ended else { return }
            self.task?.cancel()
            self.end()
        }
    }

    // MARK: SFSpeechRecognitionTaskDelegate (arbitrary threads)

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didHypothesizeTranscription transcription: SFTranscription) {
        let text = transcription.formattedString
        queue.async {
            guard !self.ended else { return }
            self.lastHypothesis = text
            text.withCString { rust_stt_stream_on_partial(self.id, $0) }
        }
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didFinishRecognition recognitionResult: SFSpeechRecognitionResult) {
        let text = recognitionResult.bestTranscription.formattedString
        queue.async {
            self.emitFinalAndEnd(text)
        }
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didFinishSuccessfully successfully: Bool) {
        queue.async {
            guard !self.ended else { return }
            if successfully {
                // Finished without a `didFinishRecognition` (nothing recognized).
                self.emitFinalAndEnd(self.lastHypothesis)
            } else if self.finishing {
                // Ending an empty/short request is reported as a failure by
                // the framework (kAFAssistantErrorDomain 1110 "no speech");
                // treat it as "nothing recognized", not as an error.
                self.emitFinalAndEnd(self.lastHypothesis)
            } else {
                let message = task.error?.localizedDescription ?? "Recognition failed"
                message.withCString { rust_stt_stream_on_error(self.id, $0) }
                self.end()
            }
        }
    }

    func speechRecognitionTaskWasCancelled(_ task: SFSpeechRecognitionTask) {
        queue.async { self.end() }
    }

    // MARK: Helpers (on queue)

    private func emitFinalAndEnd(_ text: String) {
        guard !ended else { return }
        if !text.isEmpty {
            text.withCString { rust_stt_stream_on_final(self.id, $0) }
        }
        end()
    }

    private func end() {
        guard !ended else { return }
        ended = true
        rust_stt_stream_on_ended(id)
        SttStreamRegistry.shared.remove(id)
    }
}

/// Live sessions by id. Ids are minted by Rust; a stale id is a no-op.
private final class SttStreamRegistry {
    static let shared = SttStreamRegistry()
    private let lock = NSLock()
    private var streams: [UInt64: SpeechStream] = [:]

    func insert(_ stream: SpeechStream) {
        lock.lock(); defer { lock.unlock() }
        streams[stream.id] = stream
    }

    func get(_ id: UInt64) -> SpeechStream? {
        lock.lock(); defer { lock.unlock() }
        return streams[id]
    }

    func remove(_ id: UInt64) {
        lock.lock(); defer { lock.unlock() }
        streams.removeValue(forKey: id)
    }

    var isEmpty: Bool {
        lock.lock(); defer { lock.unlock() }
        return streams.isEmpty
    }
}

// MARK: - C ABI (called from Rust)

/// Speech-recognition authorization: 0 undetermined, 1 authorized, 2 denied,
/// 3 restricted.
@_cdecl("stt_stream_authorization_status")
public func stt_stream_authorization_status() -> Int32 {
    switch SFSpeechRecognizer.authorizationStatus() {
    case .notDetermined: return 0
    case .authorized: return 1
    case .denied: return 2
    case .restricted: return 3
    @unknown default: return 2
    }
}

/// Prompt for speech-recognition authorization if undetermined; blocks the
/// calling thread (never the main thread!) until answered. Returns the
/// resulting status (see `stt_stream_authorization_status`).
@_cdecl("stt_stream_request_authorization")
public func stt_stream_request_authorization() -> Int32 {
    if SFSpeechRecognizer.authorizationStatus() != .notDetermined {
        return stt_stream_authorization_status()
    }
    let semaphore = DispatchSemaphore(value: 0)
    SFSpeechRecognizer.requestAuthorization { _ in semaphore.signal() }
    semaphore.wait()
    return stt_stream_authorization_status()
}

/// Start a session. Returns 0 on success or a `StreamStartError` code.
@_cdecl("stt_stream_start")
public func stt_stream_start(
    _ id: UInt64,
    _ locale: UnsafePointer<CChar>,
    _ sampleRate: Double,
    _ interimResults: UInt32,
    _ onDevice: UInt32
) -> Int32 {
    guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
        return StreamStartError.notAuthorized.rawValue
    }
    guard SttStreamRegistry.shared.get(id) == nil else {
        return StreamStartError.alreadyRunning.rawValue
    }
    let localeId = String(cString: locale)
    guard let stream = SpeechStream(
        id: id,
        locale: Locale(identifier: localeId),
        sampleRate: sampleRate,
        interimResults: interimResults != 0,
        onDevice: onDevice != 0
    ) else {
        // SFSpeechRecognizer(locale:) is nil for unsupported locales; the
        // format initializer only fails for absurd sample rates.
        return SFSpeechRecognizer.supportedLocales().contains(Locale(identifier: localeId))
            ? StreamStartError.badFormat.rawValue
            : StreamStartError.unsupportedLocale.rawValue
    }
    guard stream.isAvailable else {
        return StreamStartError.recognizerUnavailable.rawValue
    }
    SttStreamRegistry.shared.insert(stream)
    stream.start()
    NSLog("[SttStream %llu] started (%@, %.0f Hz)", id, localeId, sampleRate)
    return StreamStartError.ok.rawValue
}

@_cdecl("stt_stream_feed")
public func stt_stream_feed(_ id: UInt64, _ samples: UnsafePointer<Int16>, _ count: UInt32) {
    SttStreamRegistry.shared.get(id)?.feed(samples, count: count)
}

@_cdecl("stt_stream_finish")
public func stt_stream_finish(_ id: UInt64) {
    SttStreamRegistry.shared.get(id)?.finish()
}

@_cdecl("stt_stream_cancel")
public func stt_stream_cancel(_ id: UInt64) {
    SttStreamRegistry.shared.get(id)?.cancel()
}
