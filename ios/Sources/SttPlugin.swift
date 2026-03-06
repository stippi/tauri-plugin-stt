import AVFoundation
import Speech
import Tauri
import UIKit
import WebKit

/// Configuration for speech recognition
struct ListenConfig: Decodable {
    let language: String?
    let interimResults: Bool?
    let continuous: Bool?
    let maxDuration: Int?
    let onDevice: Bool?

    static var `default`: ListenConfig {
        return ListenConfig(language: nil, interimResults: true, continuous: false, maxDuration: nil, onDevice: nil)
    }

    init(language: String? = nil, interimResults: Bool? = true, continuous: Bool? = false, maxDuration: Int? = nil, onDevice: Bool? = nil) {
        self.language = language
        self.interimResults = interimResults
        self.continuous = continuous
        self.maxDuration = maxDuration
        self.onDevice = onDevice
    }
}

/// Tauri plugin for Speech-to-Text recognition on iOS
/// Uses Apple's Speech framework (SFSpeechRecognizer)
///
/// Thread safety: All mutable state is accessed exclusively on `pluginQueue`
/// (a serial queue). Delegate callbacks from SFSpeechRecognizer may fire on
/// arbitrary threads, so they dispatch to `pluginQueue` before touching state.
/// `trigger()` calls are dispatched to the main thread since they go through
/// the WKWebView bridge which requires main-thread access.
class SttPlugin: Plugin, SFSpeechRecognitionTaskDelegate {
    // Serial queue for all state access — avoids races between delegate callbacks,
    // stopRecognition, and Tauri commands.
    private let pluginQueue = DispatchQueue(label: "com.yellowbites.stt-plugin", qos: .userInteractive)

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?
    private var isListening = false
    private var currentLanguage: String?
    private var currentConfig: ListenConfig?
    private var isManualStop = false
    private var isStopping = false      // reentrance guard for stopRecognition
    private var tapInstalled = false     // tracks whether audio tap is installed
    private var wasListeningBeforeInterruption = false
    private var maxDurationTimer: DispatchWorkItem?

    override init() {
        super.init()
        audioEngine = AVAudioEngine()
        setupInterruptionHandling()
        NSLog("[SttPlugin] Initialized (iOS %@)", UIDevice.current.systemVersion)
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        maxDurationTimer?.cancel()
    }

    // MARK: - Thread-safe event emission

    /// Send an event to JS on the main thread.
    /// `Plugin.trigger()` goes through the WKWebView bridge which requires main-thread access.
    /// Dispatching asynchronously avoids blocking the calling thread (which may be the
    /// Speech framework's internal thread or our pluginQueue).
    private func emitEvent(_ eventName: String, data: JSObject) {
        DispatchQueue.main.async { [weak self] in
            self?.trigger(eventName, data: data)
        }
    }

    /// Emit a debug event to the frontend for logging.
    /// These show up in the app's log file (unlike NSLog which only goes to system log).
    private func emitDebug(_ message: String) {
        emitEvent("debug", data: [
            "source": "ios-stt",
            "message": message
        ] as JSObject)
    }

    // MARK: - Audio Session Interruption Handling

    /// Setup observers for audio session interruptions (phone calls, Siri, etc.)
    private func setupInterruptionHandling() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioSessionInterruption),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioRouteChange),
            name: AVAudioSession.routeChangeNotification,
            object: AVAudioSession.sharedInstance()
        )
    }

    /// Handles audio route changes (headphones plugged/unplugged, Bluetooth, etc.)
    @objc private func handleAudioRouteChange(notification: Notification) {
        pluginQueue.async { [weak self] in
            guard let self = self else { return }

            guard let userInfo = notification.userInfo,
                  let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
                  let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
                return
            }

            switch reason {
            case .oldDeviceUnavailable:
                if self.isListening {
                    NSLog("[SttPlugin] Recognition stopped due to audio route change (device unavailable)")
                    self.stopRecognitionInternal()
                    self.emitEvent("stateChange", data: ["state": "idle"] as JSObject)
                }
            case .newDeviceAvailable:
                NSLog("[SttPlugin] New audio device available")
            default:
                break
            }
        }
    }

    /// Handle audio interruptions such as phone calls
    @objc private func handleAudioSessionInterruption(notification: Notification) {
        pluginQueue.async { [weak self] in
            guard let self = self else { return }

            guard let userInfo = notification.userInfo,
                  let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
                return
            }

            switch type {
            case .began:
                if self.isListening {
                    self.wasListeningBeforeInterruption = true
                    NSLog("[SttPlugin] Recognition interrupted, stopping...")
                    self.stopRecognitionInternal()

                    self.emitEvent("error", data: [
                        "code": "CANCELLED",
                        "message": "Recognition interrupted by system",
                        "details": "iOS audio session interruption"
                    ] as JSObject)
                    self.emitEvent("stateChange", data: ["state": "idle"] as JSObject)
                }

            case .ended:
                guard let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt else { return }
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)

                if options.contains(.shouldResume) && self.wasListeningBeforeInterruption {
                    NSLog("[SttPlugin] Interruption ended, user can restart recognition")
                }
                self.wasListeningBeforeInterruption = false

            @unknown default:
                break
            }
        }
    }

    // MARK: - SFSpeechRecognitionTaskDelegate
    //
    // These delegate methods may fire on ANY thread (Speech framework internal).
    // We dispatch to pluginQueue for thread-safe state access and use emitEvent()
    // (which dispatches to main thread) for trigger() calls.

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didHypothesizeTranscription transcription: SFTranscription) {
        let transcript = transcription.formattedString
        let confidence: Float? = transcription.segments.last?.confidence

        var eventData: JSObject = [
            "transcript": transcript,
            "isFinal": false
        ]
        if let conf = confidence {
            eventData["confidence"] = conf
        }
        emitEvent("result", data: eventData)
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didFinishRecognition recognitionResult: SFSpeechRecognitionResult) {
        let transcript = recognitionResult.bestTranscription.formattedString
        let confidence: Float? = recognitionResult.bestTranscription.segments.last?.confidence
        emitDebug("didFinishRecognition: \(transcript)")

        var eventData: JSObject = [
            "transcript": transcript,
            "isFinal": true
        ]
        if let conf = confidence {
            eventData["confidence"] = conf
        }
        emitEvent("result", data: eventData)
        emitEvent("stateChange", data: ["state": "idle"] as JSObject)
    }

    /// Called when task is cancelled. Do NOT call stopRecognition() here —
    /// cancel() is called FROM stopRecognition(), so this would recurse.
    func speechRecognitionTaskWasCancelled(_ task: SFSpeechRecognitionTask) {
        emitDebug("speechRecognitionTaskWasCancelled")
        emitEvent("stateChange", data: ["state": "idle"] as JSObject)
    }

    /// Called when recognition finishes (successfully or not).
    /// May fire synchronously during cancel() or asynchronously after.
    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didFinishSuccessfully successfully: Bool) {
        emitDebug("didFinishSuccessfully: \(successfully)")

        pluginQueue.async { [weak self] in
            guard let self = self else { return }

            // Don't report error if user manually stopped recognition
            if !successfully && !self.isManualStop {
                self.emitEvent("error", data: [
                    "code": "UNKNOWN",
                    "message": "Recognition finished unsuccessfully",
                    "details": "Speech recognition task failed"
                ] as JSObject)
            }

            // If stopRecognition() already ran (manual stop or reentrant),
            // don't call it again — resources are already cleaned up.
            if self.isManualStop || self.isStopping {
                return
            }

            // Natural finish (e.g. silence timeout in non-continuous mode).
            // Capture config BEFORE stopRecognition clears it.
            let config = self.currentConfig

            self.stopRecognitionInternal()

            // Restart if continuous mode
            if let config = config, config.continuous ?? false, successfully {
                self.pluginQueue.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                    guard let self = self, !self.isManualStop else {
                        return
                    }
                    // startRecognition touches audio APIs → must run on main thread
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        do {
                            try self.startRecognition(config: config)
                        } catch {
                            NSLog("[SttPlugin] Failed to restart continuous recognition: \(error.localizedDescription)")
                            self.emitEvent("error", data: [
                                "code": "UNKNOWN",
                                "message": "Failed to restart recognition",
                                "details": error.localizedDescription
                            ] as JSObject)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Commands

    @objc public func startListening(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] startListening called")

        let args: ListenConfig
        do {
            args = try invoke.parseArgs(ListenConfig.self)
        } catch {
            NSLog("[SttPlugin] Failed to parse args, using defaults: \(error)")
            args = ListenConfig.default
        }

        if isListening {
            invoke.reject("Already listening")
            return
        }

        let speechStatus = SFSpeechRecognizer.authorizationStatus()
        let micStatus = AVAudioSession.sharedInstance().recordPermission

        if speechStatus == .authorized && micStatus == .granted {
            startListeningWithConfig(args, invoke: invoke)
            return
        }

        if speechStatus == .denied || speechStatus == .restricted {
            invoke.reject("Speech recognition permission denied. Please enable it in Settings.")
            return
        }

        if micStatus == .denied {
            invoke.reject("Microphone permission denied. Please enable it in Settings.")
            return
        }

        NSLog("[SttPlugin] Requesting permissions...")
        let group = DispatchGroup()
        var permissionError: String? = nil

        if speechStatus == .notDetermined {
            group.enter()
            SFSpeechRecognizer.requestAuthorization { status in
                if status != .authorized {
                    permissionError = "Speech recognition permission not granted"
                }
                group.leave()
            }
        }

        if micStatus == .undetermined {
            group.enter()
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                if !granted {
                    permissionError = "Microphone permission not granted"
                }
                group.leave()
            }
        }

        group.notify(queue: .main) { [weak self] in
            if let error = permissionError {
                invoke.reject(error)
                return
            }
            self?.startListeningWithConfig(args, invoke: invoke)
        }
    }

    private func startListeningWithConfig(_ args: ListenConfig, invoke: Invoke) {
        let locale: Locale
        if let language = args.language {
            locale = Locale(identifier: language)
            NSLog("[SttPlugin] Using specified locale: \(language)")
        } else {
            locale = Locale.current
            NSLog("[SttPlugin] Using current locale: \(locale.identifier)")
        }

        speechRecognizer = SFSpeechRecognizer(locale: locale)
        currentLanguage = locale.identifier

        guard let speechRecognizer = speechRecognizer else {
            NSLog("[SttPlugin] SFSpeechRecognizer is nil for locale: \(locale.identifier)")
            invoke.reject("Speech recognition not available for language: \(locale.identifier)")
            return
        }

        guard speechRecognizer.isAvailable else {
            NSLog("[SttPlugin] SFSpeechRecognizer not available for locale: \(locale.identifier)")
            invoke.reject("Speech recognition not available for language: \(locale.identifier)")
            return
        }

        do {
            try startRecognition(config: args)
            NSLog("[SttPlugin] Recognition started for locale: \(locale.identifier)")
            invoke.resolve()
        } catch {
            NSLog("[SttPlugin] Failed to start recognition: \(error)")
            invoke.reject("Failed to start recognition: \(error.localizedDescription)")
        }
    }

    @objc public func stopListening(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] stopListening called")
        isManualStop = true
        pluginQueue.async { [weak self] in
            self?.stopRecognitionInternal()
        }
        invoke.resolve()
    }

    @objc public func isAvailable(_ invoke: Invoke) throws {
        let recognizer = SFSpeechRecognizer()
        let available = recognizer?.isAvailable ?? false

        var result: JSObject = ["available": available]
        if !available {
            result["reason"] = "Speech recognition not available on this device"
        }

        invoke.resolve(result)
    }

    @objc public func getSupportedLanguages(_ invoke: Invoke) throws {
        let supportedLocales = SFSpeechRecognizer.supportedLocales()

        let languages = supportedLocales.map { locale -> [String: String] in
            return [
                "code": locale.identifier,
                "name": locale.localizedString(forIdentifier: locale.identifier) ?? locale.identifier
            ]
        }

        invoke.resolve(["languages": languages])
    }

    @objc public func checkPermission(_ invoke: Invoke) throws {
        let micStatus: String
        let micRaw = AVAudioSession.sharedInstance().recordPermission
        switch micRaw {
        case .granted:
            micStatus = "granted"
        case .denied:
            micStatus = "denied"
        case .undetermined:
            micStatus = "unknown"
        @unknown default:
            micStatus = "unknown"
        }

        let speechStatus: String
        let speechRaw = SFSpeechRecognizer.authorizationStatus()
        switch speechRaw {
        case .authorized:
            speechStatus = "granted"
        case .denied, .restricted:
            speechStatus = "denied"
        case .notDetermined:
            speechStatus = "unknown"
        @unknown default:
            speechStatus = "unknown"
        }

        invoke.resolve([
            "microphone": micStatus,
            "speechRecognition": speechStatus
        ])
    }

    @objc public func requestPermission(_ invoke: Invoke) throws {
        let group = DispatchGroup()

        var micResult = "unknown"
        var speechResult = "unknown"

        group.enter()
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            micResult = granted ? "granted" : "denied"
            group.leave()
        }

        group.enter()
        SFSpeechRecognizer.requestAuthorization { status in
            switch status {
            case .authorized:
                speechResult = "granted"
            case .denied, .restricted:
                speechResult = "denied"
            case .notDetermined:
                speechResult = "unknown"
            @unknown default:
                speechResult = "unknown"
            }
            group.leave()
        }

        group.notify(queue: .main) {
            NSLog("[SttPlugin]   Final results - mic: \(micResult), speech: \(speechResult)")
            invoke.resolve([
                "microphone": micResult,
                "speechRecognition": speechResult
            ])
        }
    }

    // MARK: - Private Methods

    private func startRecognition(config: ListenConfig) throws {
        // Reset all flags when starting new recognition
        isManualStop = false
        isStopping = false

        recognitionTask?.cancel()
        recognitionTask = nil

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothA2DP])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            NSLog("[SttPlugin] Audio session configuration failed: \(error)")
            throw error
        }

        if audioEngine == nil {
            NSLog("[SttPlugin] Creating new audio engine")
            audioEngine = AVAudioEngine()
        }

        guard let audioEngine = audioEngine else {
            NSLog("[SttPlugin] Audio engine is nil")
            throw NSError(domain: "SttPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Audio engine not initialized"])
        }
        
        NSLog("[SttPlugin] Creating recognition request...")
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()

        guard let recognitionRequest = recognitionRequest else {
            NSLog("[SttPlugin] Recognition request is nil")
            throw NSError(domain: "SttPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create recognition request"])
        }

        recognitionRequest.shouldReportPartialResults = config.interimResults ?? true

        if #available(iOS 13, *) {
            let useOnDevice = config.onDevice ?? false
            if useOnDevice {
                // Check if on-device recognition is supported for this locale
                if speechRecognizer?.supportsOnDeviceRecognition == true {
                    recognitionRequest.requiresOnDeviceRecognition = true
                    NSLog("[SttPlugin] Using on-device recognition (offline)")
                } else {
                    NSLog("[SttPlugin] On-device recognition not available for this language, using server")
                    recognitionRequest.requiresOnDeviceRecognition = false
                }
            } else {
                recognitionRequest.requiresOnDeviceRecognition = false
            }
        }

        let inputNode = audioEngine.inputNode
        NSLog("[SttPlugin] Input node obtained")
        
        guard let recognizer = speechRecognizer else {
            NSLog("[SttPlugin] Speech recognizer is nil")
            throw NSError(domain: "SttPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Speech recognizer not available"])
        }

        currentConfig = config
        
        NSLog("[SttPlugin] Starting recognition task...")
        recognitionTask = recognizer.recognitionTask(with: recognitionRequest, delegate: self)

        let recordingFormat = inputNode.outputFormat(forBus: 0)

        // Safely remove existing tap before installing new one
        if tapInstalled {
            inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        tapInstalled = true

        audioEngine.prepare()
        NSLog("[SttPlugin] Audio engine prepared")
        
        try audioEngine.start()
        NSLog("[SttPlugin] Audio engine started")
        
        isListening = true
        emitEvent("stateChange", data: ["state": "listening"] as JSObject)

        // Setup maxDuration timer if configured
        if let maxDuration = config.maxDuration, maxDuration > 0 {
            maxDurationTimer?.cancel()

            let workItem = DispatchWorkItem { [weak self] in
                guard let self = self, self.isListening else { return }
                NSLog("[SttPlugin] maxDuration reached, stopping")
                self.stopRecognitionInternal()
                self.emitEvent("stateChange", data: ["state": "idle"] as JSObject)
                self.emitEvent("error", data: [
                    "code": "TIMEOUT",
                    "message": "Maximum duration reached",
                    "details": "Recognition stopped after maxDuration limit"
                ] as JSObject)
            }
            maxDurationTimer = workItem
            pluginQueue.asyncAfter(deadline: .now() + .milliseconds(maxDuration), execute: workItem)
        }
    }

    /// Tear down audio engine, recognition request, and task.
    /// MUST be called on pluginQueue (or from main thread during startListening flow).
    ///
    /// Key safety measures:
    /// - `isStopping` reentrance guard: `recognitionTask?.cancel()` can synchronously
    ///   fire delegate callbacks which would call stopRecognition() again.
    /// - `isStopping` stays true until next `startRecognition()` so async delegate
    ///   callbacks that fire later are also guarded.
    /// - `tapInstalled` flag: `removeTap(onBus:)` throws an uncatchable ObjC
    ///   NSInternalInconsistencyException if no tap exists.
    private func stopRecognitionInternal() {
        if isStopping {
            return
        }
        isStopping = true

        NSLog("[SttPlugin] stopRecognitionInternal (isManualStop=\(isManualStop))")

        maxDurationTimer?.cancel()
        maxDurationTimer = nil

        if let engine = audioEngine {
            engine.stop()
            if tapInstalled {
                engine.inputNode.removeTap(onBus: 0)
                tapInstalled = false
            }
        }

        recognitionRequest?.endAudio()
        recognitionRequest = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        isListening = false
        currentConfig = nil

        // Note: isManualStop and isStopping are NOT reset here.
        // Delegate callbacks may fire asynchronously after cancel().
        // Both flags are reset at the start of next startRecognition().

        DispatchQueue.global(qos: .utility).async {
            do {
                try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            } catch {
                NSLog("[SttPlugin] Failed to deactivate audio session: \(error.localizedDescription)")
            }
        }
    }
}

@_cdecl("init_plugin_stt")
func initPlugin() -> Plugin {
    return SttPlugin()
}
