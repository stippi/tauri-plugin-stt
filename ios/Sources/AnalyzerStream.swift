import AVFoundation
import CoreMedia
import Foundation
import Speech

// MARK: - SpeechAnalyzer-backed recognition (iOS 26+)
//
// Same contract as `SpeechStream` (see SttStream.swift): Rust pushes PCM via
// `feed`, results go back through the `rust_stt_stream_*` callbacks — one
// `Final` with the whole session's transcript on `finish`, then `Ended`.
//
// What differs from `SFSpeechRecognizer`:
// - Always on-device. The model is a system asset (`AssetInventory`), shared
//   between apps; there is no server fallback at all.
// - Results arrive per audio range: *volatile* results are guesses for the
//   range still being analyzed and get replaced; *final* results are
//   committed and never change. We fold them into one running transcript.
//
// Module choice: `SpeechTranscriber` (Apple's new general-purpose model) when
// the device supports it and covers the locale, else `DictationTranscriber`
// (the keyboard-dictation model, on-device through the same API). When
// neither is usable *right now* — unsupported locale, assets not installed,
// setup failure — `open` returns nil and the caller falls back to
// `SFSpeechRecognizer`. A missing asset starts downloading in the background,
// so a later session picks the analyzer up without any host involvement.

@available(iOS 26, macOS 26, *)
final class AnalyzerStream: RecognitionStream {
    let id: UInt64
    /// "SpeechTranscriber" or "DictationTranscriber", for logs.
    let moduleName: String

    private let queue: DispatchQueue
    private let analyzer: SpeechAnalyzer
    private let inputStream: AsyncStream<AnalyzerInput>
    private let inputContinuation: AsyncStream<AnalyzerInput>.Continuation
    /// Starts the module's result consumer. Kept as a closure because the
    /// two transcriber types have unrelated `Result` types.
    private let consumeResults: (AnalyzerStream) -> Task<Void, Never>

    // Feed side (any thread, serialized by `feedLock`).
    private let feedLock = NSLock()
    private let sourceFormat: AVAudioFormat
    private let targetFormat: AVAudioFormat
    private let converter: AVAudioConverter?
    private var inputClosed = false

    // Session state, only touched on `queue`.
    private var finalized = ""
    private var volatile = ""
    private var ended = false
    private var finishing = false
    /// The results sequence ended on its own: the analyzer is done and needs
    /// no stopping.
    private var analyzerFinished = false
    private let interimResults: Bool
    private var resultsTask: Task<Void, Never>?
    /// `analyzer.start` returns once analysis is running; finishing waits for
    /// it so a very short session never finalizes an analyzer not yet started.
    private var startTask: Task<Void, Never>?

    private init(id: UInt64, prepared: PreparedAnalyzer, sourceFormat: AVAudioFormat, interimResults: Bool) {
        self.id = id
        self.moduleName = prepared.moduleName
        self.queue = DispatchQueue(label: "com.yellowbites.stt-analyzer.\(id)", qos: .userInteractive)
        self.analyzer = prepared.analyzer
        self.sourceFormat = sourceFormat
        self.targetFormat = prepared.targetFormat
        self.converter = sourceFormat == prepared.targetFormat
            ? nil
            : AVAudioConverter(from: sourceFormat, to: prepared.targetFormat)
        self.interimResults = interimResults
        self.consumeResults = prepared.consume
        var continuation: AsyncStream<AnalyzerInput>.Continuation!
        self.inputStream = AsyncStream(bufferingPolicy: .unbounded) { continuation = $0 }
        self.inputContinuation = continuation
    }

    // MARK: Opening

    /// Set up an analyzer session, or nil when the caller should use
    /// `SFSpeechRecognizer` instead.
    ///
    /// With an analyzer prepared in advance (`prewarm`) this returns
    /// immediately. Otherwise it **blocks** on the Speech framework's async
    /// setup — call it off the main thread; Rust's `Recognizer::open` already
    /// runs on a blocking-capable thread. Either way the next session's
    /// analyzer is prepared in the background right after.
    static func open(id: UInt64, locale: Locale, sampleRate: Double, interimResults: Bool) -> AnalyzerStream? {
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }
        defer { AnalyzerPool.shared.refill(locale: locale) }
        let prepared: PreparedAnalyzer?
        if interimResults, let warm = AnalyzerPool.shared.take(locale: locale) {
            prepared = warm
        } else {
            // Model loading is the slow part. If setup takes longer than
            // this, the session goes to SFSpeechRecognizer and the late
            // analyzer is discarded.
            prepared = blockingAwait(timeout: 4.0, onAbandoned: { $0.discard() }) {
                await PreparedAnalyzer.make(locale: locale, interimResults: interimResults, sourceFormat: sourceFormat)
            } ?? nil
        }
        guard let prepared = prepared else { return nil }
        let stream = AnalyzerStream(id: id, prepared: prepared, sourceFormat: sourceFormat, interimResults: interimResults)
        guard stream.converter != nil || sourceFormat == prepared.targetFormat else {
            NSLog("[AnalyzerStream %llu] cannot convert %@ to %@", id,
                  sourceFormat.description, prepared.targetFormat.description)
            prepared.discard()
            return nil
        }
        return stream
    }

    /// Prepare an analyzer for `locale` in the background so the next `open`
    /// starts without waiting for the model (and start downloading its
    /// assets if they are missing). Non-blocking.
    static func prewarm(locale: Locale) {
        AnalyzerPool.shared.refill(locale: locale)
    }

    // MARK: RecognitionStream

    /// Begin consuming results and analyzing input. Audio fed before this is
    /// buffered by the input stream.
    func start() {
        queue.async {
            // Cancelled before it started: nothing to start.
            guard !self.ended else { return }
            self.resultsTask = self.consumeResults(self)
            let analyzer = self.analyzer
            let input = self.inputStream
            self.startTask = Task {
                do {
                    try await analyzer.start(inputSequence: input)
                } catch {
                    self.fail(error)
                }
            }
        }
    }

    func feed(_ samples: UnsafePointer<Int16>, count: UInt32) {
        guard count > 0 else { return }
        feedLock.lock()
        defer { feedLock.unlock() }
        guard !inputClosed,
              let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(count))
        else { return }
        buffer.frameLength = AVAudioFrameCount(count)
        buffer.int16ChannelData?[0].update(from: samples, count: Int(count))
        guard let converted = convert(buffer) else { return }
        inputContinuation.yield(AnalyzerInput(buffer: converted))
    }

    func finish() {
        closeInput()
        queue.async {
            guard !self.ended, !self.finishing else { return }
            self.finishing = true
            let analyzer = self.analyzer
            let started = self.startTask
            Task {
                await started?.value
                do {
                    // Ends the results sequence once everything is final.
                    try await analyzer.finalizeAndFinishThroughEndOfInput()
                } catch {
                    self.fail(error)
                }
            }
            // Never leave the host waiting (mirrors SpeechStream).
            self.queue.asyncAfter(deadline: .now() + 3.0) { [weak self] in
                guard let self = self, !self.ended else { return }
                NSLog("[AnalyzerStream %llu] finalize timeout — using current transcript", self.id)
                self.emitFinalAndEnd()
            }
        }
    }

    func cancel() {
        closeInput()
        queue.async { self.end() }
    }

    // MARK: Results (any thread → queue)

    fileprivate func receive(_ text: String, isFinal: Bool) {
        queue.async {
            guard !self.ended else { return }
            if isFinal {
                self.finalized += text
                self.volatile = ""
            } else {
                self.volatile = text
            }
            guard self.interimResults else { return }
            let partial = self.transcript
            if !partial.isEmpty {
                partial.withCString { rust_stt_stream_on_partial(self.id, $0) }
            }
        }
    }

    fileprivate func resultsEnded(_ error: Error?) {
        if let error = error, !(error is CancellationError) {
            fail(error)
            return
        }
        queue.async {
            self.analyzerFinished = true
            self.emitFinalAndEnd()
        }
    }

    private func fail(_ error: Error) {
        queue.async {
            guard !self.ended else { return }
            if self.finishing {
                // Ending a short/empty session can surface as an error; the
                // transcript so far is still the best answer.
                NSLog("[AnalyzerStream %llu] error while finishing: %@", self.id, error.localizedDescription)
                self.emitFinalAndEnd()
                return
            }
            error.localizedDescription.withCString { rust_stt_stream_on_error(self.id, $0) }
            self.end()
        }
    }

    // MARK: Helpers

    /// Committed text plus the current guess for the range still in flight.
    private var transcript: String {
        (finalized + volatile).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func emitFinalAndEnd() {
        guard !ended else { return }
        let text = transcript
        if !text.isEmpty {
            text.withCString { rust_stt_stream_on_final(self.id, $0) }
        }
        end()
    }

    /// The one place a session is torn down, however it ends: no more input
    /// (a failed child of a multi-language session would otherwise keep
    /// being fed) and the analyzer stopped unless it finished on its own.
    private func end() {
        guard !ended else { return }
        ended = true
        closeInput()
        if !analyzerFinished {
            stopAnalyzer()
        }
        resultsTask?.cancel()
        rust_stt_stream_on_ended(id)
        SttStreamRegistry.shared.remove(id)
    }

    private func closeInput() {
        feedLock.lock()
        defer { feedLock.unlock() }
        guard !inputClosed else { return }
        inputClosed = true
        inputContinuation.finish()
    }

    /// Stop the analyzer once its start has returned — stopping it before
    /// would let a late `start` bring it back up with nobody left to stop it.
    private func stopAnalyzer() {
        let analyzer = self.analyzer
        let started = self.startTask
        Task {
            await started?.value
            await analyzer.cancelAndFinishNow()
        }
    }

    /// Convert to the analyzer's format (called under `feedLock`; the
    /// converter keeps resampler state across calls).
    private func convert(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let converter = converter else { return buffer }
        let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
        let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return nil
        }
        var consumed = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if consumed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            inputStatus.pointee = .haveData
            return buffer
        }
        if status == .error {
            NSLog("[AnalyzerStream %llu] audio conversion failed: %@", id,
                  error?.localizedDescription ?? "unknown")
            return nil
        }
        return output.frameLength > 0 ? output : nil
    }
}

// MARK: - Prepared analyzers

/// A `SpeechAnalyzer` with its transcriber, model loaded and ready to start
/// on its first input. Single use: a stream takes it and finishes it.
@available(iOS 26, macOS 26, *)
final class PreparedAnalyzer {
    let moduleName: String
    let analyzer: SpeechAnalyzer
    let targetFormat: AVAudioFormat
    /// Starts the module's result consumer for a stream. A closure because
    /// the two transcriber types have unrelated `Result` types.
    let consume: (AnalyzerStream) -> Task<Void, Never>

    private init(
        moduleName: String,
        analyzer: SpeechAnalyzer,
        targetFormat: AVAudioFormat,
        consume: @escaping (AnalyzerStream) -> Task<Void, Never>
    ) {
        self.moduleName = moduleName
        self.analyzer = analyzer
        self.targetFormat = targetFormat
        self.consume = consume
    }

    /// Release an analyzer nobody will use.
    func discard() {
        let analyzer = self.analyzer
        Task { await analyzer.cancelAndFinishNow() }
    }

    /// Prepare the transcriber `TranscriberChoice.pick` selects; nil when
    /// none is usable right now (missing assets start downloading).
    /// `sourceFormat` is a hint for the analyzer's preferred input format.
    static func make(locale: Locale, interimResults: Bool, sourceFormat: AVAudioFormat?) async -> PreparedAnalyzer? {
        guard let choice = await TranscriberChoice.pick(locale: locale, interimResults: interimResults) else {
            NSLog("[AnalyzerStream] no transcriber for %@", locale.identifier)
            return nil
        }
        let status = await AssetInventory.status(forModules: [choice.module])
        guard status == .installed else {
            NSLog("[AnalyzerStream] %@ assets for %@ not installed (%@)",
                  choice.moduleName, choice.locale.identifier, String(describing: status))
            if status != .unsupported {
                AssetDownloads.shared.ensure(module: choice.module, key: choice.key)
            }
            return nil
        }
        guard let targetFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [choice.module],
            considering: sourceFormat
        ) else {
            NSLog("[AnalyzerStream] no compatible audio format")
            return nil
        }
        let analyzer = SpeechAnalyzer(
            modules: [choice.module],
            options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .lingering)
        )
        do {
            try await analyzer.prepareToAnalyze(in: targetFormat)
        } catch {
            NSLog("[AnalyzerStream] prepareToAnalyze failed: %@", error.localizedDescription)
            return nil
        }
        return PreparedAnalyzer(
            moduleName: choice.moduleName, analyzer: analyzer,
            targetFormat: targetFormat, consume: choice.consume
        )
    }
}

/// Which transcriber serves a locale: `SpeechTranscriber` where the device
/// supports it and covers the locale, else `DictationTranscriber`.
@available(iOS 26, macOS 26, *)
struct TranscriberChoice {
    let moduleName: String
    let module: any SpeechModule
    /// The transcriber's own locale equivalent to the requested one.
    let locale: Locale
    /// Starts the module's result consumer for a stream. A closure because
    /// the two transcriber types have unrelated `Result` types.
    let consume: (AnalyzerStream) -> Task<Void, Never>

    /// Identifies the asset set (transcriber + locale).
    var key: String { "\(moduleName)/\(locale.identifier)" }

    static func pick(locale: Locale, interimResults: Bool) async -> TranscriberChoice? {
        if SpeechTranscriber.isAvailable,
           let matched = await SpeechTranscriber.supportedLocale(equivalentTo: locale) {
            let module = SpeechTranscriber(
                locale: matched,
                preset: interimResults ? .progressiveTranscription : .transcription
            )
            return TranscriberChoice(moduleName: "SpeechTranscriber", module: module, locale: matched) { stream in
                Task {
                    do {
                        for try await result in module.results {
                            stream.receive(String(result.text.characters), isFinal: result.isFinal)
                        }
                        stream.resultsEnded(nil)
                    } catch {
                        stream.resultsEnded(error)
                    }
                }
            }
        }
        if let matched = await DictationTranscriber.supportedLocale(equivalentTo: locale) {
            let module = DictationTranscriber(
                locale: matched,
                preset: interimResults ? .progressiveLongDictation : .longDictation
            )
            return TranscriberChoice(moduleName: "DictationTranscriber", module: module, locale: matched) { stream in
                Task {
                    do {
                        for try await result in module.results {
                            stream.receive(String(result.text.characters), isFinal: result.isFinal)
                        }
                        stream.resultsEnded(nil)
                    } catch {
                        stream.resultsEnded(error)
                    }
                }
            }
        }
        return nil
    }
}

// MARK: - Model assets (status and foreground install)

/// Asset state of the transcriber that would serve a locale, for the host's
/// settings UI. Raw values are the C ABI of `stt_stream_model_status`.
@available(iOS 26, macOS 26, *)
enum SpeechAssets {
    enum State: Int32 {
        /// No analyzer transcriber covers the locale (SFSpeechRecognizer
        /// still serves it).
        case unsupported = 0
        case installed = 1
        case notInstalled = 2
        case downloading = 3
    }

    static func state(locale: Locale) async -> State {
        guard let choice = await TranscriberChoice.pick(locale: locale, interimResults: true) else {
            return .unsupported
        }
        switch await AssetInventory.status(forModules: [choice.module]) {
        case .installed: return .installed
        case .downloading: return .downloading
        case .supported: return .notInstalled
        default: return .unsupported
        }
    }

    enum InstallResult: Int32 {
        case installed = 0
        case unsupported = 1
        case failed = 2
    }

    /// Download and install the assets, reporting progress (whole percent,
    /// deduplicated) through `onProgress`; then prepare an analyzer so the
    /// next session starts warm.
    static func install(locale: Locale, onProgress: @escaping (Int32) -> Void) async -> InstallResult {
        guard let choice = await TranscriberChoice.pick(locale: locale, interimResults: true) else {
            return .unsupported
        }
        do {
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [choice.module]) {
                NSLog("[AnalyzerStream] installing speech assets for %@", choice.key)
                var last: Int32 = -1
                let lock = NSLock()
                let observation = request.progress.observe(\.fractionCompleted, options: [.initial, .new]) { progress, _ in
                    let percent = Int32((progress.fractionCompleted * 100).rounded(.down))
                    let changed = lock.withLock { () -> Bool in
                        guard percent != last else { return false }
                        last = percent
                        return true
                    }
                    if changed { onProgress(min(percent, 99)) }
                }
                defer { observation.invalidate() }
                try await request.downloadAndInstall()
                NSLog("[AnalyzerStream] speech assets for %@ installed", choice.key)
            }
        } catch {
            NSLog("[AnalyzerStream] asset install for %@ failed: %@", choice.key, error.localizedDescription)
            return .failed
        }
        AnalyzerStream.prewarm(locale: locale)
        return .installed
    }
}

/// At most one ready analyzer per locale (progressive preset, which is what
/// interim-result sessions use). Filled by `prewarm` and after every `open`,
/// so push-to-talk never waits for model loading after the first prewarm.
@available(iOS 26, macOS 26, *)
private final class AnalyzerPool {
    static let shared = AnalyzerPool()
    private let lock = NSLock()
    private var ready: [String: PreparedAnalyzer] = [:]
    private var preparing = Set<String>()

    func take(locale: Locale) -> PreparedAnalyzer? {
        lock.withLock { ready.removeValue(forKey: locale.identifier) }
    }

    func refill(locale: Locale) {
        let key = locale.identifier
        let start = lock.withLock { () -> Bool in
            guard ready[key] == nil, !preparing.contains(key) else { return false }
            preparing.insert(key)
            return true
        }
        guard start else { return }
        Task.detached(priority: .utility) {
            let hint = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: false)
            let prepared = await PreparedAnalyzer.make(locale: locale, interimResults: true, sourceFormat: hint)
            self.lock.withLock {
                self.preparing.remove(key)
                if let prepared = prepared {
                    self.ready[key] = prepared
                    NSLog("[AnalyzerStream] %@ ready for %@", prepared.moduleName, key)
                }
            }
        }
    }
}

// MARK: - Background asset downloads

/// Starts at most one download per transcriber/locale per process. Nothing
/// waits for it: the next session after it completes uses the analyzer.
@available(iOS 26, macOS 26, *)
private final class AssetDownloads {
    static let shared = AssetDownloads()
    private let lock = NSLock()
    private var inFlight = Set<String>()

    func ensure(module: any SpeechModule, key: String) {
        lock.lock()
        let inserted = inFlight.insert(key).inserted
        lock.unlock()
        guard inserted else { return }
        Task.detached(priority: .utility) {
            do {
                if let request = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
                    NSLog("[AnalyzerStream] downloading speech assets for %@", key)
                    try await request.downloadAndInstall()
                    NSLog("[AnalyzerStream] speech assets for %@ installed", key)
                }
            } catch {
                NSLog("[AnalyzerStream] asset download for %@ failed: %@", key, error.localizedDescription)
            }
            _ = self.lock.withLock { self.inFlight.remove(key) }
        }
    }
}

// MARK: - Blocking bridge

/// Run `body` on the Swift concurrency pool and block until it returns.
/// Never call this on a thread the awaited work might need (the main thread
/// if it touched the main actor; nothing here does).
@available(iOS 26, macOS 26, *)
func blockingRun<T>(_ body: @escaping () async -> T) -> T {
    let box = AwaitBox<T>()
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached(priority: .userInitiated) {
        let result = await body()
        box.lock.withLock { box.value = .some(result) }
        semaphore.signal()
    }
    semaphore.wait()
    return box.lock.withLock { box.value!! }
}

private final class AwaitBox<T> {
    let lock = NSLock()
    var value: T??
    var abandoned = false
}

/// Run `body` on the Swift concurrency pool and wait up to `timeout` seconds
/// for its result. On timeout returns nil; a result that arrives later is
/// handed to `onAbandoned` so it can be torn down. Never call this on a
/// thread the awaited work might need (it does not touch the main actor).
@available(iOS 26, macOS 26, *)
private func blockingAwait<T: AnyObject>(
    timeout: TimeInterval,
    onAbandoned: @escaping (T) -> Void,
    _ body: @escaping () async -> T?
) -> T?? {
    let box = AwaitBox<T>()
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached(priority: .userInitiated) {
        let result = await body()
        let abandoned = box.lock.withLock { () -> Bool in
            if !box.abandoned { box.value = .some(result) }
            return box.abandoned
        }
        if abandoned, let result = result {
            onAbandoned(result)
        }
        semaphore.signal()
    }
    let timedOut = semaphore.wait(timeout: .now() + timeout) == .timedOut
    box.lock.lock()
    defer { box.lock.unlock() }
    if timedOut, box.value == nil {
        box.abandoned = true
        NSLog("[AnalyzerStream] setup timed out after %.1f s", timeout)
        return nil
    }
    return box.value
}
