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
    private let interimResults: Bool
    private var resultsTask: Task<Void, Never>?
    /// `analyzer.start` returns once analysis is running; finishing waits for
    /// it so a very short session never finalizes an analyzer not yet started.
    private var startTask: Task<Void, Never>?

    private init(
        id: UInt64,
        moduleName: String,
        analyzer: SpeechAnalyzer,
        sourceFormat: AVAudioFormat,
        targetFormat: AVAudioFormat,
        interimResults: Bool,
        consumeResults: @escaping (AnalyzerStream) -> Task<Void, Never>
    ) {
        self.id = id
        self.moduleName = moduleName
        self.queue = DispatchQueue(label: "com.yellowbites.stt-analyzer.\(id)", qos: .userInteractive)
        self.analyzer = analyzer
        self.sourceFormat = sourceFormat
        self.targetFormat = targetFormat
        self.converter = sourceFormat == targetFormat
            ? nil
            : AVAudioConverter(from: sourceFormat, to: targetFormat)
        self.interimResults = interimResults
        self.consumeResults = consumeResults
        var continuation: AsyncStream<AnalyzerInput>.Continuation!
        self.inputStream = AsyncStream(bufferingPolicy: .unbounded) { continuation = $0 }
        self.inputContinuation = continuation
    }

    // MARK: Opening

    /// Set up an analyzer session, or nil when the caller should use
    /// `SFSpeechRecognizer` instead. **Blocking** (it awaits the Speech
    /// framework's async setup); call it off the main thread — Rust's
    /// `Recognizer::open` already runs on a blocking-capable thread.
    static func open(id: UInt64, locale: Locale, sampleRate: Double, interimResults: Bool) -> AnalyzerStream? {
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }
        // Model loading is the slow part and only happens on the first session
        // (`modelRetention: .lingering` keeps it warm afterwards). If setup
        // takes longer than this, the session goes to SFSpeechRecognizer and
        // the analyzer is discarded once it is ready.
        return blockingAwait(timeout: 4.0, onAbandoned: { $0.discard() }) {
            await makeStream(id: id, locale: locale, sourceFormat: sourceFormat, interimResults: interimResults)
        } ?? nil
    }

    private static func makeStream(
        id: UInt64,
        locale: Locale,
        sourceFormat: AVAudioFormat,
        interimResults: Bool
    ) async -> AnalyzerStream? {
        if SpeechTranscriber.isAvailable,
           let matched = await SpeechTranscriber.supportedLocale(equivalentTo: locale) {
            let module = SpeechTranscriber(
                locale: matched,
                preset: interimResults ? .progressiveTranscription : .transcription
            )
            return await prepare(
                id: id, moduleName: "SpeechTranscriber", module: module, locale: matched,
                sourceFormat: sourceFormat, interimResults: interimResults,
                consume: { stream in
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
            )
        }
        if let matched = await DictationTranscriber.supportedLocale(equivalentTo: locale) {
            let module = DictationTranscriber(
                locale: matched,
                preset: interimResults ? .progressiveLongDictation : .longDictation
            )
            return await prepare(
                id: id, moduleName: "DictationTranscriber", module: module, locale: matched,
                sourceFormat: sourceFormat, interimResults: interimResults,
                consume: { stream in
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
            )
        }
        NSLog("[AnalyzerStream %llu] no transcriber for %@", id, locale.identifier)
        return nil
    }

    private static func prepare(
        id: UInt64,
        moduleName: String,
        module: any SpeechModule,
        locale: Locale,
        sourceFormat: AVAudioFormat,
        interimResults: Bool,
        consume: @escaping (AnalyzerStream) -> Task<Void, Never>
    ) async -> AnalyzerStream? {
        let status = await AssetInventory.status(forModules: [module])
        guard status == .installed else {
            NSLog("[AnalyzerStream %llu] %@ assets for %@ not installed (%@)",
                  id, moduleName, locale.identifier, String(describing: status))
            if status != .unsupported {
                AssetDownloads.shared.ensure(module: module, key: "\(moduleName)/\(locale.identifier)")
            }
            return nil
        }
        guard let targetFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [module],
            considering: sourceFormat
        ) else {
            NSLog("[AnalyzerStream %llu] no compatible audio format", id)
            return nil
        }
        let analyzer = SpeechAnalyzer(
            modules: [module],
            options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .lingering)
        )
        do {
            try await analyzer.prepareToAnalyze(in: targetFormat)
        } catch {
            NSLog("[AnalyzerStream %llu] prepareToAnalyze failed: %@", id, error.localizedDescription)
            return nil
        }
        let stream = AnalyzerStream(
            id: id, moduleName: moduleName, analyzer: analyzer,
            sourceFormat: sourceFormat, targetFormat: targetFormat,
            interimResults: interimResults, consumeResults: consume
        )
        guard stream.converter != nil || sourceFormat == targetFormat else {
            NSLog("[AnalyzerStream %llu] cannot convert %@ to %@", id,
                  sourceFormat.description, targetFormat.description)
            return nil
        }
        return stream
    }

    // MARK: RecognitionStream

    /// Begin consuming results and analyzing input. Audio fed before this is
    /// buffered by the input stream.
    func start() {
        queue.async {
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
                self.stopAnalyzer()
            }
        }
    }

    func cancel() {
        closeInput()
        queue.async {
            guard !self.ended else { return }
            self.stopAnalyzer()
            self.end()
        }
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
        queue.async { self.emitFinalAndEnd() }
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
            self.stopAnalyzer()
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

    private func end() {
        guard !ended else { return }
        ended = true
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

    private func stopAnalyzer() {
        let analyzer = self.analyzer
        Task { await analyzer.cancelAndFinishNow() }
    }

    /// Tear down a stream that was set up but never handed to the host.
    fileprivate func discard() {
        closeInput()
        stopAnalyzer()
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
