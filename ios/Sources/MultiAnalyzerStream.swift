import Foundation

// MARK: - One session, several languages (iOS 26+)
//
// Apple's transcribers take exactly one locale and do no language
// identification. To hear a second language (a child learning French in a
// German session) the same audio runs through one `AnalyzerStream` per
// language. Each child produces its own transcript and a confidence; when all
// have finished, every non-empty transcript is reported as an *alternative*
// (`rust_stt_stream_on_alternative`) and the most confident one as the
// session's `final`. The host decides what to do with the alternatives — the
// tutor app hands them to its LLM, which knows from the conversation which
// language the child was speaking.
//
// Live partials come from the primary (first) language only.

@available(iOS 26, macOS 26, *)
final class MultiAnalyzerStream: RecognitionStream {
    let id: UInt64
    private let queue: DispatchQueue
    private let interimResults: Bool
    private var children: [AnalyzerStream] = []

    // Only touched on `queue`.
    private var finals: [Int: (text: String, confidence: Double?)] = [:]
    private var errors: [String] = []
    private var endedCount = 0
    private var done = false

    private init(id: UInt64, interimResults: Bool) {
        self.id = id
        self.queue = DispatchQueue(label: "com.yellowbites.stt-multi.\(id)", qos: .userInteractive)
        self.interimResults = interimResults
    }

    /// Open one analyzer per locale; the first is the primary language.
    ///
    /// Nil when the primary language has no usable analyzer (the caller falls
    /// back to `SFSpeechRecognizer` for it alone). Extra languages only join
    /// when an analyzer for them is already prepared — `prewarm` them first —
    /// so a missing one never delays the session; it starts preparing and
    /// joins the next session.
    static func open(
        id: UInt64,
        locales: [Locale],
        sampleRate: Double,
        interimResults: Bool,
        contextualStrings: [String]
    ) -> MultiAnalyzerStream? {
        guard let primary = locales.first else { return nil }
        let stream = MultiAnalyzerStream(id: id, interimResults: interimResults)
        for (offset, locale) in locales.enumerated() {
            let sink = ChildSink(parent: stream, index: stream.children.count)
            guard let child = AnalyzerStream.open(
                id: id,
                locale: locale,
                sampleRate: sampleRate,
                interimResults: interimResults,
                contextualStrings: contextualStrings,
                waitForSetup: offset == 0,
                sink: sink
            ) else {
                if offset == 0 {
                    return nil
                }
                NSLog("[MultiAnalyzerStream %llu] %@ not ready, skipped this session", id, locale.identifier)
                continue
            }
            stream.children.append(child)
        }
        NSLog("[MultiAnalyzerStream %llu] %@ + %d extra language(s)",
              id, primary.identifier, stream.children.count - 1)
        return stream
    }

    var moduleNames: String {
        children.map { "\($0.moduleName)(\($0.locale.identifier))" }.joined(separator: ", ")
    }

    // MARK: RecognitionStream

    func start() {
        children.forEach { $0.start() }
    }

    func feed(_ samples: UnsafePointer<Int16>, count: UInt32) {
        children.forEach { $0.feed(samples, count: count) }
    }

    func finish() {
        children.forEach { $0.finish() }
    }

    func cancel() {
        children.forEach { $0.cancel() }
    }

    // MARK: Child events (any thread → queue)

    fileprivate func childPartial(_ index: Int, _ text: String) {
        guard index == 0, interimResults else { return }
        queue.async {
            guard !self.done else { return }
            text.withCString { rust_stt_stream_on_partial(self.id, $0) }
        }
    }

    fileprivate func childFinal(_ index: Int, _ text: String, _ confidence: Double?) {
        queue.async { self.finals[index] = (text, confidence) }
    }

    fileprivate func childError(_ index: Int, _ message: String) {
        queue.async {
            NSLog("[MultiAnalyzerStream %llu] language %d failed: %@", self.id, index, message)
            self.errors.append(message)
        }
    }

    fileprivate func childEnded(_ index: Int) {
        queue.async {
            self.endedCount += 1
            if self.endedCount == self.children.count {
                self.conclude()
            }
        }
    }

    /// All children are done: report candidates, the best as final, end.
    private func conclude() {
        guard !done else { return }
        done = true
        let candidates = finals
            .sorted { $0.key < $1.key }
            .filter { !$0.value.text.isEmpty }
        if candidates.isEmpty, let message = errors.first, errors.count == children.count {
            message.withCString { rust_stt_stream_on_error(id, $0) }
        }
        if candidates.count > 1 {
            for (index, candidate) in candidates {
                let locale = children[index].locale.identifier
                locale.withCString { localePtr in
                    candidate.text.withCString { textPtr in
                        rust_stt_stream_on_alternative(id, localePtr, textPtr, candidate.confidence ?? -1)
                    }
                }
            }
        }
        // Most confident wins; on a tie (or without confidences) the primary
        // language, which comes first.
        if let best = candidates.max(by: { ($0.value.confidence ?? 0) < ($1.value.confidence ?? 0) }) {
            best.value.text.withCString { rust_stt_stream_on_final(id, $0) }
        }
        rust_stt_stream_on_ended(id)
        SttStreamRegistry.shared.remove(id)
    }
}

/// Routes one child stream's events to its parent.
@available(iOS 26, macOS 26, *)
private final class ChildSink: StreamSink {
    private weak var parent: MultiAnalyzerStream?
    private let index: Int

    init(parent: MultiAnalyzerStream, index: Int) {
        self.parent = parent
        self.index = index
    }

    func partial(_ text: String) { parent?.childPartial(index, text) }
    func final(_ text: String, confidence: Double?) { parent?.childFinal(index, text, confidence) }
    func error(_ message: String) { parent?.childError(index, message) }
    func ended() { parent?.childEnded(index) }
}
