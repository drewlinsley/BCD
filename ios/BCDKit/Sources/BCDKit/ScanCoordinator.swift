import Foundation
import Combine

/// One overlay: a scored candidate pinned to where its detection sat in the resolved frame
/// (normalized 0-1). Plain `Double`s, no CoreGraphics, so BCDKit stays portable.
public struct ResolvedOverlay: Identifiable, Sendable {
    public let id: String            // product id — also the per-product dedup key
    public let candidate: ScoredCandidate
    public let x: Double             // normalized box center, 0-1
    public let y: Double
    public init(id: String, candidate: ScoredCandidate, x: Double, y: Double) {
        self.id = id; self.candidate = candidate; self.x = x; self.y = y
    }
}

/// Drives the scan flow — a **camera-first, fully live HUD**. There is no shutter and no freeze:
/// the engine runs a live viewfinder, this buffers its latest detections, and a fixed-rate ticker
/// re-resolves the latest frame every `intervalMs`, **replacing** the overlays each tick. Overlays
/// are always *assigned*, never appended, so nothing accumulates.
///
/// Two things keep the always-on loop cheap and useful:
///   - A held-still camera (unchanged OCR) skips the network round-trip entirely.
///   - When a frame has readable text but the catalog resolves *nothing* (a stylized label OCR'd as
///     garbage), it auto-invokes the on-device model to name the product — once per distinct frame,
///     off the tick's critical path only when it's actually stuck.
///
/// A persistent natural-language filter ("nothing over 6%") is parsed once into a structured intent
/// and applied synchronously to every tick, so it keeps filtering as the frame refreshes.
@MainActor
public final class ScanCoordinator: ObservableObject {
    /// Overlays from the most recent resolve, display-ordered, anchored to their boxes.
    @Published public private(set) var overlays: [ResolvedOverlay] = []
    @Published public private(set) var lastLatencyMs: Double?
    @Published public private(set) var isScanning = false
    /// A resolve is in flight (a live tick).
    @Published public private(set) var isResolving = false
    /// The on-device model is naming a stylized label (the automatic fallback).
    @Published public private(set) var isInterpreting = false
    /// Candidates behind the current overlays (pre-filter), so a filter change can re-pin without
    /// another round-trip.
    @Published public private(set) var candidates: [ScoredCandidate] = []
    /// The active natural-language filter text, for the HUD to display (nil = no filter).
    @Published public private(set) var filterText: String?

    private let engine: ScanEngine
    private let api: APIClientProtocol
    private let telemetry: TelemetryQueue?
    private let llm: LLMProvider?                   // on-device model for the stuck-frame fallback
    private var latestFrame: [DetectedText] = []    // most recent live detections (with boxes)
    private var currentFrame: [DetectedText] = []   // the frame the current overlays anchor to
    private var task: Task<Void, Never>?            // frame-buffer pump
    private var liveTask: Task<Void, Never>?        // fixed-rate resolve ticker
    /// The on-device-model fallback, if one is in flight. Public so a test can await it
    /// deterministically — the live loop deliberately does not.
    public private(set) var interpretation: Task<Void, Never>?
    /// Text signature of the last frame we resolved; lets a tick skip a re-resolve when the camera
    /// is held still (same OCR), keeping the fixed rate cheap and the overlays stable.
    private var lastResolvedKey: String?
    /// Text signature of the last frame we auto-ran the on-device model on, so a held-still garbled
    /// label triggers it once rather than every tick.
    private var lastInterpretKey: String?
    /// Parsed chat-bar intent, applied to every tick's candidates.
    private var filterIntent: QueryIntent?
    /// Whether the last resolve found something the frame agreed on. An in-flight model guess
    /// is dropped only if this became true while it was thinking.
    private var lastResolveCorroborated = false

    /// Cap overlays so a busy shelf stays legible (the server caps too).
    private let maxOverlays = 8

    /// How long an earned result stays up when later ticks resolve nothing. Live OCR jitters
    /// constantly — glare, a hand shake, a stylized label the catalog can't match — and every
    /// such tick used to blank the HUD immediately. That made an on-device-model result almost
    /// impossible to tap: it set the overlay without moving `lastResolvedKey`, so the very next
    /// tick re-resolved the raw garble, matched nothing, and wiped it. A result now stands until
    /// something better replaces it, the view empties, or this window passes.
    private let overlayHoldMs: Double = 2500
    private var overlaysSetAt: Date?

    /// Whether the overlays on screen are recent enough to keep through an empty resolve.
    private var isHoldingRecentOverlays: Bool {
        guard !overlays.isEmpty, let at = overlaysSetAt else { return false }
        return Date().timeIntervalSince(at) * 1000 < overlayHoldMs
    }

    public init(engine: ScanEngine, api: APIClientProtocol, telemetry: TelemetryQueue? = nil,
                llm: LLMProvider? = nil) {
        self.engine = engine
        self.api = api
        self.telemetry = telemetry
        self.llm = llm
    }

    /// Start the live viewfinder buffering frames. Nothing hits the network until a live tick —
    /// call `startLive()` for the fixed-rate loop.
    public func start(venueId: String? = nil) {
        guard !isScanning else { return }
        isScanning = true
        task = Task { [weak self] in
            guard let self else { return }
            await self.engine.start()
            for await frame in self.engine.frames {
                self.latestFrame = frame
            }
        }
    }

    /// Start the viewfinder **and** the fixed-rate resolve loop: every `intervalMs`, resolve the
    /// latest frame and replace the overlays. This is the whole scan interaction — no tapping.
    public func startLive(intervalMs: UInt64 = 350, venueId: String? = nil) {
        start(venueId: venueId)
        startTicker(intervalMs: intervalMs, venueId: venueId)
    }

    private func startTicker(intervalMs: UInt64, venueId: String?) {
        guard liveTask == nil else { return }
        liveTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: intervalMs * 1_000_000)
                guard let self, !Task.isCancelled else { return }
                await self.resolveLatest(venueId: venueId)
            }
        }
    }

    public func stop() {
        engine.stop()
        task?.cancel(); task = nil
        liveTask?.cancel(); liveTask = nil
        interpretation?.cancel(); interpretation = nil
        // The flag is raised before the task starts, so a task cancelled before its body ran
        // would otherwise leave it stuck up and block the fallback for the rest of the session.
        isInterpreting = false
        isScanning = false
    }

    /// One live tick: re-resolve the latest frame and swap overlays in place. Exposed so the
    /// fixed-rate behavior is unit-testable without a real clock.
    public func resolveLatest(venueId: String? = nil) async {
        let full = latestFrame.filter { !$0.text.isEmpty }
        await resolve(frame: Self.prioritised(full), full: full, venueId: venueId)
    }

    /// How many text lines a frame sends to the catalog. A label's brand and product name are
    /// its largest text; the rest is chrome.
    ///
    /// Every extra line costs a trigram scan server-side, priced by how common its words are
    /// rather than by how many rows come back — on a Heady Topper can "STOWE VERMONT" alone
    /// cost 1.1s and contributed nothing but a wrong answer. Sending everything made a
    /// six-line frame ~1.6s against a 700ms tick.
    static let maxTextLines = 3

    /// The lines worth resolving, largest first. Barcodes are never dropped: one is a
    /// definitive answer and costs a keyed lookup, not a scan.
    ///
    /// The returned array is what gets sent *and* what overlays anchor to, so it must stay the
    /// single source of truth for a candidate's `detectionIndex`.
    static func prioritised(_ frame: [DetectedText]) -> [DetectedText] {
        let barcodes = frame.filter { $0.kind == "barcode" }
        let text = frame.filter { $0.kind != "barcode" }
            .sorted(by: preferred)
            .prefix(maxTextLines)
        return barcodes + text
    }

    /// A strict total order, so the same frame always sends the same lines: box area, then
    /// OCR confidence, then length, then the text itself. Area alone ties too often — a
    /// detector that reports no box at all gives every line an area of zero.
    private static func preferred(_ lhs: DetectedText, _ rhs: DetectedText) -> Bool {
        let (la, ra) = ((lhs.w ?? 0) * (lhs.h ?? 0), (rhs.w ?? 0) * (rhs.h ?? 0))
        if la != ra { return la > ra }
        let (lc, rc) = (lhs.confidence ?? 0, rhs.confidence ?? 0)
        if lc != rc { return lc > rc }
        if lhs.text.count != rhs.text.count { return lhs.text.count > rhs.text.count }
        return lhs.text < rhs.text
    }

    /// The single resolve path. Resolve the frame, pin an overlay to each detection's box, dedupe
    /// per product, cap — then, if nothing matched but the label carried text, fall back to the
    /// on-device model. Always *assigns* overlays, so nothing accumulates.
    private func resolve(frame: [DetectedText], full: [DetectedText],
                         venueId: String?) async {
        guard !frame.isEmpty else {
            // Nothing in view: clear so a stale result doesn't linger over an empty shelf.
            overlays = []; candidates = []; currentFrame = []
            overlaysSetAt = nil
            lastResolvedKey = nil; lastInterpretKey = nil
            return
        }
        // Skip the round-trip when the OCR is unchanged since the last resolve (camera held still).
        let key = Self.signature(frame)
        if key == lastResolvedKey { return }

        isResolving = true
        defer { isResolving = false }
        let req = ScanResolveRequest(detections: frame, venueId: venueId, includeScore: true)
        do {
            let resp = try await api.resolveScan(req)
            lastLatencyMs = resp.latencyMs
            // Branch on what the *catalog* returned, not on what survives the filter: an
            // active filter legitimately hides everything and must keep doing so, while a
            // tick that resolved nothing at all should not throw away the last good result.
            if !resp.candidates.isEmpty {
                candidates = resp.candidates
                currentFrame = frame
                overlays = Self.anchor(Self.orderedForDisplay(resp.candidates, filterIntent),
                                       to: frame, cap: maxOverlays, presorted: true)
                overlaysSetAt = Date()
            } else if !isHoldingRecentOverlays {
                candidates = []; currentFrame = []; overlays = []
                overlaysSetAt = nil
            }
            lastResolvedKey = key
            lastResolveCorroborated = resp.corroborated
            await telemetry?.log("scan_frame_batch", tier: .personalization, [
                "n_detections": .int(frame.count),
                "n_resolved": .int(resp.candidates.count),
                "server_latency_ms": .double(resp.latencyMs ?? 0),
                "mode": .string("live"),
                "ocr_strings": .stringList(frame.map { $0.text }),
            ])
            // Auto-fallback: readable text, but the catalog did not really recognise it. Once
            // per distinct OCR frame, so a held-still garbled label doesn't re-run every tick.
            //
            // The trigger is "nothing the frame corroborates", not "nothing came back". Those
            // were assumed to be the same thing and are not: on a real Heady Topper can the
            // wordmark OCR'd as Cyrillic, a rim fragment matched a distillery named `Chemist`,
            // and that single confident-looking row was enough to make `candidates.isEmpty`
            // false for eleven frames running — so the fallback built for exactly this label
            // never once ran. A guess off one fragment must not suppress the model; only real
            // agreement across the frame should.
            if !resp.corroborated, !isInterpreting, let llm, key != lastInterpretKey {
                lastInterpretKey = key
                // The model call takes ~1s; awaiting it here froze the whole HUD for that
                // long. Detached, the fixed-rate loop keeps ticking.
                //
                // Crucially it is NOT cancelled and restarted when the frame changes. It was,
                // and at a 350ms tick against a ~1s call that meant every attempt was killed
                // by the next tick: on a real Focal Banger can the model completed **once in
                // 19 frames**, because garbled OCR is never byte-identical three ticks running
                // and only an unchanged frame let a call survive. Starting one only when none
                // is in flight is what actually lets the fallback run.
                //
                // `isInterpreting` is set here rather than inside the task: the tick that
                // would clobber it can run before the task body starts, so the flag has to go
                // up synchronously at the moment we decide to think.
                isInterpreting = true
                interpretation = Task { [weak self] in
                    // `full`, not `frame`: the model is priced per call, not per line, and
                    // the chrome the catalog cannot use is context that helps it guess.
                    await self?.interpret(frame: full, using: llm, key: key, venueId: venueId)
                }
            }
        } catch {
            // Keep the last good overlays and try again next tick.
        }
    }

    /// Stuck-frame fallback: hand the raw (garbled) OCR to the on-device model, resolve the product
    /// name it returns, and anchor to the label's most prominent text box.
    private func interpret(frame: [DetectedText], using llm: LLMProvider,
                           key: String, venueId: String?) async {
        // Raised synchronously by the caller so a tick cannot start a second call; lowered
        // here on every path out, including the guards below.
        defer { isInterpreting = false }
        let guesses = (try? await llm.interpretLabels(frame.map { $0.text })) ?? []
        guard !guesses.isEmpty else { return }
        // Staleness used to mean "the OCR changed while we were thinking". On a label the
        // camera cannot read, the OCR changes every single tick — same can, different garble —
        // so that test threw away nearly every guess it did manage to produce. What actually
        // makes a guess stale is the catalog having recognised something on its own since we
        // asked; a re-read of the same unreadable can has not.
        guard !lastResolveCorroborated else { return }
        let box = frame.max { ($0.w ?? 0) * ($0.h ?? 0) < ($1.w ?? 0) * ($1.h ?? 0) } ?? frame[0]
        let synthetic = guesses.map {
            DetectedText(text: $0, kind: "text", x: box.x, y: box.y, w: box.w, h: box.h)
        }
        let req = ScanResolveRequest(detections: synthetic, venueId: venueId, includeScore: true)
        guard let resp = try? await api.resolveScan(req), !resp.candidates.isEmpty else { return }
        candidates = resp.candidates
        currentFrame = synthetic
        overlays = Self.anchor(Self.orderedForDisplay(resp.candidates, filterIntent),
                               to: synthetic, cap: maxOverlays, presorted: true)
        overlaysSetAt = Date()   // starts the hold window, so this one is tappable
        lastLatencyMs = resp.latencyMs
        await telemetry?.log("scan_frame_batch", tier: .personalization, [
            "n_detections": .int(frame.count),
            "n_resolved": .int(resp.candidates.count),
            "mode": .string("llm_assist"),
            "llm_guesses": .stringList(guesses),
            "ocr_strings": .stringList(frame.map { $0.text }),
        ])
    }

    // MARK: - persistent natural-language filter (the chat bar)

    /// Set the chat-bar filter. Parse the ask into a structured intent ONCE (via the LLM), then
    /// apply it synchronously to every live tick — so "nothing over 6%" keeps hiding the 8% IPA as
    /// the frame refreshes, instead of a one-shot reorder the next tick would wipe.
    public func setFilter(_ ask: String) async {
        let trimmed = ask.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { clearFilter(); return }
        filterText = trimmed
        if let llm { filterIntent = try? await llm.parseQuery(trimmed) }
        reanchor()
    }

    public func clearFilter() {
        filterText = nil
        filterIntent = nil
        reanchor()
    }

    /// Re-pin the current candidates under the current filter without a new network round-trip.
    private func reanchor() {
        overlays = Self.anchor(Self.orderedForDisplay(candidates, filterIntent),
                               to: currentFrame, cap: maxOverlays, presorted: true)
    }

    // MARK: - helpers

    private static func signature(_ frame: [DetectedText]) -> String {
        frame.map { $0.text }.sorted().joined(separator: "\u{1}")
    }

    /// Filter candidates by the parsed intent, then order them for display. With no intent, order
    /// by predicted enjoyment (personal, else match). The scan payload carries no price, so a
    /// `.price` ask falls back to that same order.
    private static func orderedForDisplay(_ cands: [ScoredCandidate],
                                          _ intent: QueryIntent?) -> [ScoredCandidate] {
        var out = cands
        if let intent {
            out = out.filter { c in
                let abv = c.resolved.product.spec.abvPct?.value
                if let mx = intent.maxAbv, let a = abv, a > mx { return false }
                if let mn = intent.minAbv, let a = abv, a < mn { return false }
                if let style = intent.styleContains, !style.isEmpty {
                    let s = style.lowercased()
                    let hit = (c.resolved.product.style?.value.lowercased().contains(s) ?? false)
                        || c.resolved.product.name.lowercased().contains(s)
                    if !hit { return false }
                }
                return true
            }
        }
        switch intent?.sortBy ?? .personal {
        case .abv:
            out.sort { ($0.resolved.product.spec.abvPct?.value ?? .greatestFiniteMagnitude)
                     < ($1.resolved.product.spec.abvPct?.value ?? .greatestFiniteMagnitude) }
        case .relevance:
            out.sort { $0.matchScore > $1.matchScore }
        case .personal, .price:
            out.sort { ($0.personalScore ?? $0.matchScore) > ($1.personalScore ?? $1.matchScore) }
        }
        return out
    }

    /// Pin candidates to their detection's box center, one per product, capped. `presorted` keeps
    /// the caller's display order (the filter/sort already decided it).
    private static func anchor(_ candidates: [ScoredCandidate], to frame: [DetectedText],
                               cap: Int, presorted: Bool = false) -> [ResolvedOverlay] {
        let ordered = presorted ? candidates : candidates.sorted {
            ($0.personalScore ?? $0.matchScore) > ($1.personalScore ?? $1.matchScore)
        }
        var out: [ResolvedOverlay] = []
        var seen = Set<String>()
        for c in ordered {
            let pid = c.resolved.product.id
            guard !seen.contains(pid),
                  c.detectionIndex >= 0, c.detectionIndex < frame.count else { continue }
            let d = frame[c.detectionIndex]
            out.append(ResolvedOverlay(
                id: pid, candidate: c,
                x: (d.x ?? 0) + (d.w ?? 0) / 2,
                y: (d.y ?? 0) + (d.h ?? 0) / 2))
            seen.insert(pid)
            if out.count >= cap { break }
        }
        return out
    }
}
