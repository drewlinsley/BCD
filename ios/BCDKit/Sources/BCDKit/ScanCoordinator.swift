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

/// How the viewfinder decides *when* to resolve.
///   - `.capture` waits for a shutter tap and then freezes the result.
///   - `.live(intervalMs:)` re-resolves the latest frame on a fixed cadence, **replacing**
///     overlays each tick — never accumulating. This is the on-device / fixed-update-rate mode.
public enum ScanMode: Sendable, Equatable {
    case capture
    case live(intervalMs: UInt64)
}

/// Drives the scan flow. The engine runs a live viewfinder and this buffers its latest
/// detections. In `.capture` mode a shutter tap resolves that one frame and freezes it; in
/// `.live` mode a fixed-rate ticker resolves the latest frame every `intervalMs` and swaps the
/// overlays in place.
///
/// Both paths funnel through one `resolve(frame:freeze:)`: resolve once, pin an overlay to each
/// detection's bounding box, dedupe per product, cap. Overlays are always **assigned**, never
/// appended — so nothing accumulates the way the original stream-and-merge design did.
@MainActor
public final class ScanCoordinator: ObservableObject {
    /// Overlays from the most recent resolve, best-first, anchored to their boxes.
    @Published public private(set) var overlays: [ResolvedOverlay] = []
    @Published public private(set) var lastLatencyMs: Double?
    @Published public private(set) var isScanning = false
    /// A resolve is in flight (shutter tap, or a live tick).
    @Published public private(set) var isResolving = false
    /// A result is frozen on screen (capture mode). Live mode leaves this false — overlays
    /// keep updating — so the UI shows a "live" affordance instead of "Rescan".
    @Published public private(set) var captured = false
    /// Candidates behind the current overlays — used by the chat-bar rerank.
    @Published public private(set) var capturedCandidates: [ScoredCandidate] = []
    /// Current resolve cadence, so the UI (and tests) can see which mode is running.
    @Published public private(set) var mode: ScanMode = .capture

    private let engine: ScanEngine
    private let api: APIClientProtocol
    private let telemetry: TelemetryQueue?
    private var latestFrame: [DetectedText] = []   // most recent live detections (with boxes)
    private var capturedFrame: [DetectedText] = []  // the frame the current overlays anchor to
    private var task: Task<Void, Never>?            // frame-buffer pump
    private var liveTask: Task<Void, Never>?        // fixed-rate resolve ticker
    /// Text signature of the last frame we resolved; lets a live tick skip a re-resolve when the
    /// camera is held still (same OCR), keeping the fixed rate cheap and the overlays stable.
    private var lastResolvedKey: String?

    /// Cap overlays so a busy shelf stays legible (the server caps too).
    private let maxOverlays = 8

    public init(engine: ScanEngine, api: APIClientProtocol, telemetry: TelemetryQueue? = nil) {
        self.engine = engine
        self.api = api
        self.telemetry = telemetry
    }

    /// Start the live viewfinder buffering frames. Nothing hits the network until a `capture()`
    /// or a live tick — call `startLive()` for the fixed-rate mode.
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

    /// Start the viewfinder **and** a fixed-rate resolve loop: every `intervalMs`, resolve the
    /// latest frame and replace the overlays. This is the on-device fixed-update-rate mode.
    public func startLive(intervalMs: UInt64 = 700, venueId: String? = nil) {
        mode = .live(intervalMs: intervalMs)
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
        isScanning = false
    }

    /// Shutter. Freeze the current frame's result: pause any live ticker and resolve once.
    public func capture(venueId: String? = nil) async {
        pauseTicker()
        await resolve(frame: latestFrame.filter { !$0.text.isEmpty }, freeze: true, venueId: venueId)
    }

    /// One live tick: re-resolve the latest frame and swap overlays in place. Exposed so the
    /// fixed-rate behavior is unit-testable without a real clock.
    public func resolveLatest(venueId: String? = nil) async {
        await resolve(frame: latestFrame.filter { !$0.text.isEmpty }, freeze: false, venueId: venueId)
    }

    /// The single resolve path. `freeze` = capture mode (pin the result and stop updating);
    /// otherwise it's a live tick that will be replaced on the next cadence.
    private func resolve(frame: [DetectedText], freeze: Bool, venueId: String?) async {
        guard !frame.isEmpty else {
            // Nothing in view. Capture freezes an empty "no products" result; a live tick just
            // clears the overlays and stays live so a stale result doesn't linger over an empty shelf.
            overlays = []; capturedCandidates = []; capturedFrame = []; lastResolvedKey = nil
            if freeze { captured = true }
            return
        }
        // Live tick: skip the round-trip when the OCR is unchanged since the last resolve.
        let key = frame.map { $0.text }.sorted().joined(separator: "\u{1}")
        if !freeze, key == lastResolvedKey { return }

        isResolving = true
        defer { isResolving = false }
        let req = ScanResolveRequest(detections: frame, venueId: venueId, includeScore: true)
        do {
            let resp = try await api.resolveScan(req)
            lastLatencyMs = resp.latencyMs
            capturedCandidates = resp.candidates
            capturedFrame = frame
            overlays = Self.anchor(resp.candidates, to: frame, cap: maxOverlays)
            lastResolvedKey = key
            if freeze { captured = true }
            await telemetry?.log("scan_frame_batch", tier: .personalization, [
                "n_detections": .int(frame.count),
                "n_resolved": .int(resp.candidates.count),
                "server_latency_ms": .double(resp.latencyMs ?? 0),
                "mode": .string(freeze ? "capture" : "live"),
                "ocr_strings": .stringList(frame.map { $0.text }),
            ])
        } catch {
            // Live: keep the last good overlays and try again next tick. Capture: stay unfrozen
            // so the shutter can simply be tapped again.
        }
    }

    /// Leave a frozen capture and return to the viewfinder. In live mode this resumes the ticker.
    public func rescan() {
        overlays = []; capturedCandidates = []; capturedFrame = []
        captured = false; lastResolvedKey = nil
        if case .live(let ms) = mode { startTicker(intervalMs: ms, venueId: nil) }
    }

    private func pauseTicker() {
        liveTask?.cancel(); liveTask = nil
    }

    /// Reorder the current overlays by an explicit product-id order (the chat-bar rerank),
    /// keeping every overlay pinned to its box.
    public func rerank(order: [String]) {
        guard !order.isEmpty, !capturedCandidates.isEmpty else { return }
        let rank = Dictionary(order.enumerated().map { ($1, $0) },
                              uniquingKeysWith: { a, _ in a })
        let reordered = capturedCandidates.sorted {
            (rank[$0.resolved.product.id] ?? Int.max) < (rank[$1.resolved.product.id] ?? Int.max)
        }
        overlays = Self.anchor(reordered, to: capturedFrame, cap: maxOverlays, presorted: true)
    }

    /// Pin candidates to their detection's box center, one per product, best-first, capped.
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
