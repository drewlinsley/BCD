import Foundation
import Combine

/// One frozen overlay: a scored candidate pinned to where its detection sat in the captured
/// frame (normalized 0-1). Plain `Double`s, no CoreGraphics, so BCDKit stays portable.
public struct ResolvedOverlay: Identifiable, Sendable {
    public let id: String            // product id — also the per-product dedup key
    public let candidate: ScoredCandidate
    public let x: Double             // normalized box center, 0-1
    public let y: Double
    public init(id: String, candidate: ScoredCandidate, x: Double, y: Double) {
        self.id = id; self.candidate = candidate; self.x = x; self.y = y
    }
}

/// Drives the **capture-based** scan flow. The engine runs a live viewfinder and this buffers
/// its latest detections *without resolving* — pointing the camera costs nothing. A shutter tap
/// (`capture`) resolves that one frame, pins an overlay to each detection's bounding box, and
/// freezes them. `rescan` returns to the live viewfinder.
///
/// This deliberately replaces the old stream-and-accumulate design: resolving every frame
/// firehosed the backend with half-read OCR and piled overlays up forever. One sharp frame,
/// one resolve, a bounded anchored set — steadier, cheaper, and correct.
@MainActor
public final class ScanCoordinator: ObservableObject {
    /// Frozen overlays from the last capture, best-first, anchored to their boxes.
    @Published public private(set) var overlays: [ResolvedOverlay] = []
    @Published public private(set) var lastLatencyMs: Double?
    @Published public private(set) var isScanning = false
    /// A capture is in flight (shutter → response).
    @Published public private(set) var isResolving = false
    /// A result is frozen on screen (true even when nothing matched, so the UI can say so).
    @Published public private(set) var captured = false
    /// Candidates behind the current capture — used by the chat-bar rerank.
    @Published public private(set) var capturedCandidates: [ScoredCandidate] = []

    private let engine: ScanEngine
    private let api: APIClientProtocol
    private let telemetry: TelemetryQueue?
    private var latestFrame: [DetectedText] = []   // most recent live detections (with boxes)
    private var capturedFrame: [DetectedText] = []  // the frame the frozen overlays anchor to
    private var task: Task<Void, Never>?

    /// Cap overlays so a busy shelf stays legible (the server caps too).
    private let maxOverlays = 8

    public init(engine: ScanEngine, api: APIClientProtocol, telemetry: TelemetryQueue? = nil) {
        self.engine = engine
        self.api = api
        self.telemetry = telemetry
    }

    /// Start the live viewfinder. Frames are buffered, not resolved — nothing hits the network
    /// until `capture()`.
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

    public func stop() {
        engine.stop()
        task?.cancel()
        isScanning = false
    }

    /// Shutter. Resolve the current frame exactly once and freeze anchored overlays onto it.
    public func capture(venueId: String? = nil) async {
        let frame = latestFrame.filter { !$0.text.isEmpty }
        capturedFrame = frame
        guard !frame.isEmpty else {          // nothing in view — freeze an empty result
            overlays = []; capturedCandidates = []; captured = true
            return
        }
        isResolving = true
        defer { isResolving = false }
        let req = ScanResolveRequest(detections: frame, venueId: venueId, includeScore: true)
        do {
            let resp = try await api.resolveScan(req)
            lastLatencyMs = resp.latencyMs
            capturedCandidates = resp.candidates
            overlays = Self.anchor(resp.candidates, to: frame, cap: maxOverlays)
            captured = true
            await telemetry?.log("scan_frame_batch", tier: .personalization, [
                "n_detections": .int(frame.count),
                "n_resolved": .int(resp.candidates.count),
                "server_latency_ms": .double(resp.latencyMs ?? 0),
                "ocr_strings": .stringList(frame.map { $0.text }),
            ])
        } catch {
            // Leave `captured` false so the shutter can simply be tapped again.
        }
    }

    /// Back to the live viewfinder.
    public func rescan() {
        overlays = []; capturedCandidates = []; capturedFrame = []
        captured = false
    }

    /// Reorder the frozen overlays by an explicit product-id order (the chat-bar rerank),
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
