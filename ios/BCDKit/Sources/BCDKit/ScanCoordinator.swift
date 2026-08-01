import Foundation
import Combine

/// Ties the pieces together: consumes detection frames from a `ScanEngine`, batches and
/// dedupes them, asks the backend to resolve + score, and publishes ranked candidates the
/// HUD renders. This is the client half of the latency-critical path; the on-device
/// dedupe here is what keeps us inside the <400ms budget by not re-querying stable text.
@MainActor
public final class ScanCoordinator: ObservableObject {
    @Published public private(set) var candidates: [ScoredCandidate] = []
    @Published public private(set) var lastLatencyMs: Double?
    @Published public private(set) var isScanning = false

    private let engine: ScanEngine
    private let api: APIClientProtocol
    private let telemetry: TelemetryQueue?
    private var seenTexts: Set<String> = []
    private var task: Task<Void, Never>?

    public init(engine: ScanEngine, api: APIClientProtocol, telemetry: TelemetryQueue? = nil) {
        self.engine = engine
        self.api = api
        self.telemetry = telemetry
    }

    public func start(venueId: String? = nil) {
        guard !isScanning else { return }
        isScanning = true
        task = Task { [weak self] in
            guard let self else { return }
            await self.engine.start()
            for await frame in self.engine.frames {
                await self.handle(frame, venueId: venueId)
            }
        }
    }

    public func stop() {
        engine.stop()
        task?.cancel()
        isScanning = false
    }

    /// Reset dedupe state, e.g. when the user pans to a new shelf.
    public func resetView() { seenTexts.removeAll(); candidates.removeAll() }

    private func handle(_ frame: [DetectedText], venueId: String?) async {
        // Only send detections we haven't already resolved this view — the core cost cut.
        let fresh = frame.filter { !seenTexts.contains($0.text) && !$0.text.isEmpty }
        guard !fresh.isEmpty else { return }
        for d in fresh { seenTexts.insert(d.text) }

        let req = ScanResolveRequest(detections: fresh, venueId: venueId, includeScore: true)
        do {
            let resp = try await api.resolveScan(req)
            merge(resp)
            lastLatencyMs = resp.latencyMs
            await telemetry?.log("scan_frame_batch", tier: .personalization, [
                "n_detections": .int(fresh.count),
                "n_resolved": .int(resp.candidates.count),
                "server_latency_ms": .double(resp.latencyMs ?? 0),
                "ocr_strings": .stringList(fresh.map { $0.text }),
            ])
        } catch {
            // Network hiccup: allow these texts to retry on the next frame.
            for d in fresh { seenTexts.remove(d.text) }
        }
    }

    private func merge(_ resp: ScanResolveResponse) {
        var byId: [String: ScoredCandidate] = [:]
        for c in candidates { byId[c.resolved.id] = c }
        for c in resp.candidates { byId[c.resolved.id] = c }
        // Best predicted enjoyment first; the HUD color-codes by this.
        candidates = byId.values.sorted {
            ($0.personalScore ?? $0.matchScore) > ($1.personalScore ?? $1.matchScore)
        }
    }
}
