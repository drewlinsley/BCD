import Foundation
import Combine

/// Ties the pieces together — the client half of the coarse-to-fine scan path.
///
///   coarse   `ScanEngine` frames → `ObjectTracker` → objects with stable evidence
///   server   one `/v1/scan/resolve` call per batch of ready objects → a verdict each
///   fine 1   `resolved`?  overlay.  Otherwise a careful OCR pass on that object's crop
///            (`FineTextReader`) → new text → re-query.
///   fine 2   still `ambiguous`? the on-device model picks among the server's shortlist
///            (`LLMProvider.pickProduct`) — it cannot invent an answer.
///   else     the object stays `unresolved` / `ambiguous` and the HUD shows *no name*.
///
/// The old coordinator queried every fresh text line and showed the first hit. This one
/// never shows a name the server or the model didn't clear, and it queries per object,
/// after two frames of agreement, which is what keeps wrong overlays off the screen and
/// the request rate inside the latency budget.
@MainActor
public final class ScanCoordinator: ObservableObject {
    public struct Policy: Sendable {
        /// Raise the server's floor for a `resolved` verdict (nil = server default).
        public var minMatchScore: Double?
        /// Try a careful OCR pass on unresolved/ambiguous objects (once each).
        public var fineReadEnabled = true
        /// Let the on-device model adjudicate an ambiguous shortlist (once each).
        public var adjudicateAmbiguous = true
        public var tracker = ObjectTracker.Config()
        public init() {}
    }

    /// Which step produced a resolution — surfaced in telemetry and, subtly, in the HUD.
    public enum Stage: String, Sendable {
        case barcode, coarse, fineOCR = "fine_ocr", llmPick = "llm_pick", user
    }

    public enum ObjectStatus: Sendable {
        case tracking                              // seen; not enough evidence yet
        case resolving                             // a query is in flight
        case resolved(ScoredCandidate, Stage)
        case ambiguous([ScoredCandidate])          // shortlist; user may pick
        case unresolved

        public var candidate: ScoredCandidate? {
            if case .resolved(let c, _) = self { return c }
            return nil
        }
        public var isFinal: Bool {
            switch self {
            case .resolved, .ambiguous, .unresolved: true
            case .tracking, .resolving: false
            }
        }
    }

    /// One object as the HUD should draw it. `box` is in the engine's content space (see
    /// `ScanEngine.contentAspect`); `anchored` is false for box-less mock detections.
    public struct SceneObject: Identifiable, Sendable {
        public let id: String
        public var box: BoundingBox
        public var anchored: Bool
        public var label: String?
        public var texts: [String]
        public var framesSeen: Int
        public var status: ObjectStatus
    }

    @Published public private(set) var objects: [SceneObject] = []
    /// Every resolved candidate in view, best predicted enjoyment first (feeds the chat
    /// bar's rerank and any list UI). Two cans of the same beer appear twice.
    @Published public private(set) var candidates: [ScoredCandidate] = []
    @Published public private(set) var lastLatencyMs: Double?
    @Published public private(set) var isScanning = false
    /// Resolve calls made this session — mostly for tests and the debug readout.
    public private(set) var resolveCount = 0

    private let engine: ScanEngine
    private let api: APIClientProtocol
    private let llm: LLMProvider?
    private let telemetry: TelemetryQueue?
    private let policy: Policy
    private let tracker: ObjectTracker
    private var task: Task<Void, Never>?
    private var inFlight = false
    private var venueId: String?
    private var verdicts: [String: ObjectStatus] = [:]
    private var fineRead: Set<String> = []
    private var adjudicated: Set<String> = []

    public init(engine: ScanEngine, api: APIClientProtocol, llm: LLMProvider? = nil,
                telemetry: TelemetryQueue? = nil, policy: Policy = Policy()) {
        self.engine = engine
        self.api = api
        self.llm = llm
        self.telemetry = telemetry
        self.policy = policy
        self.tracker = ObjectTracker(config: policy.tracker)
    }

    public func start(venueId: String? = nil) {
        guard !isScanning else { return }
        isScanning = true
        self.venueId = venueId
        task = Task { [weak self] in
            guard let self else { return }
            await self.engine.start()
            for await frame in self.engine.frames {
                self.handle(frame)
            }
        }
    }

    public func stop() {
        engine.stop()
        task?.cancel()
        isScanning = false
    }

    /// Reset tracking state, e.g. when the user pans to a new shelf.
    public func resetView() {
        tracker.reset()
        verdicts.removeAll(); fineRead.removeAll(); adjudicated.removeAll()
        objects.removeAll(); candidates.removeAll()
    }

    /// The user picked (or corrected) what an object is. The highest-value label we get.
    public func confirm(objectId: String, candidate: ScoredCandidate) {
        let shown = verdicts[objectId]?.candidate?.resolved.product.id
        verdicts[objectId] = .resolved(candidate, .user)
        publish()
        let texts = objects.first { $0.id == objectId }?.texts ?? []
        Task {
            await telemetry?.log("scan_corrected_by_user", tier: .personalization, [
                "shown_product_id": .string(shown ?? ""),
                "corrected_product_id": .string(candidate.resolved.product.id),
                "raw_text": .string(texts.joined(separator: " | ")),
            ])
        }
    }

    // MARK: - frame handling

    private func handle(_ frame: ScanFrame) {
        tracker.update(with: frame)
        prune()
        publish()
        queryIfNeeded()
    }

    private func queryIfNeeded() {
        guard !inFlight else { return }
        let ready = tracker.tracks.filter { tracker.isReady($0) && !isBarcodeFinal($0.id) }
        guard !ready.isEmpty else { return }
        let objs = ready.map { $0.detectedObject(minCount: policy.tracker.minTextCount) }
        for t in ready {
            tracker.markQueried(t.id)
            verdicts[t.id] = .resolving
        }
        publish()
        inFlight = true
        Task { [weak self] in
            guard let self else { return }
            await self.resolve(objs)
            self.inFlight = false
            self.publish()
            // Fine reads may have added evidence while we were busy — go again.
            self.queryIfNeeded()
        }
    }

    private func isBarcodeFinal(_ id: String) -> Bool {
        if case .resolved(_, .barcode)? = verdicts[id] { return true }
        if case .resolved(_, .user)? = verdicts[id] { return true }
        return false
    }

    private func resolve(_ objs: [DetectedObject]) async {
        resolveCount += 1
        let req = ScanResolveRequest(objects: objs, venueId: venueId, includeScore: true,
                                     minMatchScore: policy.minMatchScore)
        let resp: ScanResolveResponse
        do {
            resp = try await api.resolveScan(req)
        } catch {
            // Network hiccup: let these objects retry on a later frame.
            for o in objs { verdicts[o.id] = nil }
            for t in tracker.tracks where objs.contains(where: { $0.id == t.id }) {
                tracker.unmarkQueried(t.id)
            }
            return
        }
        lastLatencyMs = resp.latencyMs
        let byId = Dictionary(resp.objects.map { ($0.objectId, $0) }, uniquingKeysWith: { a, _ in a })
        for o in objs {
            if let res = byId[o.id] { await apply(res, to: o) }
            else { verdicts[o.id] = .unresolved }
        }
        await telemetry?.log("scan_frame_batch", tier: .personalization, [
            "n_detections": .int(objs.count),
            "n_resolved": .int(resp.objects.filter { $0.status == .resolved }.count),
            "server_latency_ms": .double(resp.latencyMs ?? 0),
            "ocr_strings": .stringList(objs.flatMap { $0.texts }),
            "barcode_values": .stringList(objs.compactMap { $0.barcode }),
        ])
    }

    private func apply(_ res: ObjectResolution, to obj: DetectedObject) async {
        let id = obj.id
        switch res.status {
        case .resolved:
            guard let top = res.candidates.first else { verdicts[id] = .unresolved; return }
            let stage: Stage = obj.barcode != nil ? .barcode : (fineRead.contains(id) ? .fineOCR : .coarse)
            verdicts[id] = .resolved(top, stage)
            await logOutcome(id, status: "resolved", stage: stage, score: top.matchScore,
                             n: res.candidates.count, obj: obj)

        case .ambiguous, .unresolved:
            // Fine stage 1: read the object again, carefully, once. New text re-queries.
            if policy.fineReadEnabled, !fineRead.contains(id),
               let reader = engine as? FineTextReader, let box = obj.box {
                fineRead.insert(id)
                let known = Set(obj.texts.map(ObjectTracker.normalize))
                let extra = await reader.readText(in: box)
                    .filter { !known.contains(ObjectTracker.normalize($0)) }
                if !extra.isEmpty {
                    tracker.addTexts(extra, to: id)
                    verdicts[id] = .tracking
                    return
                }
            }
            // Fine stage 2: constrained pick among the server's shortlist, once.
            if res.status == .ambiguous, policy.adjudicateAmbiguous, let llm,
               !adjudicated.contains(id) {
                adjudicated.insert(id)
                if let pick = try? await llm.pickProduct(ocr: obj.texts, candidates: res.candidates),
                   let cand = res.candidates.first(where: { $0.resolved.product.id == pick.productId }) {
                    verdicts[id] = .resolved(cand, .llmPick)
                    await logOutcome(id, status: "resolved", stage: .llmPick,
                                     score: pick.confidence, n: res.candidates.count, obj: obj)
                    return
                }
            }
            verdicts[id] = res.status == .ambiguous ? .ambiguous(res.candidates) : .unresolved
            await logOutcome(id, status: res.status.rawValue,
                             stage: fineRead.contains(id) ? .fineOCR : .coarse,
                             score: res.candidates.first?.matchScore,
                             n: res.candidates.count, obj: obj)
        }
    }

    private func logOutcome(_ id: String, status: String, stage: Stage, score: Double?,
                            n: Int, obj: DetectedObject) async {
        var props: [String: TelemetryValue] = [
            "status": .string(status), "stage": .string(stage.rawValue),
            "n_candidates": .int(n), "frames_seen": .int(obj.framesSeen),
            "n_texts": .int(obj.texts.count),
        ]
        if let score { props["match_score"] = .double(score) }
        await telemetry?.log("scan_object_resolution", tier: .analytics, props)
    }

    // MARK: - publishing

    private func prune() {
        let live = Set(tracker.tracks.map(\.id))
        verdicts = verdicts.filter { live.contains($0.key) }
        fineRead = fineRead.intersection(live)
        adjudicated = adjudicated.intersection(live)
    }

    private func publish() {
        let minCount = policy.tracker.minTextCount
        objects = tracker.tracks.map { t in
            SceneObject(id: t.id, box: t.box, anchored: t.anchored, label: t.label,
                        texts: t.stableTexts(minCount: minCount), framesSeen: t.framesSeen,
                        status: verdicts[t.id] ?? .tracking)
        }
        candidates = objects.compactMap { $0.status.candidate }.sorted {
            ($0.personalScore ?? $0.matchScore) > ($1.personalScore ?? $1.matchScore)
        }
    }
}
