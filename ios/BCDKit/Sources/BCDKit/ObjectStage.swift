import Foundation

/// The client half of the coarse-to-fine scan path, layered onto the live tick.
///
///   coarse   text frames (and the segmenter's regions, when the engine has one) →
///            `ObjectTracker` → objects with two frames of agreement behind their text
///   server   those objects ride along on the tick's `/v1/scan/resolve` call and come
///            back with a verdict each: resolved / ambiguous / unresolved
///   fine 1   not resolved? one careful OCR pass on that object's crop
///            (`FineTextReader`); anything new re-queries on the next tick
///   fine 2   still ambiguous? the on-device model picks among the server's shortlist
///            (`LLMProvider.pickProduct`) — it cannot invent an answer
///   else     the object stays unresolved and the HUD shows *no name* for it
///
/// The per-line tick is unchanged and still decides what a frame proves; this adds the
/// object as the unit of a query and the verdict as the unit of an answer. A resolved
/// object's overlay is pinned to the object's box and moves with it, for as long as the
/// tracker can see the object — instead of to the line that happened to match this tick.
@MainActor
public final class ObjectStage {
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

    public enum Status: Sendable {
        case tracking                              // seen; not enough evidence yet
        case resolving                             // a query is in flight
        case resolved(ScoredCandidate, Stage)
        case ambiguous([ScoredCandidate])          // shortlist; nothing shown
        case unresolved

        public var candidate: ScoredCandidate? {
            if case .resolved(let c, _) = self { return c }
            return nil
        }
    }

    /// One object as the HUD may draw it. `box` is in the engine's content space (see
    /// `ScanEngine.contentAspect`); `anchored` is false for box-less mock detections.
    public struct SceneObject: Identifiable, Sendable {
        public let id: String
        public var box: BoundingBox
        public var anchored: Bool
        public var label: String?
        public var texts: [String]
        public var framesSeen: Int
        public var status: Status
    }

    public private(set) var objects: [SceneObject] = []
    /// Requests that carried objects, mostly for tests and the debug readout.
    public private(set) var queryCount = 0

    public let policy: Policy
    private let tracker: ObjectTracker
    private var verdicts: [String: Status] = [:]
    private var fineRead: Set<String> = []
    private var adjudicated: Set<String> = []

    public init(policy: Policy = Policy()) {
        self.policy = policy
        self.tracker = ObjectTracker(config: policy.tracker)
    }

    // MARK: - frames in

    /// Fold one frame into the tracker. `regions` is what the segmenter last saw, if the
    /// engine has one; without it, lines are grouped into labels by geometry.
    public func ingest(_ frame: [DetectedText], regions: [ObjectRegion]? = nil) {
        tracker.update(with: ScanFrame(texts: frame, regions: regions))
        prune()
        publish()
    }

    /// Forget everything — a new shelf, or the scan stopping.
    public func reset() {
        tracker.reset()
        verdicts.removeAll(); fineRead.removeAll(); adjudicated.removeAll()
        publish()
    }

    // MARK: - queries out

    /// The objects with enough evidence to ask about and something new to ask. Marks them
    /// in flight; call `retry` if the request never reaches the server.
    public func pending() -> [DetectedObject] {
        let ready = tracker.tracks.filter { tracker.isReady($0) && !isFinal($0.id) }
        guard !ready.isEmpty else { return [] }
        let objs = ready.map { $0.detectedObject(minCount: policy.tracker.minTextCount) }
        for t in ready {
            tracker.markQueried(t.id)
            verdicts[t.id] = .resolving
        }
        queryCount += 1
        publish()
        return objs
    }

    /// A signature of what `pending()` would send, so the tick's held-still check can
    /// tell "same frame, nothing new to ask" from "same frame, an object just got ready".
    public var pendingSignature: String {
        tracker.tracks.filter { tracker.isReady($0) && !isFinal($0.id) }
            .map { $0.id + ":" + $0.signature(minCount: policy.tracker.minTextCount) }
            .sorted().joined(separator: "|")
    }

    /// The request failed before the server judged it: let the same evidence ask again.
    public func retry(_ objs: [DetectedObject]) {
        for o in objs {
            verdicts[o.id] = nil
            tracker.unmarkQueried(o.id)
        }
        publish()
    }

    private func isFinal(_ id: String) -> Bool {
        if case .resolved(_, .barcode)? = verdicts[id] { return true }
        if case .resolved(_, .user)? = verdicts[id] { return true }
        return false
    }

    // MARK: - verdicts in

    /// Apply the server's verdicts for `sent`, running the fine stage where it earns its
    /// keep. Returns true when a fine read added evidence and the object wants re-querying.
    @discardableResult
    public func apply(_ resolutions: [ObjectResolution], sent: [DetectedObject],
                      fineReader: FineTextReader?, llm: LLMProvider?,
                      telemetry: TelemetryQueue?) async -> Bool {
        let byId = Dictionary(resolutions.map { ($0.objectId, $0) },
                              uniquingKeysWith: { a, _ in a })
        var wantsRequery = false
        for o in sent {
            guard verdicts[o.id] != nil else { continue }       // pruned while in flight
            if let res = byId[o.id] {
                wantsRequery = await apply(res, to: o, fineReader: fineReader, llm: llm,
                                           telemetry: telemetry) || wantsRequery
            } else {
                verdicts[o.id] = .unresolved
            }
        }
        publish()
        return wantsRequery
    }

    private func apply(_ res: ObjectResolution, to obj: DetectedObject,
                       fineReader: FineTextReader?, llm: LLMProvider?,
                       telemetry: TelemetryQueue?) async -> Bool {
        let id = obj.id
        switch res.status {
        case .resolved:
            guard let top = res.candidates.first else { verdicts[id] = .unresolved; return false }
            let stage: Stage = obj.barcode != nil ? .barcode
                : (fineRead.contains(id) ? .fineOCR : .coarse)
            verdicts[id] = .resolved(top, stage)
            await log(telemetry, status: "resolved", stage: stage, score: top.matchScore,
                      n: res.candidates.count, obj: obj)
            return false

        case .ambiguous, .unresolved:
            // Fine stage 1: read the object again, carefully, once. New text re-queries.
            if policy.fineReadEnabled, !fineRead.contains(id),
               let fineReader, let box = obj.box {
                fineRead.insert(id)
                let known = Set(obj.texts.map(ObjectTracker.normalize))
                let extra = await fineReader.readText(in: box)
                    .filter { !known.contains(ObjectTracker.normalize($0)) }
                if !extra.isEmpty, verdicts[id] != nil {
                    tracker.addTexts(extra, to: id)
                    verdicts[id] = .tracking
                    return true
                }
            }
            // Fine stage 2: constrained pick among the server's shortlist, once.
            if res.status == .ambiguous, policy.adjudicateAmbiguous, let llm,
               !adjudicated.contains(id) {
                adjudicated.insert(id)
                if let pick = try? await llm.pickProduct(ocr: obj.texts, candidates: res.candidates),
                   let cand = res.candidates.first(where: { $0.resolved.product.id == pick.productId }),
                   verdicts[id] != nil {
                    verdicts[id] = .resolved(cand, .llmPick)
                    await log(telemetry, status: "resolved", stage: .llmPick,
                              score: pick.confidence, n: res.candidates.count, obj: obj)
                    return false
                }
            }
            guard verdicts[id] != nil else { return false }
            verdicts[id] = res.status == .ambiguous ? .ambiguous(res.candidates) : .unresolved
            await log(telemetry, status: res.status.rawValue,
                      stage: fineRead.contains(id) ? .fineOCR : .coarse,
                      score: res.candidates.first?.matchScore,
                      n: res.candidates.count, obj: obj)
            return false
        }
    }

    /// The user picked (or corrected) what an object is. The highest-value label we get.
    public func confirm(objectId: String, candidate: ScoredCandidate,
                        telemetry: TelemetryQueue?) async {
        let shown = verdicts[objectId]?.candidate?.resolved.product.id
        verdicts[objectId] = .resolved(candidate, .user)
        publish()
        let texts = objects.first { $0.id == objectId }?.texts ?? []
        await telemetry?.log("scan_corrected_by_user", tier: .personalization, [
            "shown_product_id": .string(shown ?? ""),
            "corrected_product_id": .string(candidate.resolved.product.id),
            "raw_text": .string(texts.joined(separator: " | ")),
        ])
    }

    private func log(_ telemetry: TelemetryQueue?, status: String, stage: Stage,
                     score: Double?, n: Int, obj: DetectedObject) async {
        var props: [String: TelemetryValue] = [
            "status": .string(status), "stage": .string(stage.rawValue),
            "n_candidates": .int(n), "frames_seen": .int(obj.framesSeen),
            "n_texts": .int(obj.texts.count),
        ]
        if let score { props["match_score"] = .double(score) }
        await telemetry?.log("scan_object_resolution", tier: .analytics, props)
    }

    // MARK: - out

    /// Overlays for the resolved objects, pinned to the centre of each object's box.
    /// Box-less objects (mock detections) have no place on screen and draw nothing here.
    public var overlays: [ResolvedOverlay] {
        objects.compactMap { o in
            guard o.anchored, let c = o.status.candidate else { return nil }
            return ResolvedOverlay(id: c.resolved.product.id, candidate: c,
                                   x: o.box.midX, y: o.box.midY)
        }
    }

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
    }
}
