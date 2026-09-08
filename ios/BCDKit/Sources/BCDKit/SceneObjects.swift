import Foundation

// The coarse stage's data model: turning per-frame text/barcode/region detections into
// *objects* that persist across frames and accumulate evidence.
//
//   frame → [ObjectRegion] (from a segmenter, optional) + [DetectedText]
//         → TextClusterer groups lines that sit on one label
//         → ObjectTracker matches clusters/regions to tracks across frames, counting how
//           many frames each text line was seen on each object
//         → a track is "ready" once its evidence is stable; only then is it queried.
//
// Everything here is pure Swift and runs on the host, so it is unit-tested without a
// camera. The reason it exists: a single frame's OCR is noisy, and a single text line is
// not a product. Waiting for two frames of agreement and querying per object (not per
// line) removes most of the wrong overlays before the server is even asked.

/// One object the coarse segmenter found in a frame (a can, a bottle, ...).
public struct ObjectRegion: Sendable, Equatable, Identifiable {
    public let id: String
    public let box: BoundingBox
    public let label: String?        // "can" | "bottle" | "object" | nil
    public let confidence: Double?

    public init(id: String = UUID().uuidString, box: BoundingBox, label: String? = nil,
                confidence: Double? = nil) {
        self.id = id; self.box = box; self.label = label; self.confidence = confidence
    }
}

/// Everything an engine saw in one frame.
public struct ScanFrame: Sendable {
    public let texts: [DetectedText]       // text lines and barcodes
    public let regions: [ObjectRegion]?    // nil when the engine has no segmenter
    public let timestamp: TimeInterval

    public init(texts: [DetectedText], regions: [ObjectRegion]? = nil,
                timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.texts = texts; self.regions = regions; self.timestamp = timestamp
    }
}

// MARK: - Text clustering

/// Groups OCR lines that sit on the same physical label. Two lines join when they overlap
/// horizontally and the vertical gap between them is small relative to their height —
/// i.e. they read like consecutive lines of one label rather than neighbouring cans.
public enum TextClusterer {
    public struct Cluster: Sendable, Equatable {
        public let box: BoundingBox?      // nil when the members carry no geometry
        public let members: [DetectedText]
    }

    public struct Params: Sendable {
        /// Horizontal overlap as a fraction of the narrower line's width.
        public var minHorizontalOverlap = 0.3
        /// Allowed vertical gap as a multiple of the taller line's height...
        public var maxVerticalGapFactor = 1.5
        /// ...with an absolute floor (fraction of the frame) so tiny lines still join.
        public var maxVerticalGapAbs = 0.04
        public init() {}
    }

    public static func cluster(_ texts: [DetectedText], params: Params = Params()) -> [Cluster] {
        var withBox: [(Int, BoundingBox)] = []
        var clusters: [Cluster] = []
        for (i, t) in texts.enumerated() {
            if let b = t.box { withBox.append((i, b)) }
            else { clusters.append(Cluster(box: nil, members: [t])) }
        }
        guard !withBox.isEmpty else { return clusters }

        var parent = Array(0..<withBox.count)
        func find(_ i: Int) -> Int {
            var i = i
            while parent[i] != i { parent[i] = parent[parent[i]]; i = parent[i] }
            return i
        }
        func union(_ a: Int, _ b: Int) { parent[find(a)] = find(b) }

        for a in 0..<withBox.count {
            for b in (a + 1)..<withBox.count where related(withBox[a].1, withBox[b].1, params) {
                union(a, b)
            }
        }
        var groups: [Int: [Int]] = [:]
        for i in 0..<withBox.count { groups[find(i), default: []].append(i) }
        for (_, idxs) in groups.sorted(by: { $0.key < $1.key }) {
            let members = idxs.map { texts[withBox[$0].0] }
            let box = BoundingBox.enclosing(idxs.map { withBox[$0].1 })
            clusters.append(Cluster(box: box, members: members))
        }
        return clusters
    }

    static func related(_ a: BoundingBox, _ b: BoundingBox, _ p: Params) -> Bool {
        let overlapW = min(a.maxX, b.maxX) - max(a.minX, b.minX)
        let narrower = min(a.w, b.w)
        guard overlapW > 0, narrower > 0, overlapW / narrower >= p.minHorizontalOverlap else {
            return false
        }
        let gap = max(0, max(a.minY, b.minY) - min(a.maxY, b.maxY))
        let tolerance = max(p.maxVerticalGapFactor * max(a.h, b.h), p.maxVerticalGapAbs)
        return gap <= tolerance
    }
}

// MARK: - Tracking across frames

/// Matches per-frame observations to persistent tracks and accumulates evidence on them.
/// Not thread-safe by design: owned and driven by the `ScanCoordinator` on the main actor.
public final class ObjectTracker {
    public struct Config: Sendable {
        /// Boxes overlapping at least this much are the same object.
        public var minIoU = 0.25
        /// ...or whose centers are within this distance (fraction of the frame), for
        /// small text clusters that jitter frame to frame.
        public var maxCenterDistance = 0.06
        /// Frames an object may go unseen before it is dropped (pans, hands, glare).
        public var maxMissing = 12
        /// Frames of support before an object may be queried at all.
        public var minFramesStable = 2
        /// Frames a text line must be seen on an object before it counts as evidence.
        public var minTextCount = 2
        /// Box smoothing toward each new observation (1 = jump, 0 = never move).
        public var boxSmoothing = 0.5
        /// Frames to wait after a query before new text may trigger another.
        public var requeryCooldown = 6
        public init() {}
    }

    public struct Track: Sendable, Identifiable {
        public let id: String
        public internal(set) var box: BoundingBox
        /// false when the track came from box-less text (tests/mocks) and matches by text.
        public internal(set) var anchored: Bool
        /// true when the box came from a segmenter region rather than a text cluster.
        public internal(set) var fromRegion: Bool
        public internal(set) var label: String?
        public internal(set) var confidence: Double?
        public internal(set) var textCounts: [String: Int] = [:]     // normalized -> frames
        public internal(set) var textOriginal: [String: String] = [:] // normalized -> as read
        public internal(set) var pinnedTexts: Set<String> = []        // fine-reader results
        public internal(set) var barcode: String?
        public internal(set) var symbology: String?
        public internal(set) var framesSeen = 0
        public internal(set) var framesMissing = 0
        public internal(set) var lastQueriedFrame: Int?
        public internal(set) var lastQueriedSignature: String?
        public internal(set) var pinnedSinceQuery = false

        /// Text lines with enough temporal support, most-seen first.
        public func stableTexts(minCount: Int) -> [String] {
            textCounts
                .filter { $0.value >= minCount || pinnedTexts.contains($0.key) }
                .sorted { ($0.value, $1.key) > ($1.value, $0.key) }
                .compactMap { textOriginal[$0.key] }
        }

        /// What the server would be asked. Changes only when evidence changes.
        public func signature(minCount: Int) -> String {
            let keys = textCounts
                .filter { $0.value >= minCount || pinnedTexts.contains($0.key) }
                .map(\.key).sorted()
            return (keys + [barcode ?? ""]).joined(separator: "|")
        }

        public func detectedObject(minCount: Int) -> DetectedObject {
            DetectedObject(id: id, label: label, texts: stableTexts(minCount: minCount),
                           barcode: barcode, symbology: symbology,
                           box: anchored ? box : nil, framesSeen: framesSeen,
                           confidence: confidence)
        }
    }

    struct Observation {
        var box: BoundingBox?
        var fromRegion: Bool
        var label: String?
        var confidence: Double?
        var texts: [DetectedText]
    }

    public let config: Config
    public private(set) var tracks: [Track] = []
    public private(set) var frameCount = 0

    public init(config: Config = Config()) { self.config = config }

    public func reset() { tracks.removeAll(); frameCount = 0 }

    /// Ingest one frame; returns every live track (matched or briefly missing).
    @discardableResult
    public func update(with frame: ScanFrame) -> [Track] {
        frameCount += 1
        let observations = Self.observations(from: frame)

        // Greedy assignment, best overlap first.
        var pairs: [(score: Double, obs: Int, track: Int)] = []
        for (oi, o) in observations.enumerated() {
            for (ti, t) in tracks.enumerated() {
                let s = affinity(o, t)
                if s > 0 { pairs.append((s, oi, ti)) }
            }
        }
        pairs.sort { $0.score > $1.score }
        var usedObs = Set<Int>(), usedTracks = Set<Int>()
        var matched: [(Int, Int)] = []
        for p in pairs where !usedObs.contains(p.obs) && !usedTracks.contains(p.track) {
            usedObs.insert(p.obs); usedTracks.insert(p.track)
            matched.append((p.obs, p.track))
        }

        for (oi, ti) in matched { absorb(observations[oi], into: &tracks[ti]) }
        for ti in tracks.indices where !usedTracks.contains(ti) { tracks[ti].framesMissing += 1 }
        for (oi, o) in observations.enumerated() where !usedObs.contains(oi) {
            var t = Track(id: String(UUID().uuidString.prefix(8)),
                          box: o.box ?? .unit, anchored: o.box != nil,
                          fromRegion: o.fromRegion, label: o.label, confidence: o.confidence)
            absorb(o, into: &t)
            tracks.append(t)
        }
        tracks.removeAll { $0.framesMissing > config.maxMissing }
        return tracks
    }

    /// Fold in text from the fine reader. Pinned lines count as evidence immediately.
    public func addTexts(_ texts: [String], to id: String) {
        guard let i = tracks.firstIndex(where: { $0.id == id }) else { return }
        for raw in texts {
            let key = Self.normalize(raw)
            guard !key.isEmpty else { continue }
            tracks[i].textCounts[key, default: 0] += 1
            tracks[i].textOriginal[key] = tracks[i].textOriginal[key] ?? raw
            tracks[i].pinnedTexts.insert(key)
            tracks[i].pinnedSinceQuery = true
        }
    }

    public func markQueried(_ id: String) {
        guard let i = tracks.firstIndex(where: { $0.id == id }) else { return }
        tracks[i].lastQueriedFrame = frameCount
        tracks[i].lastQueriedSignature = tracks[i].signature(minCount: config.minTextCount)
        tracks[i].pinnedSinceQuery = false
    }

    /// Undo `markQueried` after a failed request so the same evidence retries later.
    public func unmarkQueried(_ id: String) {
        guard let i = tracks.firstIndex(where: { $0.id == id }) else { return }
        tracks[i].lastQueriedFrame = nil
        tracks[i].lastQueriedSignature = nil
    }

    /// Enough support to ask the server, and something new to ask about.
    public func isReady(_ t: Track) -> Bool {
        guard t.framesMissing == 0 else { return false }
        let sig = t.signature(minCount: config.minTextCount)
        let hasEvidence = t.barcode != nil || !t.stableTexts(minCount: config.minTextCount).isEmpty
        guard hasEvidence, t.framesSeen >= config.minFramesStable else { return false }
        guard sig != t.lastQueriedSignature else { return false }
        // A re-query (evidence grew since the last one) waits out a cooldown so a label
        // whose lines cross the support threshold one frame apart doesn't fire a burst of
        // requests — unless the new evidence is decisive: a barcode, or fine-reader text.
        if let last = t.lastQueriedFrame, frameCount - last < config.requeryCooldown {
            let barcodeIsNew = t.barcode.map { !(t.lastQueriedSignature ?? "").hasSuffix("|" + $0) } ?? false
            return barcodeIsNew || t.pinnedSinceQuery
        }
        return true
    }

    // MARK: internals

    static func normalize(_ s: String) -> String {
        s.uppercased()
            .split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .joined(separator: " ")
    }

    static func observations(from frame: ScanFrame) -> [Observation] {
        var out: [Observation] = []
        var leftovers: [DetectedText] = []
        if let regions = frame.regions, !regions.isEmpty {
            out = regions.map {
                Observation(box: $0.box, fromRegion: true, label: $0.label,
                            confidence: $0.confidence, texts: [])
            }
            let regionBoxes = regions.map(\.box)
            for t in frame.texts {
                if let b = t.box,
                   let ri = regionBoxes.indices.max(by: { overlap(b, regionBoxes[$0]) < overlap(b, regionBoxes[$1]) }),
                   overlap(b, regionBoxes[ri]) > 0 {
                    out[ri].texts.append(t)
                } else {
                    leftovers.append(t)
                }
            }
        } else {
            leftovers = frame.texts
        }
        for c in TextClusterer.cluster(leftovers) {
            out.append(Observation(box: c.box, fromRegion: false, label: nil, confidence: nil,
                                   texts: c.members))
        }
        return out
    }

    /// Fraction of `text` inside `region` — so a line straddling two cans goes to the
    /// one holding more of it.
    static func overlap(_ text: BoundingBox, _ region: BoundingBox) -> Double {
        guard let i = text.intersection(region), text.area > 0 else {
            return region.contains(x: text.midX, y: text.midY) ? 1 : 0
        }
        return i.area / text.area
    }

    func affinity(_ o: Observation, _ t: Track) -> Double {
        guard let ob = o.box, t.anchored else {
            // No geometry on one side: same object iff they share a text line.
            let keys = Set(o.texts.map { Self.normalize($0.text) })
            return keys.isDisjoint(with: t.textCounts.keys) ? 0 : 0.5
        }
        let iou = ob.iou(t.box)
        if iou >= config.minIoU { return 1 + iou }
        // A text cluster sitting inside a region-backed track belongs to it.
        if t.fromRegion && !o.fromRegion && t.box.contains(x: ob.midX, y: ob.midY) { return 1 }
        if ob.distance(to: t.box) <= config.maxCenterDistance { return 0.5 + iou }
        return 0
    }

    func absorb(_ o: Observation, into t: inout Track) {
        t.framesSeen += 1
        t.framesMissing = 0
        if let ob = o.box {
            if o.fromRegion || !t.fromRegion {
                t.box = t.anchored ? t.box.blended(toward: ob, config.boxSmoothing) : ob
                t.anchored = true
                t.fromRegion = t.fromRegion || o.fromRegion
            }
        }
        t.label = o.label ?? t.label
        t.confidence = o.confidence ?? t.confidence
        for d in o.texts {
            if d.kind == "barcode" {
                if !d.text.isEmpty { t.barcode = d.text; t.symbology = d.symbology }
                continue
            }
            let key = Self.normalize(d.text)
            guard !key.isEmpty else { continue }
            t.textCounts[key, default: 0] += 1
            t.textOriginal[key] = t.textOriginal[key] ?? d.text
        }
    }
}
