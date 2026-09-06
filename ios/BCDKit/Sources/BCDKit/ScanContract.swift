import Foundation

// Wire shapes for the scan path. Mirror bcd_schema.api.

public struct DetectedText: Codable, Sendable, Identifiable {
    public var id: String { "\(kind):\(text):\(x ?? 0):\(y ?? 0)" }
    public let text: String
    public let kind: String        // "text" | "barcode"
    public let symbology: String?
    // normalized bounding box 0-1, so the HUD can anchor the overlay in the camera frame
    public let x: Double?
    public let y: Double?
    public let w: Double?
    public let h: Double?
    public let confidence: Double?

    public init(text: String, kind: String = "text", symbology: String? = nil,
                x: Double? = nil, y: Double? = nil, w: Double? = nil, h: Double? = nil,
                confidence: Double? = nil) {
        self.text = text; self.kind = kind; self.symbology = symbology
        self.x = x; self.y = y; self.w = w; self.h = h; self.confidence = confidence
    }
}

public struct ScanResolveRequest: Codable, Sendable {
    public let detections: [DetectedText]
    public let venueId: String?
    public let lat: Double?
    public let lon: Double?
    public let includeScore: Bool

    enum CodingKeys: String, CodingKey {
        case detections, lat, lon
        case venueId = "venue_id"
        case includeScore = "include_score"
    }

    public init(detections: [DetectedText], venueId: String? = nil,
                lat: Double? = nil, lon: Double? = nil, includeScore: Bool = true) {
        self.detections = detections; self.venueId = venueId
        self.lat = lat; self.lon = lon; self.includeScore = includeScore
    }
}

public struct ScoredCandidate: Codable, Sendable, Identifiable {
    public var id: String { "\(detectionIndex):\(resolved.id)" }
    public let detectionIndex: Int
    public let resolved: ResolvedProduct
    public let matchScore: Double
    public let personalScore: Double?
    public let reason: String?
    public let coldStart: Bool

    enum CodingKeys: String, CodingKey {
        case resolved, reason
        case detectionIndex = "detection_index"
        case matchScore = "match_score"
        case personalScore = "personal_score"
        case coldStart = "cold_start"
    }
}

public struct ScanResolveResponse: Codable, Sendable {
    public let candidates: [ScoredCandidate]
    public let unresolvedIndices: [Int]
    public let latencyMs: Double?
    /// Whether more than one part of the frame agreed on some candidate — the label naming both
    /// its maker and its drink, or naming one and printing a category that matches it. False
    /// means the server returned a guess off a single fragment, which looks identical to a
    /// confident answer once it is an overlay.
    public let corroborated: Bool

    public init(candidates: [ScoredCandidate], unresolvedIndices: [Int],
                latencyMs: Double?, corroborated: Bool = false) {
        self.candidates = candidates
        self.unresolvedIndices = unresolvedIndices
        self.latencyMs = latencyMs
        self.corroborated = corroborated
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        candidates = try c.decode([ScoredCandidate].self, forKey: .candidates)
        unresolvedIndices = try c.decodeIfPresent([Int].self, forKey: .unresolvedIndices) ?? []
        latencyMs = try c.decodeIfPresent(Double.self, forKey: .latencyMs)
        // Absent from an older server: read as "not corroborated", so a missing field makes the
        // fallback run more rather than silently switching it off.
        corroborated = try c.decodeIfPresent(Bool.self, forKey: .corroborated) ?? false
    }

    enum CodingKeys: String, CodingKey {
        case candidates
        case unresolvedIndices = "unresolved_indices"
        case latencyMs = "latency_ms"
        case corroborated
    }
}

/// One camera frame, for the labels OCR cannot read.
///
/// `detections` is what the on-device scanner *did* read of the same frame. It does not enter
/// the matching — a clean reading needs no corroboration from a garbled one — but it is the
/// only record of what the camera saw at the moment the picture was taken, and every diagnosis
/// on this path so far has come from having exactly that.
public struct ScanVisionRequest: Codable, Sendable {
    public let imageB64: String
    public let mediaType: String
    public let detections: [DetectedText]
    public let venueId: String?

    enum CodingKeys: String, CodingKey {
        case detections
        case imageB64 = "image_b64"
        case mediaType = "media_type"
        case venueId = "venue_id"
    }

    public init(imageB64: String, mediaType: String = "image/jpeg",
                detections: [DetectedText] = [], venueId: String? = nil) {
        self.imageB64 = imageB64; self.mediaType = mediaType
        self.detections = detections; self.venueId = venueId
    }
}

/// A resolve response, plus what the model claimed to see.
///
/// `sightings` is the honest middle of the pipeline. A name there with no candidate beside it
/// means the model read the can and the catalog does not have it — a different problem from the
/// model reading nothing, and candidates alone cannot tell the two apart. `detections` is the
/// frame the *server* built from those sightings; the client never had it, so `detectionIndex`
/// would address nothing without it coming back.
public struct ScanVisionResponse: Codable, Sendable {
    public let candidates: [ScoredCandidate]
    public let unresolvedIndices: [Int]
    public let latencyMs: Double?
    public let corroborated: Bool
    public let sightings: [String]
    public let detections: [DetectedText]
    public let provider: String?
    public let detail: String?

    enum CodingKeys: String, CodingKey {
        case candidates, corroborated, sightings, detections, provider, detail
        case unresolvedIndices = "unresolved_indices"
        case latencyMs = "latency_ms"
    }

    public init(candidates: [ScoredCandidate] = [], unresolvedIndices: [Int] = [],
                latencyMs: Double? = nil, corroborated: Bool = false,
                sightings: [String] = [], detections: [DetectedText] = [],
                provider: String? = nil, detail: String? = nil) {
        self.candidates = candidates; self.unresolvedIndices = unresolvedIndices
        self.latencyMs = latencyMs; self.corroborated = corroborated
        self.sightings = sightings; self.detections = detections
        self.provider = provider; self.detail = detail
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        candidates = try c.decodeIfPresent([ScoredCandidate].self, forKey: .candidates) ?? []
        unresolvedIndices = try c.decodeIfPresent([Int].self, forKey: .unresolvedIndices) ?? []
        latencyMs = try c.decodeIfPresent(Double.self, forKey: .latencyMs)
        corroborated = try c.decodeIfPresent(Bool.self, forKey: .corroborated) ?? false
        sightings = try c.decodeIfPresent([String].self, forKey: .sightings) ?? []
        detections = try c.decodeIfPresent([DetectedText].self, forKey: .detections) ?? []
        provider = try c.decodeIfPresent(String.self, forKey: .provider)
        detail = try c.decodeIfPresent(String.self, forKey: .detail)
    }
}
