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

    enum CodingKeys: String, CodingKey {
        case candidates
        case unresolvedIndices = "unresolved_indices"
        case latencyMs = "latency_ms"
    }
}
