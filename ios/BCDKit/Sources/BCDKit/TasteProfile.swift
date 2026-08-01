import Foundation

/// Client mirror of the server `TasteProfile`. Evolves weekly; the app shows the delta as
/// a "your week" card with falsifiable predictions.
public struct TasteProfile: Codable, Sendable {
    public let userId: String
    public var version: Int
    public var styleAffinities: [String: Double]
    public var abvBandMin: Double?
    public var abvBandMax: Double?
    public var noveltyAppetite: Double?
    public var memo: String?

    enum CodingKeys: String, CodingKey {
        case version, memo
        case userId = "user_id"
        case styleAffinities = "style_affinities"
        case abvBandMin = "abv_band_min"
        case abvBandMax = "abv_band_max"
        case noveltyAppetite = "novelty_appetite"
    }
}

public struct WeeklyPrediction: Codable, Sendable, Identifiable {
    public var id: String { text }
    public let text: String
    public let kind: String
    public let confidence: Double
    public var resolved: Bool?
}

public struct WeeklyProfileDelta: Codable, Sendable {
    public let userId: String
    public let fromVersion: Int
    public let toVersion: Int
    public let summary: String
    public let predictions: [WeeklyPrediction]

    enum CodingKeys: String, CodingKey {
        case summary, predictions
        case userId = "user_id"
        case fromVersion = "from_version"
        case toVersion = "to_version"
    }
}
