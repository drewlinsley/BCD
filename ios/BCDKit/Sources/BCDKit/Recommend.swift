import Foundation

/// One drink the server thinks this person would like, from `POST /v1/recommend`.
///
/// Not a `ResolvedProduct`: the ranker reads half a million rows and answers with the handful
/// of fields a list row needs, so the response is deliberately flat. Opening one means
/// fetching the product by name through the existing search route.
public struct Recommendation: Codable, Sendable, Identifiable, Equatable {
    public var id: String { productId }
    public let productId: String
    public let name: String
    /// Absent when the catalog row has no producer of its own — the registry files plenty.
    public let producer: String?
    /// Predicted enjoyment, 0-1. The list arrives already ordered; this is for showing, not
    /// for re-sorting (the server's order also weighs the evidence behind each vector, which
    /// the score alone does not carry).
    public let score: Double
    /// The one-line "why", written in the same band the score falls in ("matches your
    /// tropical preference", "some citrus, which you like").
    public let reason: String
    /// Scored from chemistry or a style prior rather than from reviews — the thing that lets
    /// a drink nobody has rated still be recommended.
    public let coldStart: Bool
    /// What stands behind the vector: `rated` by drinkers, `known` (a profile of this product),
    /// or `guessed` (its style's centroid).
    public let evidence: Evidence

    public enum Evidence: String, Codable, Sendable {
        case rated, known, guessed

        /// What to tell someone about where the answer came from.
        public var blurb: String {
            switch self {
            case .rated: return "Drinkers rated this"
            case .known: return "We know this one"
            case .guessed: return "Guessed from its style"
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case name, producer, score, reason, evidence
        case productId = "product_id"
        case coldStart = "cold_start"
    }

    public init(productId: String, name: String, producer: String? = nil, score: Double,
                reason: String, coldStart: Bool = false, evidence: Evidence = .guessed) {
        self.productId = productId
        self.name = name
        self.producer = producer
        self.score = score
        self.reason = reason
        self.coldStart = coldStart
        self.evidence = evidence
    }

    /// Tolerant of a server that adds a tier or drops the reason: an unknown `evidence`
    /// reads as a guess rather than failing the whole list, the same way the scan contract
    /// treats its enums.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        productId = try c.decode(String.self, forKey: .productId)
        name = try c.decode(String.self, forKey: .name)
        producer = try c.decodeIfPresent(String.self, forKey: .producer)
        score = try c.decodeIfPresent(Double.self, forKey: .score) ?? 0
        reason = try c.decodeIfPresent(String.self, forKey: .reason) ?? ""
        coldStart = try c.decodeIfPresent(Bool.self, forKey: .coldStart) ?? false
        let raw = try c.decodeIfPresent(String.self, forKey: .evidence) ?? ""
        evidence = Evidence(rawValue: raw) ?? .guessed
    }
}

public struct RecommendResponse: Codable, Sendable {
    public let userId: String
    public let results: [Recommendation]

    enum CodingKeys: String, CodingKey {
        case results
        case userId = "user_id"
    }

    public init(userId: String, results: [Recommendation]) {
        self.userId = userId
        self.results = results
    }
}
