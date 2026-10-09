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

/// What `GET /v1/product/{id}/score` answers: how likely this drinker is to like one drink.
///
/// The detail screen is reachable by four doors and only one of them is the recommender, so a
/// drink found by name or opened from another drink's Similar profile arrived with nothing
/// personal attached and the screen's seal said "not scored for you yet" about a beer the
/// server would have called a 91% match. This is that question asked on its own.
///
/// `scored` is false rather than the score being zero, because they are different claims: no
/// profile yet, or no flavour vector on this row, is not a prediction of nought. The server
/// never answers from its seed profile here — a number stamped on a label has to be about the
/// person reading it.
public struct PersonalScore: Decodable, Sendable, Equatable {
    public let productId: String
    public let scored: Bool
    public let personalScore: Double?
    public let reason: String?
    public let coldStart: Bool
    public let evidence: Recommendation.Evidence?
    /// Why there is no score, when there is none: `no_profile` or `no_vector`. `yours` when
    /// there is one. Carried for diagnosis — the screen only reads `scored`.
    public let basis: String?

    enum CodingKeys: String, CodingKey {
        case scored, reason, evidence, basis
        case productId = "product_id"
        case personalScore = "personal_score"
        case coldStart = "cold_start"
    }

    public init(productId: String, scored: Bool, personalScore: Double? = nil,
                reason: String? = nil, coldStart: Bool = false,
                evidence: Recommendation.Evidence? = nil, basis: String? = nil) {
        self.productId = productId
        self.scored = scored
        self.personalScore = personalScore
        self.reason = reason
        self.coldStart = coldStart
        self.evidence = evidence
        self.basis = basis
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        productId = try c.decode(String.self, forKey: .productId)
        scored = try c.decodeIfPresent(Bool.self, forKey: .scored) ?? false
        personalScore = try c.decodeIfPresent(Double.self, forKey: .personalScore)
        reason = try c.decodeIfPresent(String.self, forKey: .reason)
        coldStart = try c.decodeIfPresent(Bool.self, forKey: .coldStart) ?? false
        // An evidence tier this build does not know reads as nil rather than failing the
        // call, the same tolerance `Recommendation` shows.
        evidence = (try? c.decodeIfPresent(Recommendation.Evidence.self, forKey: .evidence))
            ?? nil
        basis = try c.decodeIfPresent(String.self, forKey: .basis)
    }
}
