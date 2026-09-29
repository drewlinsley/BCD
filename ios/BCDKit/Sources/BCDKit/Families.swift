import Foundation

// Discover, one shelf at a time.
//
// Gin recommendations, bourbon recommendations and vodka recommendations are three questions,
// and `/v1/recommend` can only answer whichever one the drinker's taste sits nearest. A profile
// built from two IPAs puts the whole nearest neighbourhood inside the IPA shelf, so gin never
// appears in that list -- not ranked low, absent.

/// One shelf: what it is called, what its order means, and what is on it.
public struct FamilyPicks: Codable, Sendable, Identifiable {
    public var id: String { family }
    /// Stable key (`gin`, `bourbon`), for identity and telemetry. Never shown.
    public let family: String
    /// What the drinker reads ("Tequila & mezcal").
    public let label: String
    public let basis: Basis
    public let results: [FamilyPick]

    /// What the order of this shelf means. The three are not interchangeable, which is the
    /// whole reason the server says which one it used.
    public enum Basis: String, Codable, Sendable {
        /// Ranked by this drinker's taste, which they earned by rating on this shelf.
        case yours
        /// Ranked by a taste learned on some other shelf, because they asked for that.
        case cross
        /// Not ranked for anyone: best-evidenced first, no score. Shown dark.
        case unrated
    }

    /// Whether anything here is a claim about the reader.
    public var isPersonal: Bool { basis != .unrated }
}

/// One drink on a shelf. `score` and `reason` are absent on an unrated shelf, and that absence
/// is the point -- a number there would read as a prediction about someone who has never rated
/// anything on this shelf, which is the one thing known for certain to be untrue.
public struct FamilyPick: Codable, Sendable, Identifiable, Equatable {
    public var id: String { productId }
    public let productId: String
    public let name: String
    public let producer: String?
    public let score: Double?
    public let reason: String?
    public let coldStart: Bool
    public let evidence: Recommendation.Evidence

    enum CodingKeys: String, CodingKey {
        case name, producer, score, reason, evidence
        case productId = "product_id"
        case coldStart = "cold_start"
    }

    public init(productId: String, name: String, producer: String? = nil, score: Double? = nil,
                reason: String? = nil, coldStart: Bool = false,
                evidence: Recommendation.Evidence = .guessed) {
        self.productId = productId
        self.name = name
        self.producer = producer
        self.score = score
        self.reason = reason
        self.coldStart = coldStart
        self.evidence = evidence
    }
}

public struct FamilyResponse: Codable, Sendable {
    /// How many drinks this install has been judged on, as the server counts them. What decides
    /// whether any shelf can be `yours`.
    public let rated: Int
    public let families: [FamilyPicks]

    public init(rated: Int = 0, families: [FamilyPicks] = []) {
        self.rated = rated
        self.families = families
    }
}
