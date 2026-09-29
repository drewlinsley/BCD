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

/// An aisle: Beer, Spirits, and whatever belongs to neither. Two levels rather than one because
/// twenty-two shelves in a flat list is a scroll, and because beer-or-spirits is a division the
/// drinker already made before they opened the screen.
public struct FamilyGroup: Codable, Sendable, Identifiable {
    public var id: String { group }
    /// Stable key (`beer`, `spirits`). Never shown.
    public let group: String
    public let label: String
    /// Whether any shelf in here is one they have rated on. What decides which aisle opens.
    public let ratedIn: Bool
    public let families: [FamilyPicks]

    enum CodingKeys: String, CodingKey {
        case group, label, families
        case ratedIn = "rated_in"
    }

    public init(group: String, label: String, ratedIn: Bool = false,
                families: [FamilyPicks] = []) {
        self.group = group
        self.label = label
        self.ratedIn = ratedIn
        self.families = families
    }
}

public struct FamilyResponse: Codable, Sendable {
    /// How many drinks this install has been judged on, as the server counts them. What decides
    /// whether any shelf can be `yours`.
    public let rated: Int
    public let groups: [FamilyGroup]

    public init(rated: Int = 0, groups: [FamilyGroup] = []) {
        self.rated = rated
        self.groups = groups
    }
}
