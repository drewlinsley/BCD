import Foundation

/// Client mirror of the server `TasteProfile` — what the app has learned about one drinker,
/// rebuilt server-side from their verdicts and served by `GET /v1/profile`.
///
/// Every field here is optional or empty-able, because a profile arrives long before it has
/// anything to say: a fresh install gets `version: 0` and nothing else, and the screen showing
/// it has to be able to tell that apart from a profile that simply failed to load.
public struct TasteProfile: Codable, Sendable {
    public let userId: String
    public var version: Int
    public var styleAffinities: [String: Double]
    /// The taste centroid the recommender aims at — the same 25 axes a product carries, so
    /// `SensoryAxis.note` says a person's leaning out loud exactly as it says a bottle's.
    public var sensoryIdeal: SensoryVector?
    public var abvBandMin: Double?
    public var abvBandMax: Double?
    public var noveltyAppetite: Double?
    public var memo: String?
    public var updatedAt: String?

    enum CodingKeys: String, CodingKey {
        case version, memo
        case userId = "user_id"
        case styleAffinities = "style_affinities"
        case sensoryIdeal = "sensory_ideal"
        case abvBandMin = "abv_band_min"
        case abvBandMax = "abv_band_max"
        case noveltyAppetite = "novelty_appetite"
        case updatedAt = "updated_at"
    }

    public init(userId: String, version: Int = 0, styleAffinities: [String: Double] = [:],
                sensoryIdeal: SensoryVector? = nil, abvBandMin: Double? = nil,
                abvBandMax: Double? = nil, noveltyAppetite: Double? = nil,
                memo: String? = nil, updatedAt: String? = nil) {
        self.userId = userId
        self.version = version
        self.styleAffinities = styleAffinities
        self.sensoryIdeal = sensoryIdeal
        self.abvBandMin = abvBandMin
        self.abvBandMax = abvBandMax
        self.noveltyAppetite = noveltyAppetite
        self.memo = memo
        self.updatedAt = updatedAt
    }

    /// Decoded field by field so a missing key is an absent fact rather than a failed screen.
    /// The server fills `style_affinities` today; an older build of it, or a future one that
    /// drops a field, should still leave the drinker looking at their memo.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        userId = try c.decodeIfPresent(String.self, forKey: .userId) ?? ""
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 0
        styleAffinities = try c.decodeIfPresent([String: Double].self,
                                                forKey: .styleAffinities) ?? [:]
        sensoryIdeal = try c.decodeIfPresent(SensoryVector.self, forKey: .sensoryIdeal)
        abvBandMin = try c.decodeIfPresent(Double.self, forKey: .abvBandMin)
        abvBandMax = try c.decodeIfPresent(Double.self, forKey: .abvBandMax)
        noveltyAppetite = try c.decodeIfPresent(Double.self, forKey: .noveltyAppetite)
        memo = try c.decodeIfPresent(String.self, forKey: .memo)
        updatedAt = try c.decodeIfPresent(String.self, forKey: .updatedAt)
    }
}

public extension TasteProfile {
    /// A style the drinker's verdicts pushed *toward*, strongest first. Ties break on the name
    /// so two styles of equal weight do not swap places between launches.
    var stylesLiked: [(style: String, weight: Double)] {
        styleAffinities.filter { $0.value > 0 }
            .map { (style: $0.key, weight: $0.value) }
            .sorted { $0.weight == $1.weight ? $0.style < $1.style : $0.weight > $1.weight }
    }

    /// A style the verdicts pushed *away from*. Affinities run [-1, 1] and the negative half is
    /// as earned as the positive one — a drinker who spat something out said as much as one
    /// who chugged it, and a profile that only ever shows likes throws half of that away.
    var stylesAvoided: [(style: String, weight: Double)] {
        styleAffinities.filter { $0.value < 0 }
            .map { (style: $0.key, weight: $0.value) }
            .sorted { $0.weight == $1.weight ? $0.style < $1.style : $0.weight < $1.weight }
    }

    /// The flavour axes this drinker leans into, strongest first — structure axes excluded,
    /// since "body 0.33" is a property of a drink and not a thing anyone says they like.
    var notes: [(axis: SensoryAxis, value: Double)] {
        (sensoryIdeal?.ranked ?? []).filter { !$0.axis.isStructure && $0.value > 0 }
    }

    /// Nothing has been learned yet. Distinct from a profile that failed to load: this one
    /// arrived and is genuinely blank, which is what a fresh install looks like and what the
    /// screen must say plainly instead of dressing up a seed as the drinker's own.
    var hasLearnedNothing: Bool {
        (memo?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            && styleAffinities.isEmpty
            && notes.isEmpty
    }
}

public struct WeeklyPrediction: Codable, Sendable, Identifiable {
    public var id: String { text }
    public let text: String
    public let kind: String
    public let confidence: Double
    public var resolved: Bool?

    public init(text: String, kind: String, confidence: Double, resolved: Bool? = nil) {
        self.text = text
        self.kind = kind
        self.confidence = confidence
        self.resolved = resolved
    }
}

public struct WeeklyProfileDelta: Codable, Sendable {
    public let userId: String
    public let fromVersion: Int
    public let toVersion: Int
    public let summary: String
    public let predictions: [WeeklyPrediction]

    public init(userId: String, fromVersion: Int, toVersion: Int,
                summary: String, predictions: [WeeklyPrediction]) {
        self.userId = userId
        self.fromVersion = fromVersion
        self.toVersion = toVersion
        self.summary = summary
        self.predictions = predictions
    }

    enum CodingKeys: String, CodingKey {
        case summary, predictions
        case userId = "user_id"
        case fromVersion = "from_version"
        case toVersion = "to_version"
    }
}
