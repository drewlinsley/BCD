import Foundation

// The taste half of a product, and the sentence that says it out loud.
//
// The server has carried a 25-axis sensory profile on nearly every product for a while;
// this client simply dropped it on decode. It is the densest field in the catalog —
// present on 914 of 915 products, where `style` is present on 9 — so it is what the
// detail screen has to lean on when it wants to say more than a name and an ABV.

/// The descriptor axes, in the order `bcd_schema.sensory.SENSORY_AXES` declares them.
/// New axes are appended server-side, never reordered, and an axis this build has never
/// heard of is dropped on decode rather than failing the product.
public enum SensoryAxis: String, Codable, Sendable, CaseIterable {
    // aroma / flavour families
    case citrus, tropical
    case stoneFruit = "stone_fruit"
    case berry, floral, herbal
    case pineyResinous = "piney_resinous"
    case grassy
    case spicyPhenolic = "spicy_phenolic"
    case maltyBready = "malty_bready"
    case caramelToffee = "caramel_toffee"
    case roastedCoffeeChoc = "roasted_coffee_choc"
    case nutty
    case vanillaOak = "vanilla_oak"
    case smokyPeat = "smoky_peat"
    case honey
    case bananaEster = "banana_ester"
    case funkBrett = "funk_brett"
    case sourTart = "sour_tart"
    case sweet
    // structure
    case bitterness
    case bodyFullness = "body_fullness"
    case carbonation
    case alcoholWarmth = "alcohol_warmth"
    case drynessFinish = "dryness_finish"

    /// Structure axes describe the shape of a drink, flavour axes name what's in it. They
    /// sit in different places in a sentence, so the split belongs to the type rather than
    /// to whoever is writing the copy.
    public var isStructure: Bool {
        switch self {
        case .bitterness, .bodyFullness, .carbonation, .alcoholWarmth, .drynessFinish:
            return true
        default:
            return false
        }
    }

    /// How the axis is said out loud. `malty_bready` is a column name; "bready malt" is
    /// what a person standing in a shop actually says.
    public var note: String {
        switch self {
        case .citrus: return "citrus"
        case .tropical: return "tropical fruit"
        case .stoneFruit: return "stone fruit"
        case .berry: return "berry"
        case .floral: return "floral"
        case .herbal: return "herbal"
        case .pineyResinous: return "pine"
        case .grassy: return "grassy hop"
        case .spicyPhenolic: return "peppery spice"
        case .maltyBready: return "bready malt"
        case .caramelToffee: return "caramel"
        case .roastedCoffeeChoc: return "roast and cocoa"
        case .nutty: return "nutty"
        case .vanillaOak: return "vanilla oak"
        case .smokyPeat: return "smoke and peat"
        case .honey: return "honey"
        case .bananaEster: return "banana"
        case .funkBrett: return "funk"
        case .sourTart: return "tartness"
        case .sweet: return "sweetness"
        // Structure axes are never named this way — they earn a clause in `TasteSummary`
        // instead — but a label keeps the enum total and useful in a chip or a debug view.
        case .bitterness: return "bitterness"
        case .bodyFullness: return "body"
        case .carbonation: return "carbonation"
        case .alcoholWarmth: return "warmth"
        case .drynessFinish: return "dry finish"
        }
    }
}

public enum SensorySource: String, Codable, Sendable {
    case chemistryPrior = "chemistry_prior"
    case reviewConsensus = "review_consensus"
    case reconciled
    case stylePrior = "style_prior"
}

public struct SensoryVector: Codable, Sendable {
    public let source: SensorySource
    public let confidence: Double
    public let axes: [String: Double]

    public init(source: SensorySource, confidence: Double = 0.5,
                axes: [String: Double] = [:]) {
        self.source = source
        self.confidence = confidence
        self.axes = axes
    }

    /// Axes this build understands, strongest first. Ties break on the axis name so the
    /// order is stable across launches — a taste note that reshuffles itself between two
    /// views of the same product reads as a bug.
    public var ranked: [(axis: SensoryAxis, value: Double)] {
        var known: [(axis: SensoryAxis, value: Double)] = []
        known.reserveCapacity(axes.count)
        for (key, value) in axes {
            guard let axis = SensoryAxis(rawValue: key) else { continue }
            known.append((axis: axis, value: value))
        }
        known.sort { lhs, rhs in
            lhs.value == rhs.value
                ? lhs.axis.rawValue < rhs.axis.rawValue
                : lhs.value > rhs.value
        }
        return known
    }

    public func value(_ axis: SensoryAxis) -> Double? { axes[axis.rawValue] }
}

/// Turns a sensory vector into the one or two sentences the detail screen shows.
///
/// Everything here is deliberately conservative. The vectors behind it average 0.32
/// confidence and most are style priors, so the copy names only what clears a threshold
/// and never reaches for a flourish the numbers don't support.
public enum TasteSummary {
    /// Flavour axes below this are noise in a style prior, not a note anyone would report.
    static let noteFloor = 0.35
    /// At most this many notes. A list of six descriptors reads as a spec sheet.
    static let maxNotes = 3

    /// Below this the vector describes a category, not a drink.
    ///
    /// The enrich step stamps 0.25 exactly when no style keyword matched and it fell back
    /// to the broad category centroid, so every one of those products shares a vector with
    /// its whole category: across the 293 that land there, the 202 beers produce two
    /// sentences and all 91 spirits produce the single word "Warming." Printed under a
    /// heading that says what this drink tastes like, that reads as a claim about the
    /// bottle in your hand rather than the wallpaper it is — so it is withheld.
    ///
    /// A named style scores 0.35 and stays: those 585 products yield 39 distinct
    /// sentences, which is a real description. The gate sits between the two tiers rather
    /// than on a curve, because confidence here takes five discrete values and anything
    /// from 0.26 to 0.35 selects exactly the same products.
    static let minConfidence = 0.30

    public static func sentence(for sensory: SensoryVector) -> String? {
        guard sensory.confidence >= minConfidence else { return nil }
        let clauses = [notes(sensory), structure(sensory)].compactMap { $0 }
        return clauses.isEmpty ? nil : clauses.joined(separator: " ")
    }

    /// "Bready malt, grassy hop and citrus."
    static func notes(_ sensory: SensoryVector) -> String? {
        let named = sensory.ranked
            .filter { !$0.axis.isStructure && $0.value >= noteFloor }
            .prefix(maxNotes)
            .map { $0.axis.note }
        guard let joined = list(Array(named)) else { return nil }
        return sentence(joined)
    }

    /// "Medium-bodied and mildly bitter, with a dry finish."
    static func structure(_ sensory: SensoryVector) -> String? {
        var adjectives: [String] = []

        if let body = sensory.value(.bodyFullness) {
            adjectives.append(body >= 0.65 ? "full-bodied"
                              : (body >= 0.35 ? "medium-bodied" : "light-bodied"))
        }
        if let bitter = sensory.value(.bitterness), bitter >= 0.35 {
            adjectives.append(bitter >= 0.65 ? "firmly bitter" : "mildly bitter")
        }
        if let fizz = sensory.value(.carbonation) {
            if fizz >= 0.65 { adjectives.append("lively") }
            else if fizz <= 0.25 { adjectives.append("barely sparkling") }
        }
        if let warmth = sensory.value(.alcoholWarmth), warmth >= 0.65 {
            adjectives.append("warming")
        }

        // The finish is the last thing you notice, so it gets the last clause rather than
        // being flattened into the adjective list.
        var finish: String?
        if let dry = sensory.value(.drynessFinish) {
            if dry >= 0.55 { finish = "a dry finish" }
            else if dry <= 0.25 { finish = "a sweet finish" }
        }

        switch (list(adjectives), finish) {
        case let (adj?, finish?): return sentence("\(adj), with \(finish)")
        case let (adj?, nil): return sentence(adj)
        case let (nil, finish?): return sentence("Has \(finish)")
        case (nil, nil): return nil
        }
    }

    /// Oxford-free list: "a", "a and b", "a, b and c".
    static func list(_ items: [String]) -> String? {
        switch items.count {
        case 0: return nil
        case 1: return items[0]
        case 2: return "\(items[0]) and \(items[1])"
        default: return items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
        }
    }

    static func sentence(_ body: String) -> String {
        body.prefix(1).uppercased() + body.dropFirst() + "."
    }
}
