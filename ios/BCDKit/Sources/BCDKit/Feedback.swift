import Foundation

// The taste verdict, client side. `/v1/feedback` records a real `rating_submitted` event
// and folds it straight into the caller's TasteProfile, so a tap here moves the same
// Rocchio centroid the scan HUD ranks with — the reaction set is the input to that loop,
// not decoration.

/// One rung of the reaction scale. Levels map 1-5 onto the signed weight the profile
/// builder uses; 3 is the pivot and contributes no direction, so it must never be
/// presented as a mild negative.
public enum Reaction: Int, CaseIterable, Codable, Sendable, Identifiable {
    case spatItOut = 1, pouredItOut, fine, pinkieOut, chuggedIt

    public var id: Int { rawValue }

    public var label: String {
        switch self {
        case .spatItOut: "Spat it out"
        case .pouredItOut: "Poured it out"
        case .fine: "Fine"
        case .pinkieOut: "Pinkie out"
        case .chuggedIt: "Chugged it"
        }
    }

    /// The signed weight this rating carries into the taste centroid.
    public var weight: Double { (Double(rawValue) - 3.0) / 2.0 }

    /// What picking this actually does to the profile — shown under the picker so the
    /// scale is legible as a mechanism rather than a mood.
    public var note: String {
        switch self {
        case .spatItOut:
            "Pushes your taste centroid away from this product, damped by gamma 0.4 so one "
            + "bad pour can't erase a whole style."
        case .pouredItOut:
            "A soft negative. Moves the centroid away, at half the weight of a drain pour."
        case .fine:
            "The pivot. Contributes no direction to the centroid, but still counts as a "
            + "rated product."
        case .pinkieOut:
            "A soft positive. Pulls the centroid toward this product at half weight."
        case .chuggedIt:
            "Pulls the centroid hard toward this product and lifts the matching style affinity."
        }
    }
}

public struct FeedbackRequest: Codable, Sendable {
    public let productId: String
    public let rating: Double
    public let aspects: [String: Double]?

    public init(productId: String, rating: Double, aspects: [String: Double]? = nil) {
        self.productId = productId
        self.rating = rating
        self.aspects = aspects
    }

    public init(productId: String, reaction: Reaction) {
        self.init(productId: productId, rating: Double(reaction.rawValue))
    }

    enum CodingKeys: String, CodingKey {
        case rating, aspects
        case productId = "product_id"
    }
}

public struct FeedbackResponse: Codable, Sendable {
    public let accepted: Bool
    public let profile: TasteProfile
}

/// The pseudonymous per-install identity the server keys a profile on. Not an account id
/// and never derived from anything about the person — a random value, minted once and
/// kept in UserDefaults so a reinstall starts a genuinely new profile.
public enum InstallIdentity {
    private static let key = "bcd.install_id"

    public static var current: String {
        let defaults = UserDefaults.standard
        if let existing = defaults.string(forKey: key), !existing.isEmpty { return existing }
        let minted = UUID().uuidString.lowercased()
        defaults.set(minted, forKey: key)
        return minted
    }
}

/// What this install has already rated, so a product shows its own verdict on recall
/// without a round trip. The server stays the source of truth for the profile; this is a
/// display cache and is treated as disposable.
public final class ReactionLog: @unchecked Sendable {
    private let key = "bcd.reactions"
    private let defaults: UserDefaults
    private let lock = NSLock()

    public init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    public func reaction(for productId: String) -> Reaction? {
        lock.lock(); defer { lock.unlock() }
        guard let raw = defaults.dictionary(forKey: key)?[productId] as? Int else { return nil }
        return Reaction(rawValue: raw)
    }

    public func record(_ reaction: Reaction, for productId: String) {
        lock.lock(); defer { lock.unlock() }
        var all = defaults.dictionary(forKey: key) ?? [:]
        all[productId] = reaction.rawValue
        defaults.set(all, forKey: key)
    }
}
