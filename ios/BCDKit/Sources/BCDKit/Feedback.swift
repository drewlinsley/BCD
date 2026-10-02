import Foundation
import Combine

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

    /// What the rung is called on screen, which is not what the case is called.
    ///
    /// Shorter than the case names, because these sit in a row of five under the glyphs and
    /// "Poured it out" was the one label that wrapped to two lines, making that column taller
    /// and the whole row ragged. The cases keep their longer spellings: the `rawValue` is the
    /// stored and wire identity, so a label is free to be re-worded and a rung is not.
    ///
    /// Rung 3 is "OK" rather than "Fine" — both are the pivot, and neither may read as a mild
    /// negative, but "Fine" is the English for faint disappointment as often as for contentment.
    public var label: String {
        switch self {
        case .spatItOut: "Spat out"
        case .pouredItOut: "Poured out"
        case .fine: "OK"
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

/// What `POST /v1/feedback/withdraw` takes — the id of the verdict being taken back.
///
/// Its own type rather than a `FeedbackRequest` with a nil rating: a withdrawal is not a
/// rating with something missing, and the route must not be reachable by forgetting a field.
/// The client sets no key strategy, so the one key is spelled out (see `ContributionWire`).
struct WithdrawBody: Encodable {
    let productId: String

    enum CodingKeys: String, CodingKey {
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
///
/// An `ObservableObject`, because a verdict is read in one place and changed in another: a
/// search row draws the glyph for a drink whose rating sheet is two screens away. Without a
/// change signal an already-drawn row kept the face the log held when the row was built --
/// rate a drink, search for it, then take the rating back from its detail screen, and the
/// row behind still showed the old glyph until the next launch. The stored data was never
/// wrong; nothing told SwiftUI to ask again.
public final class ReactionLog: ObservableObject, @unchecked Sendable {
    private let key = "bcd.reactions"
    private let defaults: UserDefaults
    private let lock = NSLock()

    /// Bumped on every write, and that is the whole of the change signal: a view holding
    /// this log re-reads when it moves.
    ///
    /// A counter rather than the verdicts themselves. The store is `UserDefaults` and every
    /// reader asks by product id, so publishing the dictionary would mean keeping a second
    /// copy of it in step with the first -- two answers to one question, which is the bug
    /// this fixes.
    @Published public private(set) var revision = 0

    public init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    public func reaction(for productId: String) -> Reaction? {
        lock.lock(); defer { lock.unlock() }
        guard let raw = defaults.dictionary(forKey: key)?[productId] as? Int else { return nil }
        return Reaction(rawValue: raw)
    }

    /// How many verdicts this install has given. The recommendation list needs it to say
    /// whose taste it is showing: with nothing rated, the server answers from its seed
    /// profile, and calling that "for you" would be a lie.
    public var count: Int {
        lock.lock(); defer { lock.unlock() }
        return (defaults.dictionary(forKey: key) ?? [:]).count
    }

    public func record(_ reaction: Reaction, for productId: String) {
        write { $0[productId] = reaction.rawValue }
    }

    /// Forget a verdict. The picker could move a rating between rungs but never take one off,
    /// so a face tapped by mistake stood for good — and because `rated_products` reads the
    /// same verdicts, it also barred that drink from "For you" permanently.
    ///
    /// Clearing here is only half of it: this log is a display cache and the server holds the
    /// profile, so the caller withdraws on the server too (`APIClientProtocol.withdrawFeedback`).
    public func remove(for productId: String) {
        write { $0.removeValue(forKey: productId) }
    }

    /// The one writer, so a later one cannot be added that forgets to announce itself.
    ///
    /// `revision` moves after the lock is dropped. `objectWillChange` delivers to its
    /// subscribers synchronously, and a subscriber here is a view that may read straight
    /// back through `reaction(for:)` -- which wants this same non-recursive lock.
    ///
    /// Called from the views, on the main actor, which is where SwiftUI requires a
    /// published change to come from; nothing in the package writes off it.
    private func write(_ change: (inout [String: Any]) -> Void) {
        lock.lock()
        var all = defaults.dictionary(forKey: key) ?? [:]
        change(&all)
        defaults.set(all, forKey: key)
        lock.unlock()
        revision += 1
    }
}
