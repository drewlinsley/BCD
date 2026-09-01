import SwiftUI
import BCDKit

/// The user's consent tiers, persisted and shared. Previously the Privacy toggles were
/// view-local `@State` and changed nothing; a reaction posted to `/v1/feedback` becomes a
/// personalization-tier event, so the picker has to be able to read a real answer before
/// it sends one.
@MainActor
final class ConsentStore: ObservableObject {
    @Published var analytics: Bool { didSet { persist("analytics", analytics) } }
    @Published var personalization: Bool { didSet { persist("personalization", personalization) } }
    @Published var dataSharing: Bool { didSet { persist("data_sharing", dataSharing) } }

    private let defaults: UserDefaults
    private static let prefix = "bcd.consent."

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        // Analytics defaults on (the tier that carries no taste signal); the two tiers
        // that shape a profile stay off until the user says otherwise.
        analytics = defaults.object(forKey: Self.prefix + "analytics") as? Bool ?? true
        personalization = defaults.bool(forKey: Self.prefix + "personalization")
        dataSharing = defaults.bool(forKey: Self.prefix + "data_sharing")
    }

    var state: ConsentState {
        ConsentState(analytics: analytics, personalization: personalization,
                     dataSharing: dataSharing)
    }

    private func persist(_ key: String, _ value: Bool) {
        defaults.set(value, forKey: Self.prefix + key)
    }
}
