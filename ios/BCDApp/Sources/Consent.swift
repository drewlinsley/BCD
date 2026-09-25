import SwiftUI
import BCDKit

/// The user's consent tiers, persisted and shared. Previously the Privacy toggles were
/// view-local `@State` and changed nothing; a reaction posted to `/v1/feedback` becomes a
/// personalization-tier event, so the picker has to be able to read a real answer before
/// it sends one.
@MainActor
final class ConsentStore: ObservableObject {
    @Published var analytics: Bool { didSet { changed("analytics", analytics) } }
    @Published var personalization: Bool { didSet { changed("personalization", personalization) } }
    @Published var dataSharing: Bool { didSet { changed("data_sharing", dataSharing) } }
    /// Whether a photo of the label may be sent for identification when text alone fails.
    ///
    /// Deliberately not folded into `dataSharing`. That toggle reads "ads & insights" and is
    /// about what is done with a taste profile; this is a picture of whatever the camera is
    /// pointed at, which is a different thing to agree to and belongs on its own switch.
    @Published var labelPhotos: Bool { didSet { changed("label_photos", labelPhotos) } }

    /// Called after a tier settles, so whoever is gating on consent can be told. The
    /// telemetry queue is the one that has to hear it: built with a snapshot, it went on
    /// dropping personalization events for the rest of the run after the user said yes.
    var onChange: ((ConsentState) -> Void)?

    private let defaults: UserDefaults
    private static let prefix = "bcd.consent."

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        // Analytics defaults on (the tier that carries no taste signal); the two tiers
        // that shape a profile stay off until the user says otherwise.
        analytics = defaults.object(forKey: Self.prefix + "analytics") as? Bool ?? true
        personalization = defaults.bool(forKey: Self.prefix + "personalization")
        dataSharing = defaults.bool(forKey: Self.prefix + "data_sharing")
        labelPhotos = defaults.bool(forKey: Self.prefix + "label_photos")
    }

    var state: ConsentState {
        ConsentState(analytics: analytics, personalization: personalization,
                     dataSharing: dataSharing)
    }

    /// `didSet`, so the new value is already stored and `state` reads true.
    private func changed(_ key: String, _ value: Bool) {
        defaults.set(value, forKey: Self.prefix + key)
        onChange?(state)
    }
}
