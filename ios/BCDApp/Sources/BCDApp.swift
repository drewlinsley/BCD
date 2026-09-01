import SwiftUI
import BCDKit

// App entry point + composition root. This is where concrete implementations are chosen;
// every screen below depends only on BCDKit protocols, so the on-device model, cloud
// model, or mock can be swapped here without touching a view.

@main
struct BCDApp: App {
    @StateObject private var env = AppEnvironment.live()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(env)
                .environmentObject(env.consent)
        }
    }
}

@MainActor
final class AppEnvironment: ObservableObject {
    let api: APIClientProtocol
    let llm: LLMProvider
    let telemetry: TelemetryQueue
    let makeScanEngine: () -> ScanEngine
    /// Consent tiers, shared and persisted — the picker has to read a real answer before
    /// it turns a tap into a personalization-tier event.
    let consent: ConsentStore
    /// What this install has already rated, so a product can show its own verdict on
    /// recall without a round trip.
    let reactions: ReactionLog
    /// Drinks the user has opened — the queue the Rate tab works through.
    let seen: SeenLog
    /// Pseudonymous per-install id. The only identity the server keys a profile on.
    let installId: String

    init(api: APIClientProtocol, llm: LLMProvider, telemetry: TelemetryQueue,
         makeScanEngine: @escaping () -> ScanEngine,
         consent: ConsentStore = ConsentStore(),
         reactions: ReactionLog = ReactionLog(),
         seen: SeenLog = SeenLog(),
         installId: String = InstallIdentity.current) {
        self.api = api
        self.llm = llm
        self.telemetry = telemetry
        self.makeScanEngine = makeScanEngine
        self.consent = consent
        self.reactions = reactions
        self.seen = seen
        self.installId = installId
    }

    static func live() -> AppEnvironment {
        let api = APIClient(baseURL: Self.apiBaseURL())
        // Consent is read from what the user actually chose last run, not assumed.
        let consent = ConsentStore()
        let telemetry = TelemetryQueue(consent: consent.state, sink: api,
                                       storeURL: Self.telemetryStoreURL())
        // Pick the LLM provider available on this device. Foundation Models is used only
        // where it exists AND is ready; otherwise the mock (or a cloud provider) stands in.
        let llm = Self.bestLLMProvider()
        return AppEnvironment(
            api: api, llm: llm, telemetry: telemetry,
            makeScanEngine: { Self.makeScanEngine() }, consent: consent
        )
    }

    /// API base URL, resolved in priority order:
    ///  1. `BCD_API_BASE` env var — set in the Xcode scheme for a quick dev override.
    ///  2. `BCDAPIBase` from Info.plist — baked in at build time from the BCD_API_BASE build
    ///     setting (Local.xcconfig), so a device build launched by tapping the icon points at
    ///     your Mac's LAN IP rather than localhost.
    ///  3. `http://localhost:8000` — Simulator default.
    private static func apiBaseURL() -> URL {
        if let env = ProcessInfo.processInfo.environment["BCD_API_BASE"], !env.isEmpty,
           let url = URL(string: env) { return url }
        if let s = Bundle.main.object(forInfoDictionaryKey: "BCDAPIBase") as? String,
           !s.isEmpty, let url = URL(string: s) { return url }
        return URL(string: "http://localhost:8000")!
    }

    private static func bestLLMProvider() -> LLMProvider {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            let provider = FoundationModelsProvider()
            if provider.isAvailable { return provider }
        }
        #endif
        return MockLLMProvider()
    }

    private static func makeScanEngine() -> ScanEngine {
        #if canImport(VisionKit) && os(iOS)
        if #available(iOS 18.0, *) { return VisionKitScanEngine() }
        #endif
        // Host/preview fallback so the app is runnable in the Simulator on Intel too.
        return MockScanEngine(scripted: [
            [DetectedText(text: "Heady Topper", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.08)],
            [DetectedText(text: "Pliny the Elder", kind: "text", x: 0.15, y: 0.5, w: 0.6, h: 0.08)],
        ])
    }

    private static func telemetryStoreURL() -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory,
                                           in: .userDomainMask)[0]
        return dir.appendingPathComponent("bcd_telemetry_queue.json")
    }
}
