import SwiftUI
import BCDKit

struct RootView: View {
    @EnvironmentObject var env: AppEnvironment

    var body: some View {
        TabView {
            ScanView()
                .tabItem { Label("Scan", systemImage: "camera.viewfinder") }
            SearchView()
                .tabItem { Label("Search", systemImage: "magnifyingglass") }
            RateView()
                .tabItem { Label("Rate", systemImage: "hand.thumbsup") }
            AlertsView()
                .tabItem { Label("Alerts", systemImage: "bell.badge") }
            ProfileView()
                .tabItem { Label("You", systemImage: "person.crop.circle") }
        }
        .tint(Brand.amber)
        .task { try? await env.telemetry.log("session_start", tier: .analytics) }
    }
}
