import SwiftUI
import BCDKit

struct RootView: View {
    @EnvironmentObject var env: AppEnvironment

    var body: some View {
        // Five, because iPhone shows five and hides the rest behind a "More" list. Alerts is
        // the one that gives: every alert in it is `SentinelAlert.demo` and both its buttons
        // are empty closures, so it costs nothing to take off the bar and is one line to put
        // back when the sentinel actually fires. `AlertsView` stays in the target.
        TabView {
            ScanView()
                .tabItem { Label("Scan", systemImage: "camera.viewfinder") }
            // `lightbulb.max` is the bulb with rays, not the bare one: a lit bulb reads as an
            // idea, an unlit one reads as a lamp. `wand.and.stars` was the other candidate and
            // is four separate marks in a 25pt box; the bulb is one closed shape with the rays
            // hung off it, which is what survives at tab-bar size. There is no
            // `lightbulb.and.sparkles` in SF Symbols.
            DiscoverView()
                .tabItem { Label("Discover", systemImage: "lightbulb.max") }
            SearchView()
                .tabItem { Label("Search", systemImage: "magnifyingglass") }
            RateView()
                .tabItem { Label("Rate", systemImage: "hand.thumbsup") }
            ProfileView()
                .tabItem { Label("You", systemImage: "person.crop.circle") }
        }
        .tint(Brand.amber)
        .task { try? await env.telemetry.log("session_start", tier: .analytics) }
    }
}
