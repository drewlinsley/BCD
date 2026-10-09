import SwiftUI
import BCDKit

struct RootView: View {
    @EnvironmentObject var env: AppEnvironment

    /// Whether the first-run quiz has been put to this install — answered OR skipped. Its own
    /// flag rather than `reactions.count == 0`, because someone who answered and then removed
    /// their only rating has already been asked, and being asked twice reads as the app
    /// forgetting them. @AppStorage so it survives a relaunch, which is the point.
    @AppStorage("bcd.quizAsked") private var quizAsked = false

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
            // The pivot face, not the top one. A tab names a place, and every other rung is a
            // verdict -- "Chugged it" on the bar would read as a tab full of drinks you liked.
            // Rung 3 is the one that contributes no direction to the centroid, so it is the
            // scale standing for itself. Also the app's own mark rather than SF Symbols'
            // `hand.thumbsup`, which promised two directions for a five-rung scale.
            RateView()
                .tabItem { Label("Rate", image: "Reaction3Small") }
            ProfileView()
                .tabItem { Label("You", systemImage: "person.crop.circle") }
        }
        .tint(Brand.amber)
        .task { try? await env.telemetry.log("session_start", tier: .analytics) }
        // The one screen that comes before the app. A drinker with no profile gets
        // recommendations built from a seed that is somebody else's taste, and the list says
        // "matches your tropical preference" to someone who has never said anything.
        //
        // A sheet, not a gate: it carries its own Skip, and dismissing it for any reason
        // marks it asked. A first run nobody can get past would be a worse failure than a
        // cold start, and rating anything builds the same profile by the same path.
        .fullScreenCover(isPresented: .init(get: { !quizAsked },
                                            set: { if !$0 { quizAsked = true } })) {
            QuizView { quizAsked = true }
                .environmentObject(env)
        }
    }
}
