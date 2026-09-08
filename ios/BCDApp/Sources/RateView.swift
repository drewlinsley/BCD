import SwiftUI
import BCDKit

// The rating surface: the drinks you've looked at, waiting for a verdict.
//
// Deliberately a thin host. Everything that matters happens in `ReactionPicker`, which knows
// nothing about this screen — it takes a product id and talks to the profile. If rating later
// belongs somewhere else (a "had it" list, a sheet off the scan HUD, a widget), moving it is a
// question of where the picker is called from, not a rewrite. Keep it that way.

struct RateView: View {
    @EnvironmentObject var env: AppEnvironment
    @State private var seen: [SeenProduct] = []

    var body: some View {
        NavigationStack {
            Group {
                if seen.isEmpty { emptyState } else { queue }
            }
            .navigationTitle("Rate")
        }
        // Re-read on every appearance: the list is built from what the scan tab recorded
        // while this view wasn't on screen.
        .onAppear { seen = env.seen.all() }
    }

    private var queue: some View {
        List {
            Section {
                ForEach(seen) { item in
                    VStack(alignment: .leading, spacing: 10) {
                        header(for: item)
                        ReactionPicker(productId: item.id)
                    }
                    .padding(.vertical, 6)
                }
                .onDelete(perform: dismiss)
            } header: {
                Text("Recently opened")
            }
        }
    }

    private func header(for item: SeenProduct) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(item.name)
                .font(.headline)
                .foregroundStyle(Brand.text)
            Text(subtitle(for: item))
                .font(.caption)
                .foregroundStyle(Brand.textMuted)
        }
    }

    private func subtitle(for item: SeenProduct) -> String {
        var parts = [item.producer]
        if let abv = item.abvPct { parts.append(String(format: "%.1f%% ABV", abv)) }
        return parts.joined(separator: " · ")
    }

    private var emptyState: some View {
        ContentUnavailableView(
            "Nothing to rate yet",
            image: "Reaction3",
            description: Text("Drinks you open from a scan show up here.")
        )
    }

    /// Swiping a row away only drops it from this queue — a verdict already sent stays in the
    /// profile, because it was the user's answer and this list is just a worklist.
    private func dismiss(at offsets: IndexSet) {
        for index in offsets { env.seen.remove(seen[index].id) }
        seen.remove(atOffsets: offsets)
    }
}
