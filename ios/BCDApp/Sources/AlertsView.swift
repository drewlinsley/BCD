import SwiftUI
import BCDKit

// Sentinel hits land here. Tier-1 agentic action (v1) is a deep link into the retailer's
// own checkout — never an autonomous purchase (see docs/06-legal.md).

struct AlertsView: View {
    @State private var alerts: [SentinelAlert] = SentinelAlert.demo

    var body: some View {
        NavigationStack {
            List(alerts) { alert in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: alert.icon)
                        Text(alert.title).font(.headline)
                        Spacer()
                        Text(alert.kind.rawValue).font(.caption2)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(.quaternary, in: Capsule())
                    }
                    Text(alert.detail).font(.subheadline).foregroundStyle(.secondary)
                    if let venue = alert.venue {
                        Label(venue, systemImage: "mappin.and.ellipse").font(.caption)
                    }
                    HStack {
                        Button("Open in store") {}.buttonStyle(.borderedProminent)
                        Button("Wishlist") {}.buttonStyle(.bordered)
                    }.padding(.top, 4)
                }
                .padding(.vertical, 4)
            }
            .navigationTitle("Alerts")
        }
    }
}

struct SentinelAlert: Identifiable {
    enum Kind: String { case release, collab, allocatedDrop = "allocated", menuChange = "menu" }
    let id = UUID()
    let title: String
    let detail: String
    let kind: Kind
    let venue: String?
    var icon: String {
        switch kind {
        case .release: "sparkles"
        case .collab: "person.2.fill"
        case .allocatedDrop: "flame.fill"
        case .menuChange: "list.bullet.rectangle"
        }
    }

    static let demo: [SentinelAlert] = [
        .init(title: "The Alchemist × Hill Farmstead collab",
              detail: "Matches your hazy IPA + Brett affinities. Cans drop Saturday.",
              kind: .collab, venue: nil),
        .init(title: "Buffalo Trace Antique Collection",
              detail: "Allocated drop tracked at 2 stores within 5 miles.",
              kind: .allocatedDrop, venue: "Binny's — Lincoln Park"),
        .init(title: "New on tap: Russian River Pliny the Younger",
              detail: "A bar you follow just added it.",
              kind: .menuChange, venue: "Toronado"),
    ]
}
