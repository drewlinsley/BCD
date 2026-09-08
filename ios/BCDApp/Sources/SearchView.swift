import SwiftUI
import BCDKit

struct SearchView: View {
    @EnvironmentObject var env: AppEnvironment
    @State private var query = ""
    @State private var results: [ResolvedProduct] = []
    @State private var searching = false

    var body: some View {
        NavigationStack {
            List(results) { rp in
                HStack(spacing: 12) {
                    VStack(alignment: .leading) {
                        Text(rp.product.name).font(.headline)
                        Text(rp.producer.name).font(.caption).foregroundStyle(.secondary)
                        if let abv = rp.product.spec.abvPct {
                            Text("\(abv.value, specifier: "%.1f")% ABV").font(.caption2)
                        }
                    }
                    Spacer()
                    // Recall: your own verdict, at the 24pt list size (the heavier stroke).
                    ReactionBadge(reaction: env.reactions.reaction(for: rp.product.id))
                }
            }
            .overlay { if results.isEmpty && !searching { ContentUnavailableView(
                "Search the catalog", systemImage: "magnifyingglass",
                description: Text("Type a beer or spirit name")) } }
            .navigationTitle("Search")
            .searchable(text: $query)
            .onSubmit(of: .search) { Task { await run() } }
        }
    }

    private func run() async {
        guard !query.isEmpty else { return }
        searching = true
        defer { searching = false }
        results = (try? await env.api.searchProducts(query)) ?? []
    }
}
