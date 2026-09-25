import SwiftUI
import BCDKit

// Looking something up by name, and getting back to what you looked up before.
//
// This tab used to be two screens wearing one icon: what the server suggests until you typed,
// and search results after. Discover has its own tab now, so this one only does the thing its
// magnifying glass promises — and the space before you type belongs to your own history
// rather than to a placeholder.
//
// Recent searches earn that space because the catalog has 534k rows and the registry files a
// lot of near-names: the words that found a drink last week are worth more than remembering
// which of three spellings was the one that worked.

struct SearchView: View {
    @EnvironmentObject var env: AppEnvironment
    @State private var query = ""
    @State private var results: [ResolvedProduct] = []
    @State private var searching = false
    /// The query the results belong to, or nil when nothing has been searched yet. Typing is
    /// not searching: keyed on `query` alone the screen told you "No matches" for a word you
    /// had not submitted, which is a claim about the catalog rather than about the screen.
    @State private var submitted: String?
    @State private var recents: [String] = []
    @State private var detail: ScoredCandidate?

    var body: some View {
        NavigationStack {
            Group {
                if submitted == nil { recentsList } else { resultsList }
            }
            .navigationTitle("Search")
            .searchable(text: $query, prompt: "Beer or spirit name")
            .onSubmit(of: .search) { Task { await run(query) } }
            // Clearing the field puts the history back rather than leaving a stale list.
            .onChange(of: query) { _, new in
                if new.isEmpty { results = []; submitted = nil; recents = env.recents.all() }
            }
            .task { recents = env.recents.all() }
            .sheet(item: $detail) { ProductDetailView(candidate: $0) }
        }
    }

    // MARK: - what you looked up before

    @ViewBuilder private var recentsList: some View {
        if recents.isEmpty {
            ContentUnavailableView(
                "Search the catalog", systemImage: "magnifyingglass",
                description: Text("534,000 beers and spirits. Type a name, or point the "
                                  + "camera at a label from the Scan tab."))
        } else {
            List {
                Section {
                    ForEach(recents, id: \.self) { term in
                        Button { query = term; Task { await run(term) } } label: {
                            HStack(spacing: 12) {
                                Image(systemName: "clock.arrow.circlepath")
                                    .font(.callout)
                                    .foregroundStyle(Brand.textMuted)
                                Text(term).foregroundStyle(Brand.text)
                                Spacer(minLength: 8)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                    .onDelete { offsets in
                        for i in offsets { env.recents.remove(recents[i]) }
                        recents.remove(atOffsets: offsets)
                    }
                } header: {
                    HStack {
                        Text("Recent")
                        Spacer()
                        Button("Clear") {
                            env.recents.clear()
                            recents = []
                        }
                        .font(.caption)
                        .textCase(nil)
                    }
                }
            }
        }
    }

    // MARK: - results

    private var resultsList: some View {
        List(results) { rp in
            Button { open(rp) } label: {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(DisplayName.label(rp.product.name, brand: rp.brand.name))
                            .font(.headline).foregroundStyle(Brand.text)
                        Text(DisplayName.producer(rp.producer.name))
                            .font(.caption).foregroundStyle(.secondary)
                        if let abv = rp.product.spec.abvPct {
                            Text("\(abv.value, specifier: "%.1f")% ABV")
                                .font(.caption2).foregroundStyle(Brand.textMuted)
                        }
                    }
                    Spacer(minLength: 8)
                    // Recall: your own verdict, at the 24pt list size (the heavier stroke).
                    ReactionBadge(reaction: env.reactions.reaction(for: rp.product.id))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .overlay {
            if searching {
                ProgressView()
            } else if results.isEmpty {
                ContentUnavailableView(
                    "No matches for “\(submitted ?? "")”",
                    systemImage: "magnifyingglass",
                    description: Text("Try fewer words — the registry files a lot of near-names."))
            }
        }
    }

    /// A search result is already the whole product, so this opens straight from what is on
    /// screen — unlike a recommendation, which is a name and an id and has to be fetched.
    /// No personal score: nothing here was ranked for anyone, and a seal on the detail screen
    /// would be inventing one.
    private func open(_ rp: ResolvedProduct) {
        detail = ScoredCandidate(resolved: rp, matchScore: 1.0)
    }

    private func run(_ term: String) async {
        let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        searching = true
        submitted = trimmed
        defer { searching = false }
        results = (try? await env.api.searchProducts(trimmed)) ?? []
        // Recorded whatever came back: "no matches" is a thing you may well want to try again
        // later, spelled differently, and hiding it makes the history a record of the app's
        // successes rather than of what you were after.
        env.recents.record(trimmed)
        recents = env.recents.all()
    }
}
