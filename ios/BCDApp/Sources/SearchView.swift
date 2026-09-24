import SwiftUI
import BCDKit

// Search, and — on the empty screen you land on before typing — what the server suggests.
//
// The tab used to show a placeholder ("Type a beer or spirit name") on a catalog of 534k
// products with a ranker already built for it. `/v1/recommend` answers the question that
// placeholder was standing in for, so it fills the same space.
//
// The header is careful about whose taste it is. The server answers from a learned profile
// once you have rated something and from its seed profile before that, and both come back
// looking identical — so the client, which is the only side that knows how many verdicts
// this install has given, says which one you are reading.

struct SearchView: View {
    @EnvironmentObject var env: AppEnvironment
    @State private var query = ""
    @State private var results: [ResolvedProduct] = []
    @State private var searching = false
    /// The query the results belong to, or nil when nothing has been searched yet. Typing is
    /// not searching: keyed on `query` alone the screen told you "No matches" for a word you
    /// had not submitted, which is a claim about the catalog rather than about the screen.
    @State private var submitted: String?

    @State private var picks: [Recommendation] = []
    @State private var picksState = PicksState.idle
    /// The row whose product is being fetched, so it can show it is working. A recommendation
    /// carries a name and an id, not a whole product, and the detail screen needs the product.
    @State private var opening: String?
    @State private var detail: ScoredCandidate?

    enum PicksState: Equatable { case idle, loading, ready, failed }

    private var showingPicks: Bool { submitted == nil }

    var body: some View {
        NavigationStack {
            Group {
                if showingPicks { picksList } else { searchList }
            }
            .navigationTitle(showingPicks ? "Discover" : "Search")
            .searchable(text: $query)
            .onSubmit(of: .search) { Task { await run() } }
            // Clearing the field puts the suggestions back rather than leaving a stale list.
            .onChange(of: query) { _, new in
                if new.isEmpty { results = []; submitted = nil }
            }
            .task { await loadPicks() }
            .sheet(item: $detail) { ProductDetailView(candidate: $0) }
        }
    }

    // MARK: - what the server suggests

    @ViewBuilder private var picksList: some View {
        switch picksState {
        case .idle, .loading:
            ProgressView("Finding something you'd like…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed:
            ContentUnavailableView {
                Label("Can't reach the server", systemImage: "wifi.exclamationmark")
            } description: {
                Text("Search still works if you know what you're after.")
            } actions: {
                Button("Try again") { Task { await loadPicks() } }
            }
        case .ready:
            List {
                Section {
                    ForEach(Array(picks.enumerated()), id: \.element.id) { rank, pick in
                        Button { Task { await open(pick, rank: rank) } } label: {
                            PickRow(pick: pick,
                                    mine: env.reactions.reaction(for: pick.productId),
                                    busy: opening == pick.productId)
                        }
                        .buttonStyle(.plain)
                    }
                } header: {
                    Text(rated == 0 ? "Somewhere to start" : "For you")
                } footer: {
                    Text(rated == 0
                         ? "These come from a starting profile, not yours yet. Rate a few "
                           + "drinks and the list becomes your own."
                         : "Built from the \(rated) drink\(rated == 1 ? "" : "s") you've rated.")
                }
            }
            .refreshable { await loadPicks() }
            .overlay { if picks.isEmpty { ContentUnavailableView(
                "Nothing to suggest yet", systemImage: "sparkles",
                description: Text("The catalog has no scored drinks for this profile.")) } }
        }
    }

    private var rated: Int { env.reactions.count }

    private func loadPicks() async {
        if picksState != .ready { picksState = .loading }
        do {
            picks = try await env.api.recommend(limit: 15)
            picksState = .ready
            _ = await env.telemetry.log(
                TelemetryEvent.recommendationsShown.rawValue, tier: .analytics,
                ["n_results": .int(picks.count), "n_rated": .int(rated),
                 "top_evidence": .string(picks.first?.evidence.rawValue ?? "guessed")])
        } catch {
            picksState = .failed
        }
    }

    /// A recommendation is a name and an id; the detail screen wants the product. The name
    /// goes back through the search route and the id picks the right row out of the answer --
    /// names repeat in the registry, so matching on the name alone would sometimes open a
    /// different beer than the one tapped.
    private func open(_ pick: Recommendation, rank: Int) async {
        opening = pick.productId
        defer { opening = nil }
        let hits = (try? await env.api.searchProducts(pick.name)) ?? []
        guard let match = hits.first(where: { $0.product.id == pick.productId }) else { return }
        detail = ScoredCandidate(resolved: match, matchScore: 1.0, personalScore: pick.score,
                                 reason: pick.reason, coldStart: pick.coldStart)
        _ = await env.telemetry.log(
            TelemetryEvent.recommendationOpened.rawValue, tier: .analytics,
            ["product_id": .string(pick.productId), "rank": .int(rank),
             "evidence": .string(pick.evidence.rawValue)])
    }

    // MARK: - search

    private var searchList: some View {
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

    private func run() async {
        guard !query.isEmpty else { return }
        searching = true
        submitted = query
        defer { searching = false }
        results = (try? await env.api.searchProducts(query)) ?? []
    }
}

/// One suggestion. The score is shown because the whole point is that it is a prediction
/// about you, and the evidence chip because a guess from a style centroid and a profile of
/// this exact drink are not the same claim.
struct PickRow: View {
    let pick: Recommendation
    let mine: Reaction?
    let busy: Bool

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(pick.name)
                    .font(.headline).foregroundStyle(Brand.text)
                    .lineLimit(2)
                if let producer = pick.producer, !producer.isEmpty {
                    Text(DisplayName.producer(producer))
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack(spacing: 6) {
                    Text(pick.reason)
                        .font(.caption).foregroundStyle(Brand.amber)
                        .lineLimit(1)
                    EvidenceChip(evidence: pick.evidence)
                }
            }
            Spacer(minLength: 8)
            if busy {
                ProgressView().controlSize(.small)
            } else if mine != nil {
                ReactionBadge(reaction: mine)
            } else {
                // Truncated, not rounded -- the detail screen prints `Int(score * 100)`,
                // and a number that changes by one when you tap the row reads as a bug.
                Text("\(Int(pick.score * 100))")
                    .font(.callout.monospacedDigit().weight(.semibold))
                    .foregroundStyle(Brand.amber)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

struct EvidenceChip: View {
    let evidence: Recommendation.Evidence

    private var tint: Color {
        switch evidence {
        case .rated: return .green
        case .known: return Brand.amber
        case .guessed: return Brand.textMuted
        }
    }

    var body: some View {
        Text(evidence.rawValue)
            .font(.caption2)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(tint.opacity(0.15), in: Capsule())
            .foregroundStyle(tint)
            .accessibilityLabel(evidence.blurb)
    }
}
