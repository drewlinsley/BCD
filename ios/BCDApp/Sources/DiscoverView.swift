import SwiftUI
import BCDKit

// What the server suggests you drink next.
//
// Its own tab since 2026-09-25. It used to be what Search showed before you typed anything,
// which made the app's one genuinely proactive screen a thing you reached by not using the
// screen you were on — and gave two different jobs one icon.
//
// The header is careful about whose taste it is. The server answers from a learned profile
// once you have rated something and from its seed profile before that, and both come back
// looking identical — so the client, which is the only side that knows how many verdicts
// this install has given, says which one you are reading.

struct DiscoverView: View {
    @EnvironmentObject var env: AppEnvironment
    @State private var picks: [Recommendation] = []
    @State private var state = PicksState.idle
    /// The row whose product is being fetched, so it can show it is working. A recommendation
    /// carries a name and an id, not a whole product, and the detail screen needs the product.
    @State private var opening: String?
    @State private var detail: ScoredCandidate?

    enum PicksState: Equatable { case idle, loading, ready, failed }

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Discover")
                .task { await load() }
                // A rating moves the profile this list is ranked with, so the list it produced
                // a moment ago is no longer the answer. Reloaded here rather than left to a
                // pull: watching the suggestions change is the point of having rated anything.
                .onChange(of: env.ratingsVersion) { _, _ in Task { await load() } }
                .sheet(item: $detail) { ProductDetailView(candidate: $0) }
        }
    }

    @ViewBuilder private var content: some View {
        switch state {
        case .idle, .loading:
            ProgressView("Finding something you'd like…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed:
            ContentUnavailableView {
                Label("Can't reach the server", systemImage: "wifi.exclamationmark")
            } description: {
                Text("Search still works if you know what you're after.")
            } actions: {
                Button("Try again") { Task { await load() } }
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
            .refreshable { await load() }
            .overlay { if picks.isEmpty { ContentUnavailableView(
                "Nothing to suggest yet", systemImage: "sparkles",
                description: Text("The catalog has no scored drinks for this profile.")) } }
        }
    }

    private var rated: Int { env.reactions.count }

    private func load() async {
        if state != .ready { state = .loading }
        do {
            picks = try await env.api.recommend(limit: 15)
            state = .ready
            _ = await env.telemetry.log(
                TelemetryEvent.recommendationsShown.rawValue, tier: .analytics,
                ["n_results": .int(picks.count), "n_rated": .int(rated),
                 "top_evidence": .string(picks.first?.evidence.rawValue ?? "guessed")])
        } catch {
            state = .failed
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
