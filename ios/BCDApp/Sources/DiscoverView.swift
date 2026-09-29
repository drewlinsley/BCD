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
    @State private var aisles: [FamilyGroup] = []
    /// Rank the shelves they have never rated on by the taste they built elsewhere. Off by
    /// default, and deliberately: asking which gin is most like an IPA is a real cosine and not
    /// a real recommendation, and it amplifies bad style data rather than surviving it. Measured
    /// on the live catalog, it topped Bourbon, Rum AND Tequila with the same Japanese whisky,
    /// because the registry filed each one's cask finish as its style.
    @AppStorage("bcd.discover.crossStyle") private var crossStyle = false
    @State private var open: Set<String> = []
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
                // It can also light up a whole shelf -- rate one gin and Gin stops being dark.
                .onChange(of: env.ratingsVersion) { _, _ in Task { await load() } }
                .onChange(of: crossStyle) { _, _ in Task { await load() } }
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
            // Before anything is rated there is one list, because there is one profile behind
            // every shelf and it is the seed's. Splitting it into twenty-two would be twenty-two
            // ways of saying the same thing about a drinker nobody knows yet.
            if rated == 0 { startingList } else { shelfList }
        }
    }

    private var startingList: some View {
        List {
            Section {
                ForEach(Array(picks.enumerated()), id: \.element.id) { rank, pick in
                    Button { Task { await openPick(pick, rank: rank) } } label: {
                        PickRow(pick: pick,
                                mine: env.reactions.reaction(for: pick.productId),
                                busy: opening == pick.productId)
                    }
                    .buttonStyle(.plain)
                }
            } header: {
                Text("Somewhere to start")
            } footer: {
                Text("These come from a starting profile, not yours yet. Rate a few drinks "
                     + "and the list becomes your own.")
            }
        }
        .refreshable { await load() }
        .overlay { if picks.isEmpty { ContentUnavailableView(
            "Nothing to suggest yet", systemImage: "sparkles",
            description: Text("The catalog has no scored drinks for this profile.")) } }
    }

    // One dropdown per shelf, the ones they have rated on first and open. A dark shelf is still
    // listed and still opens: hiding it would mean a drinker could not find out that gin is
    // something this app has an opinion about, or that rating one gin is what earns the opinion.
    private var shelfList: some View {
        List {
            // One Section for all of them. Twenty-two sections put an inset card and a gap
            // around every shelf, which turned a list you scan into a list you scroll.
            Section {
                ForEach(aisles) { aisle in
                    // Cider and Sake are an aisle holding one shelf of the same name. Nesting
                    // them would make the reader open "Cider" to find "Cider", so a lone shelf
                    // is shown at the aisle's own level and opens straight onto its drinks.
                    if aisle.families.count == 1, let only = aisle.families.first {
                        DisclosureGroup(isExpanded: expansion(of: aisle.id)) {
                            shelfBody(only)
                        } label: {
                            aisleLabel(aisle)
                        }
                    } else {
                        DisclosureGroup(isExpanded: expansion(of: aisle.id)) {
                            ForEach(aisle.families) { shelf in
                                DisclosureGroup(isExpanded: expansion(of: shelf.id)) {
                                    shelfBody(shelf)
                                } label: {
                                    ShelfHeader(shelf: shelf)
                                }
                            }
                        } label: {
                            aisleLabel(aisle)
                        }
                    }
                }
            }
            Section {
                Toggle("Guess across styles", isOn: $crossStyle)
                Text("Ranks the shelves you have not rated on by the taste you built "
                     + "elsewhere. It is a flavour match, not a verdict: your IPAs say which "
                     + "gin is nearest them, which is not the same as which gin you would like.")
                    .font(.caption).foregroundStyle(Brand.textMuted)
            }
        }
        .refreshable { await load() }
    }

    private func aisleLabel(_ aisle: FamilyGroup) -> some View {
        Text(aisle.label)
            .font(.title3.weight(.semibold))
            .foregroundStyle(aisle.ratedIn ? Brand.text : Brand.textMuted)
    }

    @ViewBuilder private func shelfBody(_ shelf: FamilyPicks) -> some View {
        ForEach(Array(shelf.results.enumerated()), id: \.element.id) { rank, pick in
            Button { Task { await openShelfPick(pick, rank: rank) } } label: {
                ShelfRow(pick: pick,
                         mine: env.reactions.reaction(for: pick.productId),
                         busy: opening == pick.productId,
                         personal: shelf.isPersonal)
            }
            .buttonStyle(.plain)
        }
        if !shelf.isPersonal {
            Text("Nothing here is scored for you \u{2014} rate one and this shelf becomes yours.")
                .font(.caption).foregroundStyle(Brand.textMuted)
        }
    }

    /// Open by key, for aisles and shelves alike -- both are `DisclosureGroup`s over the same
    /// set, so one aisle and one shelf can be open without a second piece of state.
    private func expansion(of key: String) -> Binding<Bool> {
        Binding(
            get: { open.contains(key) },
            set: { isOpen in
                if isOpen { open.insert(key) } else { open.remove(key) }
            })
    }

    private var rated: Int { env.reactions.count }

    private func load() async {
        if state != .ready { state = .loading }
        do {
            if rated == 0 {
                picks = try await env.api.recommend(limit: 15)
                aisles = []
            } else {
                let answer = try await env.api.familyPicks(limit: 6, crossStyle: crossStyle)
                aisles = answer.groups
                // Open the aisle they have rated in and the shelf inside it, and nothing else:
                // the screen should start as three rows and one answer, not as a scroll.
                // Only on the first load -- reopening what the reader shut, every time a rating
                // lands, would undo them.
                if open.isEmpty {
                    open = Set(answer.groups.filter(\.ratedIn).map(\.id))
                        .union(answer.groups.flatMap(\.families)
                            .filter(\.isPersonal).map(\.id))
                }
            }
            state = .ready
            _ = await env.telemetry.log(
                TelemetryEvent.recommendationsShown.rawValue, tier: .analytics,
                ["n_results": .int(rated == 0 ? picks.count
                                   : aisles.flatMap(\.families)
                                       .reduce(0) { $0 + $1.results.count }),
                 "n_rated": .int(rated),
                 "top_evidence": .string(rated == 0
                                         ? (picks.first?.evidence.rawValue ?? "guessed")
                                         : (aisles.first?.families.first?.results.first?
                                            .evidence.rawValue ?? "guessed"))])
        } catch {
            state = .failed
        }
    }

    /// A recommendation is a name and an id; the detail screen wants the product. The name
    /// goes back through the search route and the id picks the right row out of the answer --
    /// names repeat in the registry, so matching on the name alone would sometimes open a
    /// different beer than the one tapped.
    private func openShelfPick(_ pick: FamilyPick, rank: Int) async {
        await openProduct(id: pick.productId, name: pick.name, score: pick.score,
                          reason: pick.reason, coldStart: pick.coldStart,
                          evidence: pick.evidence, rank: rank)
    }

    private func openPick(_ pick: Recommendation, rank: Int) async {
        await openProduct(id: pick.productId, name: pick.name, score: pick.score,
                          reason: pick.reason, coldStart: pick.coldStart,
                          evidence: pick.evidence, rank: rank)
    }

    private func openProduct(id: String, name: String, score: Double?, reason: String?,
                             coldStart: Bool, evidence: Recommendation.Evidence,
                             rank: Int) async {
        opening = id
        defer { opening = nil }
        let hits = (try? await env.api.searchProducts(name)) ?? []
        guard let match = hits.first(where: { $0.product.id == id }) else { return }
        detail = ScoredCandidate(resolved: match, matchScore: 1.0, personalScore: score,
                                 reason: reason, coldStart: coldStart)
        _ = await env.telemetry.log(
            TelemetryEvent.recommendationOpened.rawValue, tier: .analytics,
            ["product_id": .string(id), "rank": .int(rank),
             "evidence": .string(evidence.rawValue)])
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


/// A shelf's name, and whether it is about the reader. A dark shelf is dimmed rather than
/// hidden: it still says gin exists and still opens, so the drinker can see what is on it and
/// what rating one would earn them.
private struct ShelfHeader: View {
    let shelf: FamilyPicks

    var body: some View {
        HStack(spacing: 8) {
            Text(shelf.label)
                .font(.headline)
                .foregroundStyle(shelf.isPersonal ? Brand.text : Brand.textMuted)
            if shelf.basis == .cross {
                Text("guessed").font(.caption2)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Brand.textMuted.opacity(0.15), in: Capsule())
                    .foregroundStyle(Brand.textMuted)
            }
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(shelf.isPersonal ? shelf.label
                            : "\(shelf.label), not rated by you yet")
    }
}

/// One drink on a shelf. The score appears only when there is one -- an unrated shelf shows the
/// drink and its evidence and stops there, because the number people read as "how much you'd
/// like it" is exactly what has not been worked out.
private struct ShelfRow: View {
    let pick: FamilyPick
    let mine: Reaction?
    let busy: Bool
    let personal: Bool

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(pick.name)
                    .font(.subheadline.weight(.medium)).foregroundStyle(Brand.text)
                    .lineLimit(2)
                if let producer = pick.producer, !producer.isEmpty {
                    Text(DisplayName.producer(producer))
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack(spacing: 6) {
                    if let reason = pick.reason {
                        Text(reason).font(.caption).foregroundStyle(Brand.amber).lineLimit(1)
                    }
                    EvidenceChip(evidence: pick.evidence)
                }
            }
            Spacer(minLength: 8)
            if busy {
                ProgressView().controlSize(.small)
            } else if mine != nil {
                ReactionBadge(reaction: mine)
            } else if let score = pick.score {
                // Truncated, not rounded -- the detail screen prints `Int(score * 100)`, and a
                // number that changes by one when you tap the row reads as a bug.
                Text("\(Int(score * 100))")
                    .font(.callout.monospacedDigit().weight(.semibold))
                    .foregroundStyle(Brand.amber)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .opacity(personal ? 1 : 0.72)
    }
}
