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
    /// Observed, not read through `env`: these rows outlive the sheet that changes a
    /// verdict, and a row that cannot hear the log goes stale behind it.
    @EnvironmentObject var reactions: ReactionLog
    @State private var query = ""
    /// `waiting` before a search, `answered` with whatever came back -- including nothing,
    /// which is a fact about the catalog -- and `failed` when the asking did not work. Three
    /// states, because the empty state now offers to add the drink, and "we do not have it"
    /// is the one thing a failed request does not establish.
    @State private var results: Asked<[ResolvedProduct]> = .waiting
    @State private var searching = false
    /// The query the results belong to, or nil when nothing has been searched yet. Typing is
    /// not searching: keyed on `query` alone the screen told you "No matches" for a word you
    /// had not submitted, which is a claim about the catalog rather than about the screen.
    @State private var submitted: String?
    @State private var recents: [String] = []
    @State private var detail: ScoredCandidate?
    /// Describing a drink the catalog could not produce. Reached from the empty state, which
    /// until now was a dead end: `AddDrinkView` had exactly one door, the scan HUD, so a
    /// drink you could name but not photograph could not be added at all (user, 2026-10-16).
    @State private var adding = false
    /// A match recognised inside the add sheet -- it was in the catalog under another
    /// spelling. Opened after that sheet closes, because two presentations overlapping is
    /// what reliably drops both.
    @State private var picked: ResolvedProduct?

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
                if new.isEmpty {
                    results = .waiting; submitted = nil; recents = env.recents.all()
                }
            }
            .task { recents = env.recents.all() }
            .sheet(item: $detail) {
                ProductDetailView(candidate: $0, referrer: .search)
                    .presentationDragIndicator(.visible)
            }
            // The same sheet the scan HUD opens, from the other door. `onDismiss` rather than
            // setting `detail` from inside it, for the same reason ScanView does: a drink
            // recognised in the match list has to replace this sheet, not sit under it.
            .sheet(isPresented: $adding, onDismiss: openPickedMatch) {
                AddDrinkView(sightings: [],
                             typedName: submitted ?? query,
                             onSubmit: submitContribution,
                             search: { term in
                                 (try? await env.api.searchProducts(term)) ?? []
                             },
                             onPick: { picked = $0 })
            }
        }
    }

    private func openPickedMatch() {
        guard let match = picked else { return }
        picked = nil
        open(match)
    }

    /// Write it down first -- the log is the record of truth and has to survive a failed
    /// upload -- then try to send it. Same two steps `ScanView` takes, deliberately: a
    /// contribution from the search box and one from the camera are the same claim.
    private func submitContribution(_ contribution: DrinkContribution) {
        env.contributions.add(contribution)
        Task { await env.contributionUploader.drain() }
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
        List(results.answer ?? []) { rp in
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
                    ReactionBadge(reaction: reactions.reaction(for: rp.product.id))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .overlay {
            switch results {
            case _ where searching:
                ProgressView()
            case .answered(let found) where found.isEmpty:
                nothingFound
            case .failed:
                // Not the shrug. The shrug says the catalog does not have this drink and
                // offers to be told about it, and a request that never arrived has not
                // established that -- offering to add a drink we may well already hold would
                // be the app's own failure dressed up as the catalog's gap.
                ContentUnavailableView {
                    Label("Couldn't search", systemImage: "arrow.clockwise")
                } description: {
                    Text("The catalog couldn't be reached.")
                } actions: {
                    Button("Try again") { Task { await run(submitted ?? query) } }
                }
            case .waiting, .answered:
                EmptyView()
            }
        }
    }

    /// The catalog really does not have it.
    ///
    /// The shrug is the sixth face in the taster set: the same circle and the same glass as
    /// the five verdicts, with a question mark where the level goes. It says what the five
    /// cannot -- not how a drink was, but that we do not know what the drink is -- and it is
    /// drawn rather than borrowed, like every other mark in the app that carries meaning.
    ///
    /// Under it, the way out of the dead end. This screen used to say "try fewer words", which
    /// is advice for the case where the catalog has the drink; when it genuinely does not, the
    /// only person who can fix that is the one holding the can (user, 2026-10-16).
    private var nothingFound: some View {
        VStack(spacing: 18) {
            Shrug.glyph(at: Brand.GlyphSize.display)
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: Brand.GlyphSize.display, height: Brand.GlyphSize.display)
                .foregroundStyle(Brand.amber)

            VStack(spacing: 6) {
                Text("No matches for “\(submitted ?? "")”")
                    .font(.headline)
                    .foregroundStyle(Brand.text)
                    .multilineTextAlignment(.center)
                Text("Try fewer words — the registry files a lot of near-names.")
                    .font(.subheadline)
                    .foregroundStyle(Brand.textMuted)
                    .multilineTextAlignment(.center)
            }

            Button { adding = true } label: {
                Text("TELL US MORE")
                    .font(.system(size: 12, weight: .bold))
                    .tracking(1.6)
                    .foregroundStyle(Brand.ink)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 12)
                    .background(Capsule().fill(Brand.amber))
            }
            .buttonStyle(.plain)
            .padding(.top, 2)
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Brand.surface)
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
        results = await Asked.asking { try await env.api.searchProducts(trimmed) }
        // Recorded whatever came back: "no matches" is a thing you may well want to try again
        // later, spelled differently, and hiding it makes the history a record of the app's
        // successes rather than of what you were after.
        env.recents.record(trimmed)
        recents = env.recents.all()
    }
}
