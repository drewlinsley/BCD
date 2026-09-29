import SwiftUI
import BCDKit

// The You screen: what the app has learned about this drinker, and the consent switches that
// decide whether it may learn anything at all.
//
// "Your taste" is served by `GET /v1/profile`, keyed on the same pseudonymous install id the
// drinker's verdicts were filed under, so it is theirs and no one else's. It replaced a
// hardcoded "Your week" card whose sentence -- "you leaned hoppier this week and tried two
// new sours" -- was a literal in this file, written to match the server's *demo* seed. It was
// coherent, specific, and about nobody. A profile screen that invents the profile is worse
// than no profile screen, because the drinker has no way to tell which one they are reading.
//
// The weekly framing went with it. A profile is one current row plus a version counter; there
// is no snapshot of last week, so "this week" was a claim the data could not support. When
// history exists, predictions and a delta can come back -- `WeeklyProfileDelta` is still in
// BCDKit waiting for it.

struct ProfileView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var consent: ConsentStore
    /// nil until the fetch settles. `loadFailed` is separate on purpose: a profile that came
    /// back empty and a profile that never arrived look identical if you only track one.
    @State private var profile: TasteProfile?
    @State private var loadFailed = false

    var body: some View {
        NavigationStack {
            List {
                Section("Your taste") { taste }
                Section("Privacy") {
                    Toggle("Analytics", isOn: $consent.analytics)
                    Toggle("Personalization", isOn: $consent.personalization)
                    Toggle("Data sharing (ads & insights)", isOn: $consent.dataSharing)
                    VStack(alignment: .leading, spacing: 2) {
                        Toggle("Identify labels from a photo", isOn: $consent.labelPhotos)
                        Text("Stylized cans defeat text recognition. With this on, a photo of "
                             + "the label is sent for identification when reading it fails.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Text("Personalization is the tier a reaction needs: with it off, your "
                         + "verdicts stay on this phone and no profile is built.")
                        .font(.caption).foregroundStyle(Brand.textMuted)
                    Text("Each tier is a separate opt-in. Raw camera frames are never uploaded.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    Button("Export my data") {}
                    Button("Delete my data", role: .destructive) {}
                }
            }
            .navigationTitle("You")
            .task { await load() }
            // The profile is rebuilt server-side from verdicts, so it can change while this
            // screen is open only if the switch above changes what may be collected. Re-read
            // on that rather than on every appearance.
            .onChange(of: consent.personalization) { _, _ in Task { await load() } }
        }
    }

    // MARK: - the card

    // Order is the order of what explains what. Consent first, because with it off there is
    // nothing downstream to explain; then a profile with something in it; then the two ways
    // there can be nothing to show, which must not be conflated.
    @ViewBuilder private var taste: some View {
        if !consent.personalization {
            message("Personalization is off, so nothing is being learned. "
                    + "Your verdicts stay on this phone.")
        } else if let profile, !profile.hasLearnedNothing {
            learned(profile)
        } else if profile != nil {
            message("Nothing learned yet. Rate a few drinks and this fills in — your own "
                    + "verdicts are the only thing it is built from.")
        } else if loadFailed {
            message("Can't reach the catalog server, so your taste couldn't be read.")
        } else {
            HStack(spacing: 8) {
                ProgressView()
                Text("Reading your taste…").foregroundStyle(Brand.textMuted)
            }
        }
    }

    @ViewBuilder private func learned(_ p: TasteProfile) -> some View {
        if let memo = p.memo, !memo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            Text(memo).font(.callout)
        }

        // The notes, with a bar rather than a printed number. The value is real and worth
        // showing, but "citrus 0.44" is a column read aloud -- the bar says the same thing in
        // the units a person actually has for it, which is "more than that one".
        if !p.notes.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                heading("What you go for")
                ForEach(p.notes.prefix(5), id: \.axis) { note in
                    NoteBar(note: note.axis.note, value: note.value)
                }
            }
            .padding(.vertical, 2)
        }

        // Likes only, and twice as many of them. The negative half is still decoded and still
        // steers the ranking -- `TasteProfile.stylesAvoided` is live, and the memo above can
        // still name a note the drinker turns down. It is the *list* that goes: a column of
        // styles under the drinker's own name, each one a thing they are on record disliking,
        // is a worse thing to hand someone than the same fact working quietly in their
        // recommendations. Room freed goes to the half they want to read.
        if !p.stylesLiked.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                heading("Styles you lean into")
                ForEach(p.stylesLiked.prefix(8), id: \.style) { s in
                    StyleLine(style: s.style)
                }
            }
            .padding(.vertical, 2)
        }

        Text(provenance(p)).font(.caption).foregroundStyle(Brand.textMuted)
    }

    private func heading(_ text: String) -> some View {
        Text(text.uppercased()).font(.caption2.weight(.semibold))
            .foregroundStyle(Brand.textMuted).tracking(0.6)
    }

    private func message(_ text: String) -> some View {
        Text(text).font(.callout).foregroundStyle(Brand.textMuted)
    }

    /// Where this came from, in one line. The rating count is the local one: it is what the
    /// drinker did, and it says whether a confident-sounding memo rests on three verdicts.
    private func provenance(_ p: TasteProfile) -> String {
        let n = env.reactions.count
        let ratings = n == 1 ? "1 rating" : "\(n) ratings"
        guard let when = Self.updated(p.updatedAt) else { return "Built from your \(ratings)." }
        return "Built from your \(ratings) · updated \(when)."
    }

    private static func updated(_ iso: String?) -> String? {
        guard let iso else { return nil }
        let parser = ISO8601DateFormatter()
        parser.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = parser.date(from: iso) ?? {
            let plain = ISO8601DateFormatter()
            plain.formatOptions = [.withInternetDateTime]
            return plain.date(from: iso)
        }() else { return nil }
        let out = DateFormatter()
        out.dateFormat = "d MMM"
        return out.string(from: date)
    }

    private func load() async {
        guard consent.personalization else { return }
        do {
            let p = try await env.api.profile()
            profile = p
            loadFailed = false
            try? await env.telemetry.log("taste_profile_shown", tier: .personalization,
                ["version": .int(p.version),
                 "n_styles": .int(p.styleAffinities.count),
                 "n_notes": .int(p.notes.count),
                 "n_rated": .int(env.reactions.count)])
        } catch {
            loadFailed = true
        }
    }
}

/// One flavour note and how far into it this drinker leans.
private struct NoteBar: View {
    let note: String
    let value: Double

    var body: some View {
        HStack(spacing: 12) {
            Text(note).font(.callout)
            Spacer(minLength: 12)
            Capsule().fill(Brand.hairline)
                .frame(width: 72, height: 6)
                .overlay(alignment: .leading) {
                    Capsule().fill(Brand.amber)
                        .frame(width: 72 * min(1, max(0, value)), height: 6)
                }
        }
        // The bar is the number drawn; VoiceOver gets the number itself, since a width is
        // not something a screen reader can hand over.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(note), \(Int((value * 100).rounded()) ) out of 100")
    }
}

/// A style the drinker's verdicts pushed toward.
private struct StyleLine: View {
    let style: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.up")
                .font(.caption.weight(.bold)).foregroundStyle(Reaction.chuggedIt.tint)
            Text(style).font(.callout)
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Leaning into \(style)")
    }
}
