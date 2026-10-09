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
//
// "Export my data" and "Delete my data" were here too, both `Button(...) {}` with empty
// closures. Removed rather than wired: there is nowhere to export to, and a control under a
// Privacy heading that looks live and does nothing is worse than its absence -- someone who
// taps Delete and sees no error has been told their data is gone. If either comes back it
// comes back working.

struct ProfileView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var consent: ConsentStore
    /// The provenance line counts them, so it has to be told when the count changes.
    @EnvironmentObject var reactions: ReactionLog
    /// Whether the quiz was answered on this install — the other thing the card is built
    /// from, and one the profile itself does not record.
    @AppStorage("bcd.quizAnswered") private var quizAnswered = false
    /// nil until the fetch settles. `loadFailed` is separate on purpose: a profile that came
    /// back empty and a profile that never arrived look identical if you only track one.
    @State private var profile: TasteProfile?
    @State private var loadFailed = false
    /// Whether the quiz is open over this screen. The only way back to it: it is presented
    /// once on first launch and never again, and the screen it points at is this one.
    @State private var retakingQuiz = false
    @State private var authState: AuthStore.State?
    @State private var signingIn = false
    @State private var signInError: String?
    /// nil when no Google client id was built in, in which case the button is not offered.
    private let google = GoogleIdentityProvider()

    var body: some View {
        NavigationStack {
            List {
                Section("Your taste") {
                    taste
                    quiz
                }
                Section("Account") { account }
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
            }
            .navigationTitle("You")
            .task { await load(); await readAuth() }
            // The profile is rebuilt server-side from verdicts, so it can change while this
            // screen is open only if the switch above changes what may be collected. Re-read
            // on that rather than on every appearance.
            .onChange(of: consent.personalization) { _, _ in Task { await load() } }
            // A sheet, not a `fullScreenCover`. On first launch the quiz is a screen that
            // comes before the app; reached from here it is one more thing in a settings
            // list, and taking the whole screen for it would overstate it.
            //
            // `env` and `consent` are handed over explicitly, exactly as `RootView` does:
            // answering is what grants personalization, so the sheet needs the real store.
            .sheet(isPresented: $retakingQuiz) {
                QuizView { retakingQuiz = false; Task { await load() } }
                    .environmentObject(env)
                    .environmentObject(consent)
            }
        }
    }

    /// The way back to the quiz, and the only one.
    ///
    /// It is presented once, from a `fullScreenCover` gated on `bcd.quizAsked`, and that flag
    /// is set by answering OR skipping OR dismissing — so skipping it put it out of reach for
    /// the life of the install. Two pieces of copy already promised otherwise, including the
    /// one shown when the server is unreachable: "you can answer this later under You". The
    /// case where the quiz most needs a second chance was the case that promised one and had
    /// none (2026-10-08).
    ///
    /// Retaking is supported underneath without anything new: a later answer supersedes
    /// rather than accumulates, the same rule a re-rate follows.
    @ViewBuilder private var quiz: some View {
        Button { retakingQuiz = true } label: {
            HStack(spacing: 8) {
                // Not "Retake" for someone who never took it — a skipper would be being
                // asked to do again a thing they have not done.
                Text(quizAnswered ? "Retake the quiz" : "Tell us what you drink")
                Spacer(minLength: 8)
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.textMuted)
            }
            .font(.callout)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - account

    // What signing in buys is DURABILITY, not secrecy, and the copy says so. An anonymous
    // account lives in one app install: delete the app and the taste goes with it. Signing in
    // makes the same profile reachable from a reinstall or a second phone.
    @ViewBuilder private var account: some View {
        switch authState {
        case .signedIn(let provider):
            HStack {
                Label("Signed in with \(provider.capitalized)", systemImage: "checkmark.seal")
                    .font(.callout)
                Spacer()
                Button("Sign out") { Task { await env.auth.signOut(); await readAuth() } }
                    .font(.caption)
            }
            Text("Your taste is kept with this account, so it survives reinstalling the app "
                 + "and follows you to another phone.")
                .font(.caption).foregroundStyle(Brand.textMuted)
        case .anonymous, .none:
            if let google {
                Button {
                    Task { await signIn(with: google) }
                } label: {
                    HStack(spacing: 8) {
                        if signingIn { ProgressView().controlSize(.small) }
                        Text("Sign in with Google")
                    }
                }
                .disabled(signingIn)
            }
            // Sign in with Apple is written and the server verifies it. It cannot be OFFERED
            // until the Apple Developer Program membership is paid: it is a capability free
            // provisioning will not sign, so the button would fail at the tap rather than at
            // the build. Shown disabled rather than hidden, so it is not quietly forgotten.
            Button("Sign in with Apple") {}
                .disabled(true)
            Text(signInError ?? ("Your taste lives on this phone. Signing in keeps it if you "
                                 + "reinstall, and carries it to another phone."))
                .font(.caption)
                .foregroundStyle(signInError == nil ? Brand.textMuted : .red)
        }
    }

    private func signIn(with provider: GoogleIdentityProvider) async {
        signingIn = true
        signInError = nil
        defer { signingIn = false }
        do {
            _ = try await env.auth.signIn(with: provider)
            await readAuth()
            await load()            // the profile is the account's now, not the install's
        } catch BCDKit.AuthError.cancelled {
            // Backing out of a sign-in sheet is not an error and must not be reported as one.
        } catch {
            signInError = "That sign-in did not complete. Nothing changed."
        }
    }

    private func readAuth() async { authState = await env.auth.state }

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
    ///
    /// The quiz is named alongside it, because it is the other thing that builds this card and
    /// counting only ratings misdescribed it: answer eight questions, rate nothing, and the
    /// line read "Built from your 0 ratings" under a page of real leanings (2026-10-08). The
    /// third screen to make the same mistake — a quiz answer is a family, so it writes no
    /// rating, and anything that measures the drinker by `ReactionLog.count` cannot see it.
    private func provenance(_ p: TasteProfile) -> String {
        let n = reactions.count
        let sources = [quizAnswered ? "your quiz answers" : nil,
                       n > 0 ? (n == 1 ? "1 rating" : "\(n) ratings") : nil].compactMap { $0 }
        // Neither, and yet a card: not reachable today, but a profile is the server's and this
        // count is the phone's, so they can disagree after a reinstall.
        let built = sources.isEmpty ? "Built from your taste so far"
                                    : "Built from " + sources.joined(separator: " and ")
        guard let when = Self.updated(p.updatedAt) else { return built + "." }
        return "\(built) · updated \(when)."
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
                 "n_rated": .int(reactions.count)])
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
