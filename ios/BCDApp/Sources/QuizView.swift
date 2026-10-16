import SwiftUI
import BCDKit

// The first thing a new drinker sees, and the only screen in the app whose job is to make
// the next screen possible.
//
// Without it the app cannot do the thing it is for. A drinker who has rated nothing has no
// centroid; with no centroid the server answers /v1/recommend from a seed profile, and the
// list that comes back tells someone who has never said anything that these drinks "match
// your tropical preference". The quiz is twenty seconds of saying what you reach for, and
// it is the difference between a recommendation that is theirs and one that is a stranger's.
//
// Eight questions, and which eight was measured: every drink family has a flavour centroid,
// and the familiar eight here cover 0.872 of that space against the flavour-optimal set's
// 0.875 — so asking about drinks a person actually recognises costs nothing. The server
// owns the list (`GET /v1/taste/quiz`), so it can change without an app release and this
// screen never has to know a flavour vector exists.

struct QuizView: View {
    /// How the quiz ended. "It is over" and "we never got to ask" are not the same thing, and
    /// the flag that stops the app asking twice is only entitled to the first: a quiz that
    /// never loaded has not been put to anyone. One blip on a first launch used to spend the
    /// one asking this install was ever going to get (2026-10-16).
    enum Outcome {
        /// Answers sent, profile built.
        case answered
        /// They saw the questions and chose to move on. Skip is a decision.
        case skipped
        /// The questions never arrived, so nothing was asked. Getting past this screen is not
        /// a verdict on a quiz nobody saw.
        case unreachable
    }

    @EnvironmentObject var env: AppEnvironment
    /// Answering is the moment personalization is agreed to, so this screen has to be able
    /// to set it — and to read whether it has already been given, on a retake.
    @EnvironmentObject var consent: ConsentStore
    /// Called once the profile exists, so whoever presented this can get out of the way. It
    /// is handed what happened, because the first-run caller's flag depends on it and a retake
    /// does not care.
    var onDone: (Outcome) -> Void

    /// The questions. `failed` kept apart from an empty answer: the screen said "Can't reach
    /// the catalog" for both, and -- worse -- leaving either way marked the quiz as asked.
    @State private var drinks: Asked<[QuizDrink]> = .waiting
    @State private var answers: [String: Double] = [:]
    @State private var sending = false
    @State private var failed = false
    /// That this install answered, which `bcd.quizAsked` does not say — that one covers
    /// skipping too, because its job is to not ask twice. Only the client knows this: the
    /// profile the server returns carries a centroid, not an account of where it came from.
    @AppStorage("bcd.quizAnswered") private var quizAnswered = false

    /// The answer scale lives in BCDKit (`QuizLean`), beside the rating weights it is
    /// calibrated against — the number travels to the server, so it is wire semantics and not
    /// a property of this screen. Here it only needs a colour: both positives are amber,
    /// because the word carries the degree and only one can be chosen at a time, so a second
    /// hue would be decoration that has to be learned.
    private typealias Lean = QuizLean

    var body: some View {
        NavigationStack {
            Group {
                switch drinks {
                case .waiting:
                    ProgressView().controlSize(.large)
                // Nothing to ask. `load` is already getting out of the way, so this holds the
                // spinner rather than flashing an empty form on its way past.
                case .answered(let got) where got.isEmpty:
                    ProgressView().controlSize(.large)
                case .answered(let got):
                    questions(got)
                case .failed:
                    unavailable
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Brand.surface)
            .navigationTitle("What do you drink?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    // Always escapable. A first run that cannot be got past is a worse
                    // failure than a cold start, and the drinker can answer this later from
                    // You — rating anything builds the same profile by the same path.
                    // Always `.skipped`, whatever is on screen: the word on the button is the
                    // decision. The unreachable card's own way out is the one that decides
                    // nothing, because that screen never asked anything.
                    Button("Skip") { onDone(.skipped) }.disabled(sending)
                }
            }
        }
        .task { await load() }
    }

    private func questions(_ drinks: [QuizDrink]) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("So the first thing we show you is yours, not a stranger's.")
                    .font(.subheadline)
                    .foregroundStyle(Brand.textMuted)
                    .padding(.bottom, 2)

                ForEach(drinks) { drink in
                    HStack(spacing: 10) {
                        // The name gives, never the buttons. The questions are served, so a
                        // longer drink name can arrive without a release, and when it does it
                        // must shrink rather than squeeze three controls into hyphenation.
                        Text(drink.prompt)
                            .font(.headline)
                            .foregroundStyle(Brand.text)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                        Spacer(minLength: 8)
                        ForEach(Lean.allCases, id: \.rawValue) { lean in
                            Button { pick(lean, for: drink) } label: {
                                Text(lean.label)
                                    .font(.subheadline)
                                    .lineLimit(1)
                                    .fixedSize()
                                    .padding(.horizontal, 12).padding(.vertical, 7)
                                    .background(
                                        Capsule().fill(answers[drink.family] == lean.rawValue
                                                       ? tint(lean).opacity(0.15) : .clear))
                                    .overlay(
                                        Capsule().stroke(answers[drink.family] == lean.rawValue
                                                         ? tint(lean) : Brand.hairline,
                                                         lineWidth: 1))
                                    .foregroundStyle(answers[drink.family] == lean.rawValue
                                                     ? tint(lean) : Brand.textMuted)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.vertical, 4)
                    Divider().overlay(Brand.hairline)
                }

                if failed {
                    Text("Couldn't reach the server — you can answer this later under You.")
                        .font(.caption).foregroundStyle(.orange)
                }

                consentLine

                Button { Task { await submit() } } label: {
                    HStack(spacing: 8) {
                        Text(answers.isEmpty ? "Answer one to start" : "Start")
                        if sending { ProgressView().controlSize(.mini) }
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
                    .background(Capsule().fill(answers.isEmpty ? Brand.hairline : Brand.amber))
                    .foregroundStyle(answers.isEmpty ? Brand.textMuted : Brand.ink)
                }
                .buttonStyle(.plain)
                .disabled(answers.isEmpty || sending)
                .padding(.top, 8)
            }
            .padding(20)
        }
    }

    /// What answering agrees to, said where the agreeing happens.
    ///
    /// Personalization is off on a fresh install — the two tiers that shape a profile stay
    /// off until the drinker says otherwise — and for a while this screen collected eight
    /// answers, sent them, and let the server build a centroid from them while the You tab
    /// said "Personalization is off, so nothing is being learned" (2026-10-08). The rating
    /// picker had always asked first; the quiz was the one caller that never did.
    ///
    /// Asked here rather than in a dialog of its own. An alert over a first-run screen is a
    /// modal on a modal, and "Not now" would leave eight answers given and thrown away. This
    /// screen is already explaining what the answers are for, which is the honest place to
    /// say what becomes of them — and Skip is a real way out that agrees to nothing.
    ///
    /// Gone once it has been given, because a retake is not a second asking.
    @ViewBuilder private var consentLine: some View {
        if !consent.personalization {
            // Two lines, not three: it has to sit above the button without pushing it off
            // the screen, because consent read after the tap is not consent. The fuller
            // wording lives on the picker's alert and under You.
            Text("Kept against a random id for this install \u{2014} not your name or email "
                 + "\u{2014} and you can turn it off any time under You.")
                .font(.caption)
                .foregroundStyle(Brand.textMuted)
                .padding(.top, 4)
        }
    }

    /// The server could not be asked. Not a wall: the quiz is an accelerant, not a gate.
    ///
    /// Two ways out, and they mean different things. Trying again is the one most people
    /// want -- the request has already been retried three times, but a server coming back up
    /// takes longer than a second -- and going in reports `unreachable`, so the asking is not
    /// spent on a screen that never asked.
    private var unavailable: some View {
        VStack(spacing: 12) {
            Text("Can't reach the catalog right now.")
                .font(.headline).foregroundStyle(Brand.text)
            Text("Rate anything you drink and this builds itself.")
                .font(.subheadline).foregroundStyle(Brand.textMuted)
                .multilineTextAlignment(.center)
            Button("Try again") { Task { await load() } }
                .font(.headline).foregroundStyle(Brand.amber)
                .padding(.top, 4)
            Button("Go on in") { onDone(.unreachable) }
                .font(.subheadline).foregroundStyle(Brand.textMuted)
        }
        .padding(28)
    }

    private func tint(_ lean: Lean) -> Color {
        lean == .no ? Brand.textMuted : Brand.amber
    }

    private func pick(_ lean: Lean, for drink: QuizDrink) {
        // Tapping the chosen one again clears it: changing your mind to "no opinion" has to
        // be reachable, or a mis-tap is permanent and the profile carries it.
        if answers[drink.family] == lean.rawValue {
            answers.removeValue(forKey: drink.family)
        } else {
            answers[drink.family] = lean.rawValue
        }
    }

    private func load() async {
        drinks = await Asked.asking { try await env.api.quizDrinks() }
        // An answer of no questions is still an answer, and it spends the asking: the question
        // set is the server's, and a client that treated "none" as "ask me again" would meet
        // this screen on every launch for the life of the install.
        if case .answered(let got) = drinks, got.isEmpty { onDone(.skipped) }
    }

    private func submit() async {
        sending = true
        defer { sending = false }
        // Start IS the yes. Set before anything leaves the phone, never after: the whole
        // defect was answers reaching the server ahead of the agreement to keep them. It
        // stays set if the send then fails -- consent is what the drinker said, not a
        // function of whether the network was up.
        consent.personalization = true
        let payload = answers.compactMap { family, weight in
            QuizLean(rawValue: weight)?.answer(for: family)
        }
        // The server records one `taste_quiz_answered` per answer as it builds the profile,
        // so there is nothing for the client to log here: a second copy carrying a different
        // shape would be the same quiz counted twice.
        //
        // Retried like a read, which a POST normally would not be. This one supersedes rather
        // than accumulates -- the profile takes the last weight per family -- so a send that
        // reached the server and lost its answer builds the identical centroid the second
        // time. Against that, losing a first-run drinker's eight answers to one blip is the
        // cold start this screen exists to prevent.
        switch await Asked.asking({ try await env.api.submitQuiz(payload) }) {
        case .answered:
            quizAnswered = true
            onDone(.answered)
        case .failed:
            // Kept on screen rather than dropped: these are the only answers we will get,
            // and letting them through to an empty profile would be the cold start again.
            // Start is still live, so the line is an invitation to press it again.
            failed = true
        case .waiting:
            // Cancelled -- the sheet is going away. Nothing to report and nobody to report to.
            break
        }
    }
}
