import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// One asked-for fact, and the rule for asking again.
//
// `try? await …` collapses "it failed" into `nil`, and on a screen where `nil` already means
// something the two become the same picture. The product detail screen had it twice:
//
//   similar = (try? await env.api.similar(to: product.id, limit: 6))?.results ?? []
//
// A dropped connection became an empty list, an empty list means "this drink has no
// neighbours", and the Similar profile tile vanished — which is why it seemed to come and go
// on the same drink. And:
//
//   fetchedScore = try? await env.api.productScore(for: product.id)
//
// A failure left nil, nil is also what a reader with no taste profile gets, and the seal says
// RATE MORE to a reader with no taste profile. So a 500 told people who had rated fifty drinks
// to go and rate more.
//
// Both are the same mistake: a failure rendered as an answer. Three states cannot be held in
// two, so this is the third.

/// What came back, when something was asked for over the network.
///
/// `waiting` until it answers, `answered` with whatever it said — including a legitimately
/// empty answer — and `failed` when the asking itself did not work. A view switching on this
/// cannot accidentally draw a failure as an answer, because it has to name the case.
public enum Asked<Answer: Sendable>: Sendable {
    /// Still asking, or never asked. A screen draws its loading or resting state.
    case waiting
    /// The server answered. The answer may be empty, which is a fact about the catalog.
    case answered(Answer)
    /// The asking failed, and we do not know what the answer would have been.
    case failed

    /// The answer, if there is one. For the places that only need the happy path.
    public var answer: Answer? {
        if case .answered(let answer) = self { return answer }
        return nil
    }

    public var hasFailed: Bool {
        if case .failed = self { return true }
        return false
    }
}

extension Asked: Equatable where Answer: Equatable {}

extension Asked {
    /// Ask, and ask again if the failure was the kind that might not happen twice.
    ///
    /// Returns `.answered` on success and `.failed` once the attempts are spent. A cancelled
    /// task answers `.waiting`: the view is going away or has moved to another drink, and
    /// nothing was ever answered or refused.
    ///
    /// Three attempts, 350ms then 700ms apart, because the failure this exists for is a sheet
    /// opening on a phone whose radio has not come back yet. Most of those are gone inside a
    /// second, and a reader who never sees the failed state is the point.
    /// `isolation` is the caller's actor, defaulted to wherever this is called from, so a
    /// `@MainActor` view can hand in a closure that touches the view. Without it the closure
    /// crosses an isolation boundary and has to be `Sendable`, which a closure capturing a
    /// SwiftUI `View` is not.
    public static func asking(attempts: Int = 3,
                              pause: Duration = .milliseconds(350),
                              isolation: isolated (any Actor)? = #isolation,
                              _ work: () async throws -> Answer) async -> Asked<Answer> {
        var wait = pause
        for attempt in 1...max(1, attempts) {
            do {
                return .answered(try await work())
            } catch {
                if Asking.wasCancelled(error) { return .waiting }
                guard attempt < attempts, Asking.again(after: error) else { return .failed }
                do { try await Task.sleep(for: wait) } catch { return .waiting }
                wait = wait * 2
            }
        }
        return .failed
    }
}

/// Which failures are worth a second ask.
public enum Asking {
    /// Whether to ask again.
    ///
    /// A 5xx, a timeout, a rate limit or a transport error is a blip: the thing asked for
    /// exists and the asking did not work. Any other 4xx is an answer — a 404 means there is
    /// no such row, and asking twice gets the same 404 more slowly. A body this build cannot
    /// decode is our own bug, which is not transient either; it is still a failure, and that
    /// is the part that matters, because the alternative was drawing it as an empty answer.
    public static func again(after error: Error) -> Bool {
        switch error {
        case let api as APIError:
            switch api {
            case .http(let code): return code >= 500 || code == 408 || code == 429
            case .decoding, .badURL: return false
            }
        case let url as URLError:
            switch url.code {
            case .cancelled, .badURL, .unsupportedURL, .userAuthenticationRequired: return false
            default: return true
            }
        default: return false
        }
    }

    /// Whether the task was cancelled rather than refused. Both arrive here as errors and they
    /// are not the same event: cancellation is this app deciding to stop asking.
    public static func wasCancelled(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? URLError)?.code == .cancelled
    }
}
