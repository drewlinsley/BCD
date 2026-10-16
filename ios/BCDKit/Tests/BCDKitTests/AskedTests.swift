import Testing
import Foundation
@testable import BCDKit

// A failure is not an answer. These are the three states kept apart, and the rule for which
// failures are worth asking about twice.

/// A call counter the async closures can share. `attempts` is the whole point of several of
/// these tests, so it has to be observable from outside the closure.
private final class Calls: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func bump() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
    var count: Int { lock.lock(); defer { lock.unlock() }; return n }
}

@Suite struct AskedState {
    @Test func anEmptyAnswerIsAnAnswer() async {
        // The regression this type exists for: no neighbours is a fact about the catalog,
        // and it must not be the same value a dropped connection produces.
        let asked = await Asked<[Int]>.asking(pause: .zero) { [] }
        #expect(asked == .answered([]))
        #expect(asked != .failed)
        #expect(asked.answer == [])
        #expect(!asked.hasFailed)
    }

    @Test func waitingAndFailedHoldNoAnswer() {
        #expect(Asked<Int>.waiting.answer == nil)
        #expect(Asked<Int>.failed.answer == nil)
        #expect(Asked<Int>.failed.hasFailed)
        #expect(!Asked<Int>.waiting.hasFailed)
    }
}

@Suite struct AskingAgain {
    @Test func oneSuccessIsOneCall() async {
        let calls = Calls()
        let asked = await Asked<Int>.asking(pause: .zero) { calls.bump() }
        #expect(asked == .answered(1))
        #expect(calls.count == 1)
    }

    @Test func aServerErrorIsAskedAgainAndTheSecondAnswerStands() async {
        let calls = Calls()
        let asked = await Asked<String>.asking(pause: .zero) {
            if calls.bump() == 1 { throw APIError.http(503) }
            return "Lagavulin 16"
        }
        #expect(asked == .answered("Lagavulin 16"))
        #expect(calls.count == 2)
    }

    @Test func attemptsAreSpentAndThenItFails() async {
        let calls = Calls()
        let asked = await Asked<Int>.asking(attempts: 3, pause: .zero) {
            _ = calls.bump()
            throw APIError.http(500)
        }
        #expect(asked == .failed)
        #expect(calls.count == 3)
    }

    @Test func aMissingRowIsAnAnswerAndIsNotAskedTwice() async {
        // A 404 means there is no such product. Asking again gets the same 404 more slowly.
        let calls = Calls()
        let asked = await Asked<Int>.asking(pause: .zero) {
            _ = calls.bump()
            throw APIError.http(404)
        }
        #expect(asked == .failed)
        #expect(calls.count == 1)
    }

    @Test func aBodyWeCannotDecodeFailsOnceAndLoudly() async {
        // Our bug, not the network's: not retried, but still a failure rather than an empty
        // answer, because an empty answer is what the screen used to draw for it.
        let calls = Calls()
        let asked = await Asked<Int>.asking(pause: .zero) {
            _ = calls.bump()
            throw APIError.decoding("evidence: no such tier")
        }
        #expect(asked == .failed)
        #expect(calls.count == 1)
    }

    @Test func aDroppedConnectionIsAskedAgain() async {
        let calls = Calls()
        let asked = await Asked<Int>.asking(pause: .zero) {
            if calls.bump() == 1 { throw URLError(.networkConnectionLost) }
            return 6
        }
        #expect(asked == .answered(6))
    }

    @Test func cancellationIsNotAFailure() async {
        // The sheet closed, or the reader moved to another drink. Nothing was answered and
        // nothing was refused, so the screen must not draw a failure.
        let asked = await Asked<Int>.asking(pause: .zero) { throw CancellationError() }
        #expect(asked == .waiting)
        let urlCancel = await Asked<Int>.asking(pause: .zero) { throw URLError(.cancelled) }
        #expect(urlCancel == .waiting)
    }

    @Test func blipsAreAskedAgainAndAnswersAreNot() {
        for code in [500, 502, 503, 504, 408, 429] {
            #expect(Asking.again(after: APIError.http(code)), "\(code) is a blip")
        }
        for code in [400, 401, 403, 404, 422] {
            #expect(!Asking.again(after: APIError.http(code)), "\(code) is an answer")
        }
        for code in [URLError.Code.timedOut, .notConnectedToInternet, .networkConnectionLost,
                     .cannotConnectToHost, .dnsLookupFailed] {
            #expect(Asking.again(after: URLError(code)), "\(code) is a blip")
        }
        #expect(!Asking.again(after: URLError(.cancelled)))
        #expect(!Asking.again(after: APIError.badURL))
    }
}
