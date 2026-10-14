import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public protocol APIClientProtocol: Sendable {
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse
    func resolveVision(_ req: ScanVisionRequest) async throws -> ScanVisionResponse
    func searchProducts(_ query: String) async throws -> [ResolvedProduct]
    func sendTelemetry(_ batch: TelemetryBatch) async throws
    func submitFeedback(_ req: FeedbackRequest, userId: String) async throws -> FeedbackResponse
    /// Take one verdict back. The profile comes back rebuilt without it, so the caller does
    /// not have to ask a second time to know where it left them.
    func withdrawFeedback(productId: String) async throws -> FeedbackResponse
    /// The first-run quiz's questions. Served rather than built in, so the drinks can change
    /// without an app release.
    func quizDrinks() async throws -> [QuizDrink]
    /// Answer it, and get the profile it built — the first one that is actually theirs.
    func submitQuiz(_ answers: [QuizAnswer]) async throws -> TasteProfile
    /// Drinks to suggest, best first, for the person this client speaks for.
    func recommend(limit: Int) async throws -> [Recommendation]
    /// Catalog vocabulary for the on-device recognizer's custom-words hint.
    func fetchLexicon() async throws -> [String]
    /// What else tastes like one product. About the bottle, not about you.
    func similar(to productId: String, limit: Int) async throws -> SimilarResponse
    /// How likely this drinker is to like one drink — the recommender's own arithmetic, asked
    /// of a row that did not arrive through the recommender.
    func productScore(for productId: String) async throws -> PersonalScore
    /// What the server has learned about the person this client speaks for.
    func profile() async throws -> TasteProfile
    /// Suggestions shelf by shelf. `crossStyle` ranks the shelves they have never rated on by
    /// the taste they built elsewhere; off, those shelves come back unranked and say so.
    func familyPicks(limit: Int, crossStyle: Bool) async throws -> FamilyResponse
    /// A drink the catalog could not place, as the drinker described it. The one call that sends
    /// authored data rather than asking for data.
    func contribute(_ contribution: DrinkContribution) async throws -> ContributionAck
}

extension APIClientProtocol {
    /// Defaulted so stubs and previews only implement what they exercise. The live client
    /// overrides it; anything else reports the route as unimplemented rather than
    /// pretending a verdict was recorded.
    public func submitFeedback(_ req: FeedbackRequest,
                               userId: String) async throws -> FeedbackResponse {
        throw APIError.http(501)
    }

    /// Same reasoning. A stub that cannot withdraw must say so rather than report success,
    /// because the caller clears its own copy of the verdict on the strength of this returning.
    public func withdrawFeedback(productId: String) async throws -> FeedbackResponse {
        throw APIError.http(501)
    }

    /// Same again. A stub that answers the quiz with an empty list would show a first-run
    /// screen with no questions on it, which reads as the quiz being over.
    public func quizDrinks() async throws -> [QuizDrink] { throw APIError.http(501) }

    public func submitQuiz(_ answers: [QuizAnswer]) async throws -> TasteProfile {
        throw APIError.http(501)
    }

    /// Same reasoning, and one more: the vision path is an addition to the scan, so a stub
    /// that never exercises it reports the route as unimplemented rather than pretending the
    /// camera frame went nowhere useful.
    public func resolveVision(_ req: ScanVisionRequest) async throws -> ScanVisionResponse {
        throw APIError.http(501)
    }

    /// Same reasoning again: a stub that never asks for recommendations reports the route as
    /// unimplemented rather than returning an empty list, which a caller would read as "the
    /// server has nothing to suggest".
    public func recommend(limit: Int) async throws -> [Recommendation] {
        throw APIError.http(501)
    }

    /// No vocabulary is a fine answer: the recognizer falls back to its own dictionary.
    public func fetchLexicon() async throws -> [String] { [] }

    /// Reports the route as unimplemented rather than acknowledging something it never sent —
    /// an ack here would tell `ContributionUploader` to delete what the drinker typed.
    public func contribute(_ contribution: DrinkContribution) async throws -> ContributionAck {
        throw APIError.http(501)
    }

    /// Nothing to compare against is a fine answer too, and the same one the server gives for
    /// the 95% of rows carrying their style's average: show no section rather than an error.
    public func similar(to productId: String, limit: Int) async throws -> SimilarResponse {
        SimilarResponse(basis: .styleOnly, results: [])
    }

    /// Unscored, which is what a stub honestly is and what the screen already draws. Not a
    /// 501: the seal has a state for having nothing to say, and reaching it by the same path
    /// as a real "no profile yet" is what a stub should exercise.
    public func productScore(for productId: String) async throws -> PersonalScore {
        PersonalScore(productId: productId, scored: false, basis: "no_profile")
    }

    /// A profile that has learned nothing, which is also what a real fresh install gets back.
    /// Stubs therefore exercise the empty state rather than a fabricated one.
    public func profile() async throws -> TasteProfile {
        TasteProfile(userId: "", version: 0)
    }

    /// No shelves. A stub exercises the empty state rather than a fabricated shelf.
    public func familyPicks(limit: Int = 6, crossStyle: Bool = false) async throws
        -> FamilyResponse { FamilyResponse() }
}

public enum APIError: Error, Sendable {
    case badURL
    case http(Int)
    case decoding(String)
}

/// Talks to the FastAPI backend. A `URLSession` seam keeps it unit-testable without a
/// live server (see MockURLProtocol in the tests).
public final class APIClient: APIClientProtocol, @unchecked Sendable {
    private let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    /// Where the bearer token comes from. A closure rather than the `AuthStore` itself so this
    /// stays a plain value type over `URLSession` and tests can hand it a constant.
    private let bearer: @Sendable () async throws -> String?

    /// The client no longer carries an identity it picked. It used to send `user_id` -- an id
    /// of its own choosing, which the server believed -- so every profile was readable and
    /// writable by anyone who knew one. The server mints the identity now and this presents it.
    public init(baseURL: URL, session: URLSession = .shared,
                bearer: @escaping @Sendable () async throws -> String? = { nil }) {
        self.baseURL = baseURL
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
        self.bearer = bearer
    }

    /// Convenience for the composition root: take the token from an `AuthStore`.
    public convenience init(baseURL: URL, auth: AuthStore, session: URLSession = .shared) {
        self.init(baseURL: baseURL, session: session, bearer: { try await auth.token() })
    }

    /// Scoring is personal, and the server reads who is asking from the bearer token. It used
    /// to read it from a `user_id` the client chose, which is why anyone could score as anyone.
    public func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        try await post("/v1/scan/resolve", body: req)
    }

    /// How long to wait for a picture to be read. `URLSession`'s 60s default is sized for a
    /// request, not for inference: a model running on the developer's own machine has no GPU
    /// acceleration on an Intel Mac and takes as long as it takes. Sixty seconds would fail the
    /// call just before the answer arrived, and the failure would look like the model finding
    /// nothing — the one confusion this whole path was built to remove.
    static let visionTimeout: TimeInterval = 180

    /// A camera frame for the labels OCR cannot read. Same caller as the text path:
    /// what comes back is scored for the same person.
    public func resolveVision(_ req: ScanVisionRequest) async throws -> ScanVisionResponse {
        try await post("/v1/scan/vision", body: req,
                       timeout: Self.visionTimeout)
    }

    public func fetchLexicon() async throws -> [String] {
        var comps = URLComponents(url: baseURL.appendingPathComponent("/v1/lexicon"),
                                  resolvingAgainstBaseURL: false)
        comps?.queryItems = [URLQueryItem(name: "limit", value: "5000")]
        guard let url = comps?.url else { throw APIError.badURL }
        let (data, resp) = try await session.data(from: url)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw APIError.http((resp as? HTTPURLResponse)?.statusCode ?? -1)
        }
        return try decoder.decode(LexiconResponse.self, from: data).words
    }

    public func searchProducts(_ query: String) async throws -> [ResolvedProduct] {
        guard var comps = URLComponents(url: baseURL.appendingPathComponent("/v1/product/search"),
                                        resolvingAgainstBaseURL: false) else {
            throw APIError.badURL
        }
        comps.queryItems = [URLQueryItem(name: "q", value: query)]
        guard let url = comps.url else { throw APIError.badURL }
        let (data, resp) = try await session.data(from: url)
        try Self.check(resp)
        struct Wrapper: Codable { let results: [ResolvedProduct] }
        do {
            return try decoder.decode(Wrapper.self, from: data).results
        } catch {
            throw APIError.decoding("\(error)")
        }
    }

    public func sendTelemetry(_ batch: TelemetryBatch) async throws {
        let _: EmptyAck = try await post("/v1/telemetry", body: batch)
    }

    /// A taste verdict. Which profile it folds into is decided by the bearer token, not by
    /// `userId` — that argument is kept for the call sites and is no longer sent, because a
    /// caller-supplied id is exactly what let anyone write to anyone's profile.
    public func submitFeedback(_ req: FeedbackRequest,
                               userId: String) async throws -> FeedbackResponse {
        try await post("/v1/feedback", body: req,
                       query: [])
    }

    /// A POST rather than a DELETE on the rating's own path, because a product id carries a
    /// colon (`bcd:the-alchemist-crusher`) and a body needs no escaping to survive the trip.
    public func withdrawFeedback(productId: String) async throws -> FeedbackResponse {
        try await post("/v1/feedback/withdraw", body: WithdrawBody(productId: productId))
    }

    public func quizDrinks() async throws -> [QuizDrink] {
        let got: QuizDrinks = try await get("v1/taste/quiz")
        return got.drinks
    }

    public func submitQuiz(_ answers: [QuizAnswer]) async throws -> TasteProfile {
        try await post("/v1/taste/quiz", body: QuizSubmission(answers: answers))
    }

    /// What to drink next, for whoever the bearer token says is asking. The server answers
    /// with their learned profile once they have rated anything, and with its seed profile
    /// before that — so a fresh install gets a real ranking rather than an empty screen, and
    /// the caller is the one that has to say which of those the user is looking at.
    public func recommend(limit: Int = 12) async throws -> [Recommendation] {
        let resp: RecommendResponse = try await post(
            "/v1/recommend", body: EmptyBody(),
            query: [URLQueryItem(name: "limit", value: String(limit))])
        return resp.results
    }

    /// A drink nothing could place. It lands in bronze as a *claim* about a drink, not as a
    /// catalog row — so this call never makes something the scanner can draw, and the ack says
    /// only that the claim is recorded.
    public func contribute(_ contribution: DrinkContribution) async throws -> ContributionAck {
        try await post("/v1/contribute", body: ContributionWire(contribution))
    }

    public func similar(to productId: String, limit: Int = 6) async throws -> SimilarResponse {
        // The id goes in as its own path component and is NOT pre-encoded. Ids carry colons
        // ("ttb:20315001000326", "off:8000040000802"), and `appendingPathComponent` escapes
        // them itself -- percent-encoding first gets the `%` escaped in turn, which sent
        // `ttb%253A94033604` and came back 404.
        guard var comps = URLComponents(
            url: baseURL.appendingPathComponent("v1/product")
                .appendingPathComponent(productId)
                .appendingPathComponent("similar"),
            resolvingAgainstBaseURL: false) else { throw APIError.badURL }
        comps.queryItems = [URLQueryItem(name: "limit", value: String(limit))]
        guard let url = comps.url else { throw APIError.badURL }
        let (data, resp) = try await session.data(from: url)
        try Self.check(resp)
        do {
            return try decoder.decode(SimilarResponse.self, from: data)
        } catch {
            throw APIError.decoding("\(error)")
        }
    }

    /// The drinker's own taste, as the server currently understands it — theirs because the
    /// token says so, not because the client asked for that id.
    ///
    /// A fresh install gets `version: 0` and empty fields back — a real answer meaning "nothing
    /// learned yet", not an error — so the caller must not read an empty profile as a failure.
    public func profile() async throws -> TasteProfile {
        try await get("v1/profile")
    }

    /// Scoring one row for the caller. Authorized, unlike `similar`, because this one IS about
    /// the reader — the server reads who is asking from the bearer token.
    public func productScore(for productId: String) async throws -> PersonalScore {
        // Built component by component for the same reason `similar` is: ids carry colons and
        // pre-encoding gets the `%` escaped in turn.
        let url = baseURL.appendingPathComponent("v1/product")
            .appendingPathComponent(productId)
            .appendingPathComponent("score")
        let request = try await authorized(url, method: "GET")
        let (data, resp) = try await session.data(for: request)
        try Self.check(resp)
        do {
            return try decoder.decode(PersonalScore.self, from: data)
        } catch {
            throw APIError.decoding("\(error)")
        }
    }

    /// Suggestions shelf by shelf, rated-in shelves first. One call rather than one per shelf:
    /// the server fetches all twenty-two concurrently (about a second) and a request each would
    /// fill the screen in visibly.
    public func familyPicks(limit: Int = 6, crossStyle: Bool = false) async throws
        -> FamilyResponse {
        try await get("v1/recommend/families", query: [
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "cross_style", value: crossStyle ? "true" : "false")])
    }

    // MARK: - plumbing

    /// Every request goes out through here, so the token cannot be forgotten on one route.
    private func authorized(_ url: URL, method: String) async throws -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let token = try await bearer() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func url(_ path: String, query: [URLQueryItem] = []) throws -> URL {
        let base = baseURL.appendingPathComponent(path)
        guard !query.isEmpty else { return base }
        guard var comps = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            throw APIError.badURL
        }
        comps.queryItems = query
        guard let built = comps.url else { throw APIError.badURL }
        return built
    }

    private func get<R: Decodable>(_ path: String, query: [URLQueryItem] = [],
                                   timeout: TimeInterval? = nil) async throws -> R {
        var request = try await authorized(try url(path, query: query), method: "GET")
        if let timeout { request.timeoutInterval = timeout }
        let (data, resp) = try await session.data(for: request)
        try Self.check(resp)
        do {
            return try decoder.decode(R.self, from: data)
        } catch {
            throw APIError.decoding("\(error)")
        }
    }

    private func post<B: Encodable, R: Decodable>(
        _ path: String, body: B, query: [URLQueryItem] = [], timeout: TimeInterval? = nil
    ) async throws -> R {
        var url = baseURL.appendingPathComponent(path)
        if !query.isEmpty {
            guard var comps = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
                throw APIError.badURL
            }
            comps.queryItems = query
            guard let built = comps.url else { throw APIError.badURL }
            url = built
        }
        var request = try await authorized(url, method: "POST")
        if let timeout { request.timeoutInterval = timeout }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        let (data, resp) = try await session.data(for: request)
        try Self.check(resp)
        do {
            return try decoder.decode(R.self, from: data)
        } catch {
            throw APIError.decoding("\(error)")
        }
    }

    private static func check(_ resp: URLResponse) throws {
        guard let http = resp as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.http(http.statusCode)
        }
    }
}

struct EmptyAck: Codable {}

/// `/v1/recommend` takes its arguments in the query string and no body, but it is a
/// POST, and `post` always sends one.
struct EmptyBody: Codable {}
