import Foundation

public protocol APIClientProtocol: Sendable {
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse
    func searchProducts(_ query: String) async throws -> [ResolvedProduct]
    func sendTelemetry(_ batch: TelemetryBatch) async throws
    func submitFeedback(_ req: FeedbackRequest, userId: String) async throws -> FeedbackResponse
}

extension APIClientProtocol {
    /// Defaulted so stubs and previews only implement what they exercise. The live client
    /// overrides it; anything else reports the route as unimplemented rather than
    /// pretending a verdict was recorded.
    public func submitFeedback(_ req: FeedbackRequest,
                               userId: String) async throws -> FeedbackResponse {
        throw APIError.http(501)
    }
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
    private let installId: String
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    /// `installId` is a property of the client rather than an argument to every call: it
    /// is the same pseudonymous identity for the whole session, and threading it through
    /// each request signature would only give callers a way to get it wrong.
    public init(baseURL: URL, installId: String = InstallIdentity.current,
                session: URLSession = .shared) {
        self.baseURL = baseURL
        self.installId = installId
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    /// Scoring is personal, so the scan has to say who is asking. Without `user_id` the
    /// server falls back to its seed profile and every rating the user has ever given is
    /// invisible to the number on screen.
    public func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        try await post("/v1/scan/resolve", body: req,
                       query: [URLQueryItem(name: "user_id", value: installId)])
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

    /// A taste verdict. `user_id` is the pseudonymous install id — it selects which
    /// profile the rating folds into, and is the only identity the server ever sees.
    public func submitFeedback(_ req: FeedbackRequest,
                               userId: String) async throws -> FeedbackResponse {
        try await post("/v1/feedback", body: req,
                       query: [URLQueryItem(name: "user_id", value: userId)])
    }

    // MARK: - plumbing

    private func post<B: Encodable, R: Decodable>(
        _ path: String, body: B, query: [URLQueryItem] = []
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
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
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
