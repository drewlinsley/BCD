import Foundation

public protocol APIClientProtocol: Sendable {
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse
    func searchProducts(_ query: String) async throws -> [ResolvedProduct]
    func sendTelemetry(_ batch: TelemetryBatch) async throws
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

    public init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    public func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        try await post("/v1/scan/resolve", body: req)
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

    // MARK: - plumbing

    private func post<B: Encodable, R: Decodable>(_ path: String, body: B) async throws -> R {
        let url = baseURL.appendingPathComponent(path)
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
