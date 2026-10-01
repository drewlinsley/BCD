import Foundation
#if canImport(Security)
import Security
#endif

// Who this app is, to the server.
//
// The app used to identify itself with an id it chose and put in a query string. The server
// believed it, which meant anyone could read or write anyone's profile by typing their id. Now
// the server mints the identity and the app proves it with a bearer token.
//
// A token is a credential, so it lives in the Keychain and not in `UserDefaults`.

/// Where the bearer token is kept. A protocol so tests do not touch the real Keychain and so a
/// host build (which has no entitlement to one) still runs.
public protocol TokenStorage: Sendable {
    func read() -> String?
    func write(_ token: String)
    func clear()
}

/// The real one. `kSecAttrAccessibleAfterFirstUnlock` rather than the default: the app refreshes
/// recommendations in the background, and a token readable only while unlocked would make that
/// fail in a way that looks like a server outage.
public struct KeychainTokenStorage: TokenStorage {
    private let account: String

    public init(account: String = "bcd.session") { self.account = account }

    #if canImport(Security)
    private var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: "com.jamespscott.bcd",
         kSecAttrAccount as String: account]
    }

    public func read() -> String? {
        var out: CFTypeRef?
        let status = SecItemCopyMatching(
            query.merging([kSecReturnData as String: true,
                           kSecMatchLimit as String: kSecMatchLimitOne]) { _, new in new }
                as CFDictionary, &out)
        guard status == errSecSuccess, let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    public func write(_ token: String) {
        let data = Data(token.utf8)
        // Delete-then-add rather than update: an update that matches nothing is a silent no-op,
        // and a token that failed to save looks exactly like a token that saved until relaunch.
        SecItemDelete(query as CFDictionary)
        SecItemAdd(query.merging([
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]) { _, new in new } as CFDictionary, nil)
    }

    public func clear() { SecItemDelete(query as CFDictionary) }
    #else
    public func read() -> String? { nil }
    public func write(_ token: String) {}
    public func clear() {}
    #endif
}

/// In-memory, for tests and previews.
public final class MemoryTokenStorage: TokenStorage, @unchecked Sendable {
    private let lock = NSLock()
    private var token: String?
    public init(_ token: String? = nil) { self.token = token }
    public func read() -> String? { lock.lock(); defer { lock.unlock() }; return token }
    public func write(_ t: String) { lock.lock(); defer { lock.unlock() }; token = t }
    public func clear() { lock.lock(); defer { lock.unlock() }; token = nil }
}

/// A provider identity, obtained by whatever can actually show a sign-in sheet.
///
/// BCDKit declares the seam and the app supplies it, the same way `LLMProvider` is declared here
/// and `FoundationModelsProvider` lives in the composition root. Google's flow needs a window to
/// present from; this layer must stay buildable on a host with no windows at all.
public protocol IdentityProvider: Sendable {
    /// The provider's name as the server's route spells it: `google`, `apple`.
    var name: String { get }
    /// An ID token for the signed-in person, or throw if they cancelled.
    func idToken() async throws -> String
}

public enum AuthError: Error, Sendable {
    case noToken
    case signInFailed(String)
    case cancelled
}

/// Holds the session token, gets one when there is none, and trades it up when someone signs in.
///
/// An actor because the first call from any screen may be the one that bootstraps the account,
/// and three screens asking at once must not mint three accounts.
public actor AuthStore {
    public enum State: Equatable, Sendable {
        case anonymous
        case signedIn(provider: String)
    }

    private let baseURL: URL
    private let session: URLSession
    private let storage: TokenStorage
    private var cached: String?
    private var inFlight: Task<String, Error>?
    public private(set) var state: State = .anonymous

    public init(baseURL: URL, storage: TokenStorage = KeychainTokenStorage(),
                session: URLSession = .shared) {
        self.baseURL = baseURL
        self.storage = storage
        self.session = session
        self.cached = storage.read()
    }

    /// The bearer token, asking the server for an anonymous account the first time.
    ///
    /// Concurrent callers share one request. Without that, a cold launch that paints three
    /// screens at once would mint three accounts and keep the last one, silently orphaning the
    /// other two the moment they were created.
    public func token() async throws -> String {
        if let cached { return cached }
        if let inFlight { return try await inFlight.value }
        let task = Task<String, Error> { try await self.bootstrap() }
        inFlight = task
        defer { inFlight = nil }
        return try await task.value
    }

    private func bootstrap() async throws -> String {
        let answer: SignIn = try await post("v1/auth/anonymous", body: EmptyBody(), bearer: nil)
        adopt(answer)
        return answer.token
    }

    /// Sign in. The CURRENT token is sent along, so the server can claim the anonymous account
    /// this app has been rating under instead of starting the drinker over.
    @discardableResult
    public func signIn(with provider: IdentityProvider) async throws -> State {
        let idToken = try await provider.idToken()
        let current = try? await token()
        let answer: SignIn = try await post("v1/auth/\(provider.name)",
                                            body: SignInBody(id_token: idToken), bearer: current)
        adopt(answer)
        return state
    }

    /// Forget the token. The account and its ratings stay on the server; signing in again
    /// reaches the same profile, which is the entire reason to sign in.
    public func signOut() {
        storage.clear()
        cached = nil
        state = .anonymous
    }

    private func adopt(_ answer: SignIn) {
        cached = answer.token
        storage.write(answer.token)
        state = answer.provider == "anonymous" ? .anonymous : .signedIn(provider: answer.provider)
    }

    // MARK: - plumbing

    private struct EmptyBody: Encodable {}
    private struct SignInBody: Encodable { let id_token: String }
    private struct SignIn: Decodable {
        let account_id: String
        let token: String
        let provider: String
        let claimed: Bool?
    }

    private func post<B: Encodable, R: Decodable>(_ path: String, body: B,
                                                  bearer: String?) async throws -> R {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let bearer { request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization") }
        request.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await session.data(for: request)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw AuthError.signInFailed("server said \(http.statusCode)")
        }
        do {
            return try JSONDecoder().decode(R.self, from: data)
        } catch {
            throw AuthError.signInFailed("\(error)")
        }
    }
}
