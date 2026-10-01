import AuthenticationServices
import BCDKit
import CryptoKit
import Foundation

// Sign in with Google, without the Google SDK.
//
// `ASWebAuthenticationSession` plus PKCE is the whole flow, and it needs no third-party
// framework, no entitlement and no paid membership -- which is why Google is the provider that
// can ship first. The app never holds a client SECRET: a native app cannot keep one, which is
// exactly what PKCE exists to replace.
//
// BCDKit declares `IdentityProvider` and this supplies it, the same seam `LLMProvider` uses --
// the kit has to stay buildable on a host with no windows to present from.

@MainActor
final class GoogleIdentityProvider: NSObject, IdentityProvider,
                                    ASWebAuthenticationPresentationContextProviding {
    nonisolated let name = "google"

    /// The iOS OAuth client id from console.cloud.google.com. Comes from `BCDGoogleClientID` in
    /// Info.plist (set from Local.xcconfig, like the API base), so it stays out of the repo.
    private let clientID: String
    /// Google's iOS clients redirect to the client id reversed, which is a scheme this app
    /// claims in its Info.plist. Derived rather than configured twice: two places to get it
    /// wrong is one too many.
    private var redirectURI: String {
        clientID.split(separator: ".").reversed().joined(separator: ".") + ":/oauth2redirect"
    }

    init?(clientID: String? = Bundle.main.object(forInfoDictionaryKey: "BCDGoogleClientID")
            as? String) {
        guard let clientID, !clientID.isEmpty else { return nil }
        self.clientID = clientID
    }

    nonisolated func presentationAnchor(for session: ASWebAuthenticationSession)
        -> ASPresentationAnchor {
        MainActor.assumeIsolated {
            ASPresentationAnchor(windowScene: UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }.first!)
        }
    }

    func idToken() async throws -> String {
        // PKCE: a secret this app invents per attempt, sent hashed up front and in the clear at
        // the end. It is what stops an intercepted redirect being redeemable by anyone else.
        let verifier = Self.randomVerifier()
        let challenge = Self.challenge(for: verifier)

        var auth = URLComponents(string: "https://accounts.google.com/o/oauth2/v2/auth")!
        auth.queryItems = [
            .init(name: "client_id", value: clientID),
            .init(name: "redirect_uri", value: redirectURI),
            .init(name: "response_type", value: "code"),
            // `openid` is all that is wanted. The server reads `sub` and nothing else, so
            // asking for email or profile would be collecting what nobody uses.
            .init(name: "scope", value: "openid"),
            .init(name: "code_challenge", value: challenge),
            .init(name: "code_challenge_method", value: "S256"),
        ]
        let code = try await authorize(auth.url!)
        return try await exchange(code: code, verifier: verifier)
    }

    private func authorize(_ url: URL) async throws -> String {
        let scheme = String(redirectURI.split(separator: ":").first!)
        return try await withCheckedThrowingContinuation { cont in
            let session = ASWebAuthenticationSession(url: url, callbackURLScheme: scheme) {
                callback, error in
                if let error = error as? ASWebAuthenticationSessionError,
                   error.code == .canceledLogin {
                    return cont.resume(throwing: BCDKit.AuthError.cancelled)
                }
                if let error { return cont.resume(throwing: error) }
                guard let callback,
                      let code = URLComponents(url: callback, resolvingAgainstBaseURL: false)?
                        .queryItems?.first(where: { $0.name == "code" })?.value else {
                    return cont.resume(throwing: BCDKit.AuthError.signInFailed("no code came back"))
                }
                cont.resume(returning: code)
            }
            session.presentationContextProvider = self
            // Google will not reuse a Safari cookie for a native client anyway, and asking for
            // an ephemeral session makes that explicit rather than leaving the account picker
            // showing whoever last signed in on this device.
            session.prefersEphemeralWebBrowserSession = true
            session.start()
        }
    }

    private func exchange(code: String, verifier: String) async throws -> String {
        var request = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var form = URLComponents()
        form.queryItems = [
            .init(name: "client_id", value: clientID),
            .init(name: "code", value: code),
            .init(name: "code_verifier", value: verifier),
            .init(name: "grant_type", value: "authorization_code"),
            .init(name: "redirect_uri", value: redirectURI),
        ]
        request.httpBody = Data((form.percentEncodedQuery ?? "").utf8)
        let (data, _) = try await URLSession.shared.data(for: request)
        struct Token: Decodable { let id_token: String? }
        guard let token = try? JSONDecoder().decode(Token.self, from: data),
              let id = token.id_token else {
            throw BCDKit.AuthError.signInFailed("Google returned no id_token")
        }
        return id
    }

    // MARK: - PKCE

    private static func randomVerifier() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return base64url(Data(bytes))
    }

    private static func challenge(for verifier: String) -> String {
        base64url(Data(SHA256.hash(data: Data(verifier.utf8))))
    }

    /// base64url, which is base64 with two characters swapped and the padding dropped. Sending
    /// plain base64 makes Google reject the challenge with a message about the method.
    private static func base64url(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}

#if canImport(UIKit)
import UIKit
#endif
