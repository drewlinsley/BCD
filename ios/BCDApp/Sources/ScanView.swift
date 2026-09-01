import SwiftUI
import Combine
import BCDKit
#if canImport(VisionKit) && os(iOS)
import VisionKit
import AVFoundation
#endif

// The camera HUD — the whole product thesis in one screen. It is fully live: point the phone at a
// shelf and detections become overlays anchored to their boxes, color-coded by predicted
// enjoyment, refreshed on a fixed cadence. No shutter, no freeze, no tap-to-scan. A persistent
// chat bar applies a natural-language filter ("nothing over 6%") to whatever is currently in frame.

struct ScanView: View {
    @EnvironmentObject var env: AppEnvironment
    @StateObject private var model = ScanViewModel()
    @State private var ask: String = ""
    @State private var selected: ScoredCandidate?

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraLayer(engine: model.engine)  // live DataScanner on device; gradient in Sim/host
                .ignoresSafeArea()

            GeometryReader { geo in
                ForEach(model.overlays) { overlay in
                    OverlayChip(candidate: overlay.candidate,
                                reaction: env.reactions
                                    .reaction(for: overlay.candidate.resolved.product.id))
                        .position(x: overlay.anchor.x * geo.size.width,
                                  y: overlay.anchor.y * geo.size.height)
                        .onTapGesture { selected = overlay.candidate }  // optional: open the detail receipt
                }
            }
            // Ease overlays in/out as the fixed-rate loop swaps the set each tick.
            .animation(.easeInOut(duration: 0.2), value: model.overlays.count)

            VStack(spacing: 12) {
                statusPill
                chatBar
            }
            .padding()
        }
        .task { model.configure(env: env); model.startLive() }
        .onDisappear { model.stop() }
        .sheet(item: $selected) { cand in
            ProductDetailView(candidate: cand)
        }
    }

    // A one-line status: on-device interpretation, an active filter, or the live scan state.
    @ViewBuilder private var statusPill: some View {
        if model.isInterpreting {
            pill("Reading with Apple Intelligence…", system: "sparkles")
        } else if let f = model.filterText {
            let n = model.overlays.count
            pill(n == 0 ? "None in view match “\(f)”" : "\(n) match “\(f)”",
                 system: "line.3.horizontal.decrease.circle")
        } else {
            // Live, fixed-rate: overlays refresh on their own, so no per-tick "Analyzing…" strobe.
            let n = model.overlays.count
            pill(n == 0 ? "Point at a shelf · scanning live" : "\(n) in view · live",
                 system: "dot.radiowaves.left.and.right")
        }
    }

    private func pill(_ text: String, system: String) -> some View {
        Label(text, systemImage: system)
            .font(.caption).foregroundStyle(.white)
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
    }

    // Persistent, always-on. Typing an ask sets a live filter over the in-frame items; clearing it
    // returns to the full set. No effect on the scan loop itself — the HUD keeps resolving live.
    private var chatBar: some View {
        HStack {
            Image(systemName: "sparkles")
            TextField("cheapest hazy here · nothing over 6%", text: $ask)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .onSubmit { Task { await model.applyFilter(ask) } }
            if !ask.isEmpty {
                Button {
                    ask = ""
                    Task { await model.clearFilter() }
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

/// One anchored overlay: a candidate + where to draw it (normalized 0-1).
struct HUDOverlay: Identifiable {
    let id: String
    let candidate: ScoredCandidate
    let anchor: CGPoint
}

struct OverlayChip: View {
    let candidate: ScoredCandidate
    /// This install's own verdict, if it has one — the same five-level scale, shown back.
    var reaction: Reaction?

    // Match score rides the reaction ramp so the HUD has one good-to-bad colour language
    // rather than two competing ones.
    private var tint: Color {
        guard let s = candidate.personalScore else { return Brand.reactionRest }
        return s > 0.75 ? Reaction.chuggedIt.tint
             : (s > 0.5 ? Reaction.fine.tint : Reaction.pouredItOut.tint)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(candidate.resolved.product.name).font(.subheadline.bold()).lineLimit(1)
            HStack(spacing: 6) {
                if let s = candidate.personalScore {
                    Label("\(Int(s * 100))", systemImage: "hand.thumbsup.fill").font(.caption2)
                }
                if candidate.coldStart {
                    Image(systemName: "flask.fill").font(.caption2)  // scored from chemistry
                }
                if let reaction {
                    Divider().frame(height: 10)
                    ReactionGlyph(reaction: reaction, size: 22)  // 22 is the glyph floor
                    Text("you").font(.caption2).foregroundStyle(Brand.textMuted)
                }
            }
            if let reason = candidate.reason {
                Text(reason).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(tint, lineWidth: 2))
        .frame(maxWidth: 180)
    }
}

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var overlays: [HUDOverlay] = []
    @Published var lastLatencyMs: Double?
    @Published var isResolving = false
    @Published var isInterpreting = false
    /// The active natural-language filter (nil = none), mirrored for the status pill.
    @Published var filterText: String?
    /// The engine the coordinator consumes. Exposed so the camera layer can present *this*
    /// engine's scanner view — it must be the same instance, or detections wouldn't reach the HUD.
    @Published private(set) var engine: ScanEngine?

    private var coordinator: ScanCoordinator?
    private var env: AppEnvironment?

    func configure(env: AppEnvironment) {
        guard coordinator == nil else { return }
        self.env = env
        let engine = env.makeScanEngine()
        self.engine = engine
        let coord = ScanCoordinator(engine: engine, api: env.api, telemetry: env.telemetry,
                                    llm: env.llm)
        self.coordinator = coord
        // Mirror the coordinator's box-anchored overlays straight into the view.
        coord.$overlays
            .map { ovs in
                ovs.map { HUDOverlay(id: $0.id, candidate: $0.candidate,
                                     anchor: CGPoint(x: $0.x, y: $0.y)) }
            }
            .assign(to: &$overlays)
        coord.$lastLatencyMs.assign(to: &$lastLatencyMs)
        coord.$isResolving.assign(to: &$isResolving)
        coord.$isInterpreting.assign(to: &$isInterpreting)
        coord.$filterText.assign(to: &$filterText)
    }

    /// Fixed-rate live mode: the viewfinder re-resolves the latest frame on a cadence and swaps
    /// overlays in place — no tapping, no accumulation. This is the entire scan interaction.
    func startLive() { coordinator?.startLive() }
    func stop() { coordinator?.stop() }

    /// Chat-bar filter: parse the ask once and apply it to every live tick.
    func applyFilter(_ ask: String) async { await coordinator?.setFilter(ask) }
    func clearFilter() async { await coordinator?.clearFilter() }
}

/// Camera layer. On a real device it presents VisionKit's `DataScannerViewController` (live
/// text + barcode) driven by the shared engine; in the Simulator or on the host — no camera —
/// it falls back to a neutral gradient so the HUD stays previewable.
struct CameraLayer: View {
    var engine: ScanEngine?

    var body: some View {
        #if canImport(VisionKit) && os(iOS)
        if #available(iOS 18.0, *), DataScannerViewController.isSupported,
           let vk = engine as? VisionKitScanEngine {
            DataScannerView(engine: vk)
        } else {
            placeholder
        }
        #else
        placeholder
        #endif
    }

    private var placeholder: some View {
        LinearGradient(colors: [.black, .gray.opacity(0.6)],
                       startPoint: .top, endPoint: .bottom)
    }
}

#if canImport(VisionKit) && os(iOS)
/// Presents the engine's `DataScannerViewController` and drives its lifecycle: request camera
/// access (this is what makes iOS show the permission prompt), then start scanning. Detections
/// flow out through the engine's delegate → `frames` → the coordinator, which the HUD renders.
@available(iOS 18.0, *)
struct DataScannerView: UIViewControllerRepresentable {
    let engine: VisionKitScanEngine

    func makeUIViewController(context: Context) -> DataScannerViewController {
        engine.makeScanner()
    }

    func updateUIViewController(_ scanner: DataScannerViewController, context: Context) {
        guard !context.coordinator.started else { return }
        context.coordinator.started = true
        let engine = self.engine  // @unchecked Sendable — safe to hand to the async closure
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            Task { await engine.start() }  // hops to @MainActor, starts the created scanner
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var started = false }
}
#endif
