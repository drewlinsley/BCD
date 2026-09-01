import SwiftUI
import BCDKit

// Weekly evolution card + consent controls. The card shows the taste memo and 3
// falsifiable predictions; confirming/denying each is a training label (the retention
// hook doubles as the label generator).

struct ProfileView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var consent: ConsentStore
    @State private var delta = WeeklyProfileDelta.demo

    var body: some View {
        NavigationStack {
            List {
                Section("Your week") {
                    Text(delta.summary).font(.callout)
                    ForEach(delta.predictions) { p in
                        PredictionRow(prediction: p)
                    }
                }
                Section("Privacy") {
                    Toggle("Analytics", isOn: $consent.analytics)
                    Toggle("Personalization", isOn: $consent.personalization)
                    Toggle("Data sharing (ads & insights)", isOn: $consent.dataSharing)
                    Text("Personalization is the tier a reaction needs: with it off, your "
                         + "verdicts stay on this phone and no profile is built.")
                        .font(.caption).foregroundStyle(Brand.textMuted)
                    Text("Each tier is a separate opt-in. Raw camera frames are never uploaded.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    Button("Export my data") {}
                    Button("Delete my data", role: .destructive) {}
                }
            }
            .navigationTitle("You")
            .task {
                try? await env.telemetry.log("weekly_profile_delta_shown",
                    tier: .personalization,
                    ["from_version": .int(delta.fromVersion),
                     "to_version": .int(delta.toVersion),
                     "n_predictions": .int(delta.predictions.count)])
            }
        }
    }
}

struct PredictionRow: View {
    let prediction: WeeklyPrediction
    var body: some View {
        HStack {
            Image(systemName: "target").foregroundStyle(Brand.amber)
            VStack(alignment: .leading) {
                Text(prediction.text)
                Text("\(Int(prediction.confidence * 100))% sure").font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "checkmark.circle").foregroundStyle(.green)
            Image(systemName: "xmark.circle").foregroundStyle(.red)
        }
    }
}

extension WeeklyProfileDelta {
    static let demo = WeeklyProfileDelta(
        userId: "demo", fromVersion: 3, toVersion: 4,
        summary: "You leaned hoppier this week and tried two new sours. Your ABV comfort "
               + "band widened to 5–9%. Bourbon curiosity is rising.",
        predictions: [
            .init(text: "You'll rate a hazy IPA above 4.0", kind: "style",
                  confidence: 0.72, resolved: nil),
            .init(text: "You'll try a barrel-aged stout", kind: "style",
                  confidence: 0.55, resolved: nil),
            .init(text: "You'll pass on anything under 4% ABV", kind: "abv",
                  confidence: 0.6, resolved: nil),
        ])
}
