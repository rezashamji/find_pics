import SwiftUI
import FindPicsCore

/// "Check a sample": the only honest way to audit a result set nobody can look through.
///
/// WHY THIS EXISTS (Reza, 10-11, on the 41,541 photos "photos of me" returned): "yes look like me but obviously i
/// cant look at all". Until a result set is checked it is just a number, and the top of the ranking is the easiest
/// part of it - so eyeing the first screen flatters the search. This shows ~25 results drawn AT RANDOM, one at a
/// time, big, with Yes / No, and turns the answers into a precision figure with its uncertainty
/// (Completeness.swift: exact Clopper-Pearson, so 25 of 25 reads as "83-100%", never "certain").
/// It is also the confidence number Reza wants the product to show.
struct SampleCheckSheet: View {
    let query: String
    let ids: [String]
    @Environment(\.dismiss) private var dismiss
    /// The random sample, fixed when the sheet opens so answers cannot be re-rolled.
    @State private var sample: [String] = []
    @State private var labels: [Bool] = []
    @State private var at = 0
    @State private var img: UIImage?

    var done: Bool { at >= sample.count }

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                if done { summary } else { question }
            }
            .padding()
            .navigationTitle(done ? "What we found" : "Is this right?")
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } } }
        }
        .task {
            if sample.isEmpty {
                sample = Array(ids.shuffled().prefix(precisionSampleSize(found: ids.count)))
                await load()
            }
        }
    }

    var question: some View {
        VStack(spacing: 12) {
            Text("“\(query)” — photo \(at + 1) of \(sample.count)")
                .font(.caption).foregroundStyle(.secondary)
            if let i = img {
                Image(uiImage: i).resizable().scaledToFit().frame(maxHeight: 420).cornerRadius(10)
            } else {
                ProgressView().frame(maxHeight: 420)
            }
            Text("Does this photo really match what you asked for?")
                .font(.subheadline).multilineTextAlignment(.center)
            HStack(spacing: 16) {
                Button { answer(false) } label: { Label("No", systemImage: "xmark").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered)
                Button { answer(true) } label: { Label("Yes", systemImage: "checkmark").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent)
            }
            Button("Stop here") { at = sample.count }.font(.footnote)
        }
    }

    var summary: some View {
        let p = precisionFromSample(labels: labels, found: ids.count)
        return VStack(spacing: 12) {
            Image(systemName: "checkmark.seal").font(.largeTitle).foregroundStyle(.tint)
            Text(precisionNote(p)).multilineTextAlignment(.center)
            if p.checked > 0, p.checked < precisionSampleSize(found: ids.count) {
                Text("Checking the rest of the \(precisionSampleSize(found: ids.count)) would narrow that range.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            Spacer()
            Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
        }
    }

    func answer(_ yes: Bool) {
        labels.append(yes)
        at += 1
        img = nil
        Task { await load() }
    }

    /// Same two-stage load as FullPhoto: the local thumbnail at once, then the sharp one. An iCloud-only original
    /// behind a bare spinner is what made tapping a photo unusable (10-11).
    func load() async {
        guard at < sample.count else { return }
        let id = sample[at]
        if let quick = await PhotoLibrary.image(id, side: 600, network: false), sample.indices.contains(at),
           sample[at] == id { img = quick }
        if let big = await PhotoLibrary.image(id, side: 1600, network: true), sample.indices.contains(at),
           sample[at] == id { img = big }
    }
}
