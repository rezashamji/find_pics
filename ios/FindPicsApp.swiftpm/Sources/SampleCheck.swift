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
    /// True when no judge looked at these results, so what is being audited is the FACE MATCHER, not the judge
    /// (cluster, M43-ANSWER: "a miss there means a wrong face, which the review screen can exclude; say that").
    var faceOnly = false
    @Environment(\.dismiss) private var dismiss
    /// The random sample, fixed when the sheet opens so answers cannot be re-rolled.
    @State private var sample: [String] = []
    /// Answers given in THIS sitting; `kept` are the ones from earlier sittings of the same query.
    @State private var labels: [Bool] = []
    @State private var kept: [Bool] = []
    @State private var at = 0
    @State private var img: UIImage?
    @State private var started = false
    /// Unanswered results not in the sample yet, to replace one that cannot be opened.
    @State private var pool: [String] = []
    /// This photo could not be read at all (its original is in iCloud and would not come down).
    @State private var stuck = false

    var done: Bool { at >= sample.count }
    /// Everything ever answered for this query, which is what the range is computed from.
    var all: [Bool] { kept + labels }

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
            if sample.isEmpty, !started {
                started = true
                // SKIP WHAT HE ALREADY ANSWERED (M43(b)). A second audit of the same search continues where the
                // last one stopped and narrows the range, instead of asking about the same photos again.
                let prev = await auditStore.answers(query)
                let answered = Set(prev.yes).union(prev.no)
                kept = Array(repeating: true, count: prev.yes.filter { ids.contains($0) }.count)
                    + Array(repeating: false, count: prev.no.filter { ids.contains($0) }.count)
                let want = max(0, precisionSampleSize(found: ids.count) - kept.count)
                var free = ids.filter { !answered.contains($0) }.shuffled()
                sample = Array(free.prefix(want))
                free.removeFirst(min(want, free.count))
                pool = free
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
            } else if stuck {
                VStack(spacing: 8) {
                    Image(systemName: "icloud.slash").font(.largeTitle).foregroundStyle(.secondary)
                    Text("This one would not come down from iCloud, so it was swapped for another.")
                        .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }.frame(maxHeight: 420)
            } else {
                ProgressView().frame(maxHeight: 420)
            }
            Text("Does this photo really match what you asked for?")
                .font(.subheadline).multilineTextAlignment(.center)
            HStack(spacing: 16) {
                Button { answer(false) } label: { Label("No", systemImage: "xmark").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered).disabled(img == nil)
                Button { answer(true) } label: { Label("Yes", systemImage: "checkmark").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).disabled(img == nil)
            }
            // ALWAYS A WAY PAST. Reza, 10-11: "i did 18 and it got stopped at 19". A photo whose original lives in
            // iCloud and will not download left the sheet on a bare spinner with nothing to tap but Close, which
            // ended the audit at 18 of 25. Skipping also pulls in a replacement, so the sample size still holds.
            Button("Skip this photo") { Task { await skip() } }.font(.footnote)
            Button("Stop here") { at = sample.count }.font(.footnote)
        }
    }

    var summary: some View {
        let p = precisionFromSample(labels: all, found: ids.count)
        return VStack(spacing: 12) {
            Image(systemName: "checkmark.seal").font(.largeTitle).foregroundStyle(.tint)
            Text(precisionNote(p)).multilineTextAlignment(.center)
            if !kept.isEmpty {
                Text("That includes \(kept.count) you had already answered for this search.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            if faceOnly {
                Text("These came from face matching, not from the AI looking at each photo, so a wrong one here "
                     + "is a wrong face — you can remove it on the Me screen and search again.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            if p.checked > 0, p.checked < ids.count {
                Text("Checking more narrows that range.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            Spacer()
            Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
        }
    }

    func answer(_ yes: Bool) {
        guard at < sample.count else { return }
        let id = sample[at]
        labels.append(yes)
        Task { await auditStore.record(query: query, id: id, yes: yes) }
        at += 1
        img = nil; stuck = false
        Task { await load() }
    }

    /// Swap this result for another unanswered one and carry on. Not recorded either way: an answer nobody could
    /// see is not data, and counting it would bias the number in whichever direction I guessed.
    func skip() async {
        guard at < sample.count else { return }
        if let replacement = pool.first {
            pool.removeFirst()
            sample[at] = replacement
        } else {
            sample.remove(at: at)
        }
        img = nil; stuck = false
        await load()
    }

    /// Same two-stage load as FullPhoto: the local thumbnail at once, then the sharp one. An iCloud-only original
    /// behind a bare spinner is what made tapping a photo unusable (10-11).
    func load() async {
        guard at < sample.count else { return }
        let id = sample[at]
        var got = false
        if let quick = await PhotoLibrary.image(id, side: 600, network: false), sample.indices.contains(at),
           sample[at] == id { img = quick; got = true }
        if let big = await PhotoLibrary.image(id, side: 1600, network: true), sample.indices.contains(at),
           sample[at] == id { img = big; got = true }
        guard sample.indices.contains(at), sample[at] == id else { return }
        if !got {
            // Neither size came back: swap it out rather than sit on a spinner for ever.
            stuck = true
            await skip()
        }
    }
}

/// Every Yes/No Reza has ever given, by query and by photo, in the app's own container.
///
/// M43(b), asked for by the cluster 10-11: "persist the answers per query AND per photo id, and never draw an
/// already-answered photo again; a photo answered No for 'food photos' is also a free negative for that query's
/// next run." So a second audit of the same search continues where the first stopped instead of re-asking, the
/// range narrows as he checks more over time, and the answers are labelled data this product can be measured
/// against - the only eye-truth that exists for HIS library.
actor AuditStore {
    struct Answers: Codable { var yes: [String] = [], no: [String] = [] }
    private var byQuery: [String: Answers] = [:]
    private let file: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("audits.json")
    private var loaded = false

    /// False when audits.json exists but cannot be read yet (locked since restart): then nothing is written over it.
    @discardableResult private func load() -> Bool {
        if loaded { return true }
        if FileManager.default.fileExists(atPath: file.path) {
            guard let d = try? Data(contentsOf: file),
                  let a = try? JSONDecoder().decode([String: Answers].self, from: d) else { return false }
            byQuery = a
        }
        loaded = true
        return true
    }

    /// The query key, so "Food photos " and "food photos" are the same audit.
    static func key(_ q: String) -> String { q.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }

    func answers(_ query: String) -> Answers {
        guard load() else { return Answers() }
        return byQuery[AuditStore.key(query)] ?? Answers()
    }

    func record(query: String, id: String, yes: Bool) {
        guard load() else { return }
        let k = AuditStore.key(query)
        var a = byQuery[k] ?? Answers()
        a.yes.removeAll { $0 == id }; a.no.removeAll { $0 == id }
        if yes { a.yes.append(id) } else { a.no.append(id) }
        byQuery[k] = a
        if let d = try? JSONEncoder().encode(byQuery) {
            try? d.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        }
    }
}

/// One per app run, so every album card's sheet sees the same answers.
let auditStore = AuditStore()
