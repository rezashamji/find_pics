// Throughput on the phone (RESULTS 44): two things the judge loop can stop doing, with the model call itself unchanged.
//
// (b) PREFETCH. Search.swift used to read each photo (PhotoKit, very often an iCloud download of the 896 px copy:
//     ~0.6 s each one at a time, MAC_INBOX M10) and only then start the ~1.7-4 s model call, so the GPU sat idle during
//     every read. Prefetcher loads the next photos (streamRounds' `upcoming` hint) while the current one is judged.
//
// (a) IMAGE-PREFIX REUSE. A photo that passes the main question may get a follow-up (exclude / filter question). The
//     prompt puts the image first, so everything up to <|vision_end|> depends only on the photo: the vision encoder and
//     the ~600-token image prefill can run once, with each question's ~30 tokens appended to that KV cache.
//     imagePrefixLength / tailAfterPrefix decide where the prompt splits and whether a cached prefix may be reused.
import Foundation

/// Loads values (photos) ahead of use. `want(keys)` starts loading the keys not already loading and drops every other
/// key it holds (a wrong guess costs one wasted read, never a wrong photo: values are looked up by key). `take(key)`
/// returns the prefetched value, waiting for it if it is still loading, or loads it right away on a miss.
public actor Prefetcher<Value: Sendable> {
    private let load: @Sendable (String) async -> Value?
    private var tasks: [String: Task<Value?, Never>] = [:]
    /// take() calls served by a prefetch / loaded on the spot; prefetches dropped unused (the hint was wrong).
    public private(set) var hits = 0, misses = 0, dropped = 0

    public init(load: @escaping @Sendable (String) async -> Value?) { self.load = load }

    public func want(_ keys: [String]) {
        let keep = Set(keys)
        for (k, t) in tasks where !keep.contains(k) { t.cancel(); tasks[k] = nil; dropped += 1 }
        for k in keys where tasks[k] == nil {
            let load = self.load
            tasks[k] = Task { await load(k) }
        }
    }

    public func take(_ key: String) async -> Value? {
        if let t = tasks.removeValue(forKey: key) { hits += 1; return await t.value }
        misses += 1
        return await load(key)
    }

    /// Keys loading or loaded and not yet taken (at most the last `want` list).
    public var held: Int { tasks.count }
}

/// Where a judge prompt splits into the part that depends only on the photo and the question's tail: the number of
/// tokens up to and including the LAST `visionEnd` token (Qwen3-VL: <|vision_end|>), or nil when there is none or nothing
/// follows it (then nothing can be reused).
public func imagePrefixLength(_ tokens: [Int], visionEnd: Int) -> Int? {
    guard let i = tokens.lastIndex(of: visionEnd), i + 1 < tokens.count else { return nil }
    return i + 1
}

/// The question tail to append to a cached image prefix, or nil unless `tokens` starts with exactly `prefix` and has
/// at least one more token. The same photo at the same size gives the same image tokens; another photo size, another
/// template or another model gives different ones, and is computed from scratch.
public func tailAfterPrefix(_ tokens: [Int], prefix: [Int]) -> ArraySlice<Int>? {
    guard !prefix.isEmpty, tokens.count > prefix.count, tokens.starts(with: prefix) else { return nil }
    return tokens[prefix.count...]
}
