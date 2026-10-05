// CLIP byte-level BPE tokenizer (open_clip SimpleTokenizer, clean="lower", context 32 for PE-Core): text -> token ids
// for the Core ML text tower. Vocabulary: Resources/bpe_simple_vocab_16e6.txt (OpenAI CLIP, MIT). Checked against
// open_clip on golden fixtures. Not ported: ftfy's mojibake repair (only matters for broken encodings).
import Foundation

public final class ClipTokenizer {
    public let contextLength: Int
    let encoder: [String: Int]
    let ranks: [String: Int]          // "first second" -> rank
    let byteEncoder: [UInt8: Character]
    let sot: Int, eot: Int
    var cache = [String: [String]]()

    public init(contextLength: Int = 32, vocabURL: URL? = nil) throws {
        self.contextLength = contextLength
        let url = vocabURL ?? Bundle.module.url(forResource: "bpe_simple_vocab_16e6", withExtension: "txt")!
        let lines = try String(contentsOf: url, encoding: .utf8).components(separatedBy: "\n")
        let merges = lines[1..<(49152 - 256 - 2 + 1)].map { $0.split(separator: " ").map(String.init) }
        var bs = Array(UInt8(33)...UInt8(126)) + Array(UInt8(161)...UInt8(172)) + Array(UInt8(174)...UInt8(255))
        var cs = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0...255 where !bs.contains(UInt8(b)) { bs.append(UInt8(b)); cs.append(256 + n); n += 1 }
        var be = [UInt8: Character]()
        for (b, c) in zip(bs, cs) { be[b] = Character(Unicode.Scalar(c)!) }
        byteEncoder = be
        var vocab = bs.map { String(be[$0]!) }
        vocab += vocab.map { $0 + "</w>" }
        for m in merges { vocab.append(m.joined()) }
        vocab += ["<start_of_text>", "<end_of_text>"]
        var enc = [String: Int](); for (i, v) in vocab.enumerated() { enc[v] = i }
        encoder = enc
        var r = [String: Int](); for (i, m) in merges.enumerated() { r[m[0] + " " + m[1]] = i }
        ranks = r
        sot = enc["<start_of_text>"]!; eot = enc["<end_of_text>"]!
    }

    static func clean(_ text: String) -> String {
        var t = text
        for _ in 0..<2 {   // html.unescape twice (common entities)
            for (e, c) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'"), ("&#x27;", "'"), ("&nbsp;", "\u{a0}")] {
                t = t.replacingOccurrences(of: e, with: c)
            }
        }
        return t.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
    }

    func bpe(_ token: String) -> [String] {
        if let c = cache[token] { return c }
        var word = token.map(String.init)
        word[word.count - 1] += "</w>"
        if word.count == 1 { return word }
        while true {
            var best: (Int, Int)? = nil           // (rank, index)
            for i in 0..<(word.count - 1) {
                if let rk = ranks[word[i] + " " + word[i + 1]], best == nil || rk < best!.0 { best = (rk, i) }
            }
            guard let (_, bi) = best else { break }
            let first = word[bi], second = word[bi + 1]
            var nw = [String](); var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second { nw.append(first + second); i += 2 }
                else { nw.append(word[i]); i += 1 }
            }
            word = nw
            if word.count == 1 { break }
        }
        cache[token] = word
        return word
    }

    public func encode(_ text: String) -> [Int] {
        let t = ClipTokenizer.clean(text)
        let pat = #"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
        let re = try! NSRegularExpression(pattern: pat, options: [.caseInsensitive])
        let ns = t as NSString
        var out = [Int]()
        for m in re.matches(in: t, range: NSRange(location: 0, length: ns.length)) {
            let tok = ns.substring(with: m.range)
            let mapped = String(tok.utf8.map { byteEncoder[$0]! })
            for piece in bpe(mapped) { if let id = encoder[piece] { out.append(id) } }
        }
        return out
    }

    /// [sot] + tokens + [eot], truncated (last = eot) and zero-padded to the context length.
    public func callAsFunction(_ text: String) -> [Int32] {
        var ids = [sot] + encode(text) + [eot]
        if ids.count > contextLength { ids = Array(ids[..<contextLength]); ids[contextLength - 1] = eot }
        return ids.map(Int32.init) + [Int32](repeating: 0, count: contextLength - ids.count)
    }
}
