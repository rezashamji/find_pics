// RESULTS 68 (4): the WordPiece tokenizer of towervec's text encoder (all-MiniLM-L12-v2 vocab, uncased), = Hugging Face
// tokenizers' BertNormalizer (clean text, CJK padding, strip accents, lowercase) + BertPreTokenizer (whitespace and
// punctuation split) + WordPiece (greedy longest match, "##" continuation, > 100 chars or no match = [UNK]). Checked against
// the fast tokenizer the scorers used (Tests/.../Fixtures/bert_tokens.json from eval/cascade_port/encoder.py).
// Model inputs as the scorers built them (eval/facts_scorer/scorers.py Tok): facts line [CLS] + ids[:254] + [SEP],
// question [CLS] + ids[:64] + [SEP], token type 0 everywhere, no padding for one sequence.
import Foundation

public struct BertTokenizer: Sendable {
    let vocab: [String: Int]
    public let cls: Int, sep: Int, unk: Int, pad: Int
    let maxCharsPerWord = 100

    public init(vocabLines: [String]) {
        var v = [String: Int]()
        for (i, t) in vocabLines.enumerated() where v[t] == nil { v[t] = i }
        vocab = v
        cls = v["[CLS]"] ?? 101; sep = v["[SEP]"] ?? 102; unk = v["[UNK]"] ?? 100; pad = v["[PAD]"] ?? 0
    }

    public init(vocabURL: URL) throws {
        let s = try String(contentsOf: vocabURL, encoding: .utf8)
        var lines = s.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        self.init(vocabLines: lines.map { $0.hasSuffix("\r") ? String($0.dropLast()) : $0 })
    }

    static func isControl(_ c: Unicode.Scalar) -> Bool {
        if c == "\t" || c == "\n" || c == "\r" { return false }
        switch c.properties.generalCategory {
        case .control, .format, .unassigned, .privateUse, .surrogate: return true
        default: return false
        }
    }

    static func isWhitespace(_ c: Unicode.Scalar) -> Bool {
        c == " " || c == "\t" || c == "\n" || c == "\r" || c.properties.isWhitespace
    }

    static func isChinese(_ c: Unicode.Scalar) -> Bool {
        let v = c.value
        return (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v) || (0x20000...0x2A6DF).contains(v) ||
            (0x2A700...0x2B73F).contains(v) || (0x2B740...0x2B81F).contains(v) || (0x2B920...0x2CEAF).contains(v) ||
            (0xF900...0xFAFF).contains(v) || (0x2F800...0x2FA1F).contains(v)
    }

    static func isPunctuation(_ c: Unicode.Scalar) -> Bool {
        let v = c.value
        if (33...47).contains(v) || (58...64).contains(v) || (91...96).contains(v) || (123...126).contains(v) { return true }
        switch c.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation, .initialPunctuation,
             .finalPunctuation, .otherPunctuation: return true
        default: return false
        }
    }

    /// BertNormalizer(clean_text, handle_chinese_chars, strip_accents = lowercase, lowercase)
    static func normalize(_ text: String) -> [Unicode.Scalar] {
        var a = [Unicode.Scalar]()
        for c in text.unicodeScalars {
            if c.value == 0 || c.value == 0xFFFD || isControl(c) { continue }
            if isWhitespace(c) { a.append(" "); continue }
            if isChinese(c) { a.append(" "); a.append(c); a.append(" "); continue }
            a.append(c)
        }
        // strip accents: NFD, drop nonspacing marks (Mn)
        var s = String(String.UnicodeScalarView(a)).decomposedStringWithCanonicalMapping.unicodeScalars.filter {
            $0.properties.generalCategory != .nonspacingMark
        }
        // lowercase (full Unicode mapping, per scalar)
        var out = [Unicode.Scalar]()
        for c in s { out.append(contentsOf: c.properties.lowercaseMapping.unicodeScalars) }
        s = String.UnicodeScalarView(out)
        return Array(s)
    }

    /// BertPreTokenizer: split on whitespace (removed) and on every punctuation scalar (kept as its own word)
    static func preTokenize(_ s: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var words = [[Unicode.Scalar]](), cur = [Unicode.Scalar]()
        for c in s {
            if isWhitespace(c) {
                if !cur.isEmpty { words.append(cur); cur = [] }
            } else if isPunctuation(c) {
                if !cur.isEmpty { words.append(cur); cur = [] }
                words.append([c])
            } else {
                cur.append(c)
            }
        }
        if !cur.isEmpty { words.append(cur) }
        return words
    }

    func wordPiece(_ w: [Unicode.Scalar]) -> [Int] {
        if w.count > maxCharsPerWord { return [unk] }
        var out = [Int](), start = 0
        while start < w.count {
            var end = w.count, found: Int? = nil
            while start < end {
                var sub = String(String.UnicodeScalarView(w[start..<end]))
                if start > 0 { sub = "##" + sub }
                if let id = vocab[sub] { found = id; break }
                end -= 1
            }
            guard let id = found else { return [unk] }
            out.append(id); start = end
        }
        return out
    }

    /// token ids without special tokens (= tokenizer(text, add_special_tokens=False))
    public func ids(_ text: String) -> [Int] {
        Self.preTokenize(Self.normalize(text)).flatMap { wordPiece($0) }
    }

    /// model input for a facts line: [CLS] + ids[:254] + [SEP]
    public func factsInput(_ text: String) -> [Int] { [cls] + ids(text).prefix(254) + [sep] }

    /// model input for a scorer question: [CLS] + ids[:64] + [SEP]
    public func questionInput(_ text: String) -> [Int] { [cls] + ids(text).prefix(64) + [sep] }
}
