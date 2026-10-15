"""Judge questions that are COMPOSITIONS, answered by parts (RESULTS 59d, adopted 10-15; Swift QuestionTree.swift).
  (a) a trip / place question ("Is this a photo taken during a trip to Puerto Rico?", "... with a sepia tone taken in
      Spain?"): the place comes from the photo's own place metadata (accent-, case- and spelling-tolerant match against
      the address tokens), never from the pixels; what is left of the question after the place phrase (if anything but
      frame / trip-kind / date words) is judged on the photos at that place; elsewhere P = .01.
  (b) "A or B" is judged as A and as B (P = max); "A and also B" / two questions at once as both (P = min).
Complex bench dev, all-judge stack (cached 9B): ALL +.011 [+.001, +.022] vs the stack without it (a +.011, d +.033,
g +.024, h +.030; b -.004, f -.008). The text rules are the pre-registered R59d rules of eval/facts_scorer/route.py.
"""
import re
import unicodedata

import numpy as np

PLACE_KIND = {"city", "county", "state", "province", "region", "regional", "district", "borough", "town", "township",
              "village", "municipality", "metropolitan", "metro", "area", "greater", "downtown", "prefecture", "of", "the"}
COLOURS = ("red|orange|yellow|green|blue|purple|pink|magenta|teal|turquoise|brown|beige|grey|gray|black|white|golden|"
           "gold|silver|navy|violet|maroon|cyan|colou?r|colou?rful|neon|pastel")
_TAIL = re.compile(r"\s*\(not a drawing, painting, statue, toy, model or picture of one\)", re.I)
TRIP = re.compile(r"(?i)\b(trips?|vacation|holiday|journey|tour|visit|visiting|travel\w*)\b|\btaken (in|at|during|on)\b|"
                  r"\b(photo|picture|shot) from\b|\bduring (a|an|the|our|my)\b")
_CAND = re.compile(r"(?<![\w])([A-Z][\w'’.\-]*(?:\s+(?:(?:of|and|or|de|del|la|le|el|du|&)\s+)?[A-Z][\w'’.\-]*)*)")
_SKIP = {"Is", "Was", "Are", "Were", "Does", "Do", "Did", "Has", "Have", "I", "A", "An", "The", "This", "That"}


def place_candidates(question: str) -> list:
    """Capitalised names in the question (leading question word dropped), in the order found."""
    q = _TAIL.sub("", question)
    out = []
    for m in _CAND.finditer(q):
        words = m.group(1).split()
        while words and words[0] in _SKIP:
            words = words[1:]
        if words:
            out.append(" ".join(words).rstrip(".?!,"))
    return [c for c in out if c]


FEATURE = {"falls", "fall", "waterfall", "waterfalls", "lake", "lakes", "river", "mountain", "mountains", "mount", "mt",
           "beach", "beaches", "bay", "island", "islands", "isle", "park", "national", "temple", "castle", "cathedral",
           "church", "museum", "palace", "bridge", "tower", "square", "market", "old", "valley", "canyon", "trail",
           "peak", "hill", "hills", "coast", "harbor", "harbour", "port", "airport", "station", "street", "zoo", "springs",
           "gardens", "garden", "forest", "desert", "glacier", "volcano", "reef", "cave", "caves", "ruins", "fort"}
FRAME = {"is", "are", "was", "were", "this", "that", "these", "those", "the", "a", "an", "photo", "photos", "picture",
         "pictures", "image", "images", "shot", "shots", "video", "taken", "take", "from", "during", "in", "at", "on",
         "to", "of", "near", "around", "our", "my", "we", "us", "i", "it", "trip", "trips", "vacation", "vacations",
         "holiday", "holidays", "journey", "tour", "visit", "visiting", "visited", "travel", "traveling", "travelling",
         "travels", "and", "there", "any", "real", "anywhere", "visible", "does", "do", "did", "show", "showing", "shows",
         "featuring", "place", "somewhere", "area", "region", "s",
         # a trip's KIND is trip membership, not a condition in the pixels ("a hiking trip in Mendoza", "a hike on X")
         "hike", "hikes", "hiking", "excursion", "excursions", "outing", "getaway", "honeymoon", "road", "sightseeing"}
TIME = re.compile(r"(?i)^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|"
                  r"august|september|october|november|december|spring|summer|fall|autumn|winter|early|late|mid|"
                  r"(?:19|20)\d\d|\d{1,2}(?:st|nd|rd|th)?|\d{4}s|weekend|week|month|year)$")


def fold(s: str) -> str:
    """accents and case folded, everything but letters/digits -> space ("León" -> "leon", not "le n")."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def lev(a: str, b: str, cap: int = 2) -> int:
    """Levenshtein distance, early exit once every cell exceeds cap (returns cap + 1)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


_PLACE_LIST = re.compile(r"(?i)\s+(?:and|or)\s+|\s*&\s*")


class PlaceIndex:
    """One library's place texts as token sets (built once per library)."""

    def __init__(self, place_texts):
        self.toks = [set(fold(t).split()) for t in list(place_texts.fillna("") if hasattr(place_texts, "fillna") else place_texts)]
        self.vocab = sorted(set().union(*self.toks)) if self.toks else []
        self._near = {}

    def mask_words(self, words, fuzzy):
        import numpy as np
        m = np.ones(len(self.toks), bool)
        for w in words:
            alts = {w}
            if fuzzy and len(w) >= 6:
                if w not in self._near:
                    self._near[w] = {t for t in self.vocab if len(t) >= 6 and lev(w, t) <= 2}
                alts |= self._near[w]
            m &= np.array([bool(alts & t) for t in self.toks], bool)
        return m

    def mask(self, name: str):
        if not hasattr(self, "_memo"):
            self._memo = {}
        if name not in self._memo:
            self._memo[name] = self._mask(name)
        return self._memo[name]

    def _mask(self, name: str):
        import numpy as np
        parts = [p for p in _PLACE_LIST.split(name or "") if p.strip()]
        if len(parts) > 1:
            m = np.zeros(len(self.toks), bool)
            for p in parts:
                m |= self.mask(p)
            return m
        words = [w for w in fold(name).split() if len(w) > 1]
        if not words:
            return np.zeros(len(self.toks), bool)
        core = [w for w in words if w not in PLACE_KIND]
        core2 = [w for w in core if w not in FEATURE]
        # fuzzy only for spelling variants: not when a plain word of a multiword name matches exactly on its own
        # ("London Olympics": "london" matches, "olympics" is an event, not a misspelt "olympic" park)
        solo = len(words) > 1 and any(self.mask_words([w], False).any() for w in core2)
        for fuzzy in ((False,) if solo else (False, True)):
            for ws in [words] + ([core] if core and len(core) < len(words) else []):
                m = self.mask_words(ws, fuzzy)
                if not fuzzy and len(ws) > 1:
                    m |= np.array(["".join(ws) in t for t in self.toks], bool)
                if m.any():
                    return m
        if core2 and len(core2) < len(core):
            for fuzzy in ((False,) if solo else (False, True)):
                m = self.mask_words(core2, fuzzy)
                if m.any():
                    return m
        return np.zeros(len(self.toks), bool)


def remainder(question: str, name: str) -> str:
    """The question with the place phrase removed ('' = nothing left beyond FRAME / TIME words). Removes the grounding's
    appended clause ', and was this photo taken at a <name>' and '<preposition> <article> <name>', then a dangling trip
    frame ('during a trip', 'taken')."""
    tail = bool(_TAIL.search(question))
    q = _TAIL.sub("", question).strip()
    n = re.escape(name)
    q = re.sub(rf"(?i),?\s*\band\s+(?:was|is|were)\s+(?:this|the)\s+(?:photo|picture|image|video)\s+taken\s+"
               rf"(?:at|in|on|near)\s+(?:(?:a|an|the)\s+)?{n}(?![\w])", "", q)
    q = re.sub(rf"(?i)\s*\b(?:(?:in|at|to|from|on|near|of|around|across|through|visiting)\s+)?(?:(?:a|an|the)\s+)?{n}(?![\w])",
               "", q)
    q = re.sub(r"(?i)\s*\b(?:(?:taken|shot)\s+)?(?:during|on|from|in)\s+(?:a|an|the|our|my)\s+(?:trips?|vacation|holiday|"
               r"journey|tour|visit|travels?)\b", "", q)
    q = re.sub(r"(?i)\s+(?:taken|shot)\s*(?=[?,]|$)", "", q)
    q = re.sub(r"\s+([?,.])", r"\1", re.sub(r"\s+", " ", q)).strip()
    q = re.sub(r",\s*\?$", "?", q)
    words = [w for w in fold(q).split() if w not in FRAME and not TIME.match(w)]
    if not words:
        return ""
    q = q.rstrip("?").strip()
    return q + (" (not a drawing, painting, statue, toy, model or picture of one)" if tail else "") + "?"


# ---- (b) splitting ----
DET = {"a", "an", "the", "any", "some", "its", "his", "her", "their", "my", "our"}
FUNC = {"with", "of", "in", "on", "at", "for", "to", "by", "from", "as", "and", "a", "an", "the", "taken", "is", "are",
        "or", "into", "onto", "than"}
ADJ = set(COLOURS.replace("colou?r|colou?rful|", "").split("|")) | {
    "small", "big", "large", "tiny", "little", "miniature", "mini", "huge", "giant", "tall", "short", "long", "old",
    "young", "new", "light", "dark", "bright", "pale", "striped", "spotted", "plain", "wooden", "metal", "plastic"}
AUX = r"(?:is|are|was|were|does|do|did|has|have|can|could)"
NEG = {"not", "no", "none", "non", "neither", "nor", "else", "otherwise", "something", "anything"}
OPENERS = [re.compile(p, re.I) for p in (
    r"^(?:is|are|was|were) there (?:(?:any|some) )?",
    r"^(?:is|was) this (?:an? )?(?:close-up |close up )?(?:photo|picture|image|shot|video|selfie|snapshot)"
    r"(?: taken)? (?:of|showing|with|featuring) ",
    r"^(?:is|was|are|were) (?:the|this|these) (?:person|people|photo|image|picture)(?: in (?:the|this) "
    r"(?:photo|picture|image|red box))? ",
    r"^(?:is|are|was|were|does|do|did|has|have|can) (?:this|these|it|the|there) ",
    r"^\w+ ")]
_TERM = re.compile(r",|\s*§|\s+(?:anywhere|visible|clearly|taken)\b|\s+in (?:this|the) (?:photo|image|picture|video)\b|"
                   r"\s+(?:in|at|from|near) (?:the )?(?=[A-Z])")


def _top_level(body: str, pat: str):
    """start/end of each match of pat outside parentheses."""
    out, depth = [], 0
    marks = {m.start(): m for m in re.finditer(pat, body, re.I)}
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and i in marks:
            out.append((marks[i].start(), marks[i].end()))
    return out


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _fix_article(pre: str, alt: str) -> str:
    m = re.search(r"(?i)\b(an?) $", pre)
    if m and alt:
        art = "an" if alt[0].lower() in "aeiou" else "a"
        pre = pre[:m.start(1)] + (art if m.group(1).islower() else _cap(art)) + " "
    return pre


def _word_split(body: str, conj: str):
    """'PRE X <conj> Y [<conj> Z] POST' -> [PRE X POST, PRE Y POST, ...] (None = no safe split)."""
    ors = _top_level(body, rf"\s+{conj}\s+")
    if not ors:
        return None
    m = next((o.match(body) for o in OPENERS if o.match(body)), None)
    o_end = m.end() if m else 0
    if ors[0][0] < o_end:
        return None
    last = body[ors[-1][1]:]
    t = _TERM.search(last)
    cut = t.start() if t else len(last)
    post = last[cut:]
    ys = [body[a[1]:b[0]] for a, b in zip(ors, ors[1:])] + [last[:cut]]
    region = body[o_end:ors[0][0]]
    dets = [mm for mm in re.finditer(r"\b(\w+)\s+", region) if mm.group(1).lower() in DET]
    if ys[0].split() and ys[0].split()[0].lower() in DET:
        xs = o_end + (dets[0].start() if dets else 0)
    else:
        xs = o_end + (dets[-1].end() if dets else 0)
    pre, x = body[:xs], body[xs:ors[0][0]]
    alts = [x] + ys
    if any(not a.strip() or a.split()[-1].lower() in FUNC for a in alts) or len(x.split()) > 8:
        return None
    # "... in real life or not?" is ONE yes/no question, not two (product guard added 10-15 after the R59d evaluation;
    # no TRAIN/dev judge question has such an alternative, so no measured number moves)
    if any(a.split()[0].lower() in NEG for a in ys):
        return None
    if ys[0].split()[0].lower().endswith("ing") and not x.split()[0].lower().endswith("ing") and \
            x.split()[0].lower() not in DET:
        return None
    head = ys[-1].split()
    if len(head) >= 2 and all(len(a.split()) == 1 and a.lower() in ADJ for a in alts[:-1]):
        alts = [a + " " + " ".join(head[1:]) for a in alts[:-1]] + [ys[-1]]
    return [(_fix_article(pre, a) + a + post).strip() for a in alts]


def split_parts(question: str):
    """(op, parts) with op 'or' (P = max) or 'and' (P = min); None = not a composition we split. Clause forms are cut in
    the raw question (a part keeps the grounding tail only if its own clause had it); word-level forms share the tail."""
    raw = question.strip()
    sents = [s.strip() for s in re.split(r"\?\s+(?=[A-Z])", raw) if s.strip()]
    if len(sents) > 1:     # two questions asked at once = both must hold
        return "and", [s.rstrip(" ?") + "?" for s in sents]
    cl = _top_level(raw.rstrip(" ?"), rf",\s+or\s+(?={AUX}\b)")
    if cl:
        body, pieces, s0 = raw.rstrip(" ?"), [], 0
        for a, b in cl:
            pieces.append(body[s0:a]); s0 = b
        pieces.append(body[s0:])
        return "or", [_cap(x.strip()) + "?" for x in pieces]
    T = " (not a drawing, painting, statue, toy, model or picture of one)"
    body = _TAIL.sub(" §", raw, count=1).strip().rstrip(" ?")     # the tail stays where it was, as one placeholder
    end = lambda s: re.sub(r"\s*§", T, s.rstrip(" ?.")) + "?"  # noqa: E731
    if re.search(r"(?i)\band also\b", body):
        p = _word_split(body, "and also")
        return ("and", [end(x) for x in p]) if p else None
    if _top_level(body, r"\s+or\s+"):
        p = _word_split(re.sub(r",\s+or\s+", " or ", body), "or")
        return ("or", [end(x) for x in p]) if p else None
    return None


def decompose(question: str, places: "PlaceIndex | None", depth: int = 0, meta_ok: bool = True):
    """-> ("meta", mask, child | None) | ("or" | "and", [nodes]) | ("leaf", question)."""
    if meta_ok and places is not None and TRIP.search(_TAIL.sub("", question)):
        for c in place_candidates(question):
            mk = places.mask(c)
            if mk.any():
                rem = remainder(question, c)
                return ("meta", mk, decompose(rem, places, depth + 1, False) if rem else None)
    if depth < 4:
        sp = split_parts(question)
        if sp:
            return (sp[0], [decompose(x, places, depth + 1, meta_ok) for x in sp[1]])
    return ("leaf", question)


def evaluate(node, rows: np.ndarray, judge_leaf) -> np.ndarray:
    """P per row: metadata .99 / .01, a remainder judged only at the place (.01 elsewhere), OR = max, AND = min;
    judge_leaf(rows, question) -> P for a plain question."""
    if not len(rows):
        return np.zeros(0, np.float32)
    if node[0] == "meta":
        mk = node[1][rows]
        if node[2] is None:
            return np.where(mk, 0.99, 0.01).astype(np.float32)
        out = np.full(len(rows), 0.01, np.float32)
        if mk.any():
            out[mk] = evaluate(node[2], rows[mk], judge_leaf)
        return out
    if node[0] in ("or", "and"):
        P = np.stack([evaluate(ch, rows, judge_leaf) for ch in node[1]])
        return (P.max(0) if node[0] == "or" else P.min(0)).astype(np.float32)
    return np.asarray(judge_leaf(rows, node[1]), np.float32)


def tree_json(node):
    """A decomposition as plain data (golden fixture for Swift): masks as the sorted list of matching rows."""
    if node is None:
        return None
    if node[0] == "meta":
        return {"meta": [int(i) for i in np.where(node[1])[0]], "rest": tree_json(node[2])}
    if node[0] in ("or", "and"):
        return {node[0]: [tree_json(c) for c in node[1]]}
    return {"leaf": node[1]}
