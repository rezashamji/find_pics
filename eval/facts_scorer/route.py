"""RESULTS 59b: which judge questions still need PIXELS at query time (-> the 9B judge) and which go to the facts
scorer cascade. Decided from the question text the engine passes to the judge (+ whether the engine draws a red box),
written down BEFORE any cascade was scored. The kinds are RESULTS 59's measured weak spots of the facts reader:
  red-box person look (per-photo F1 .74), count (.47), colour (.76), person attribute (.77), text / sign (.83),
  photo-level look (.81), plus the kinds read off its worst end-to-end requests: relations between things, small /
  background details, follow-up OR compositions.
Everything else (objects, scenes / places / trips, activities without a person word) -> scorer cascade.
"""
import re

COLOURS = ("red|orange|yellow|green|blue|purple|pink|magenta|teal|turquoise|brown|beige|grey|gray|black|white|golden|"
           "gold|silver|navy|violet|maroon|cyan|colou?r|colou?rful|neon|pastel")
RULES = [
    ("red-box person", None),       # crop_person == "box" or "red box" in the question
    ("count", re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|several|group of|crowd|many|multiple|\d+|"
                         r"pair of|alone|only one|single|couple)\b", re.I)),
    ("text / sign", re.compile(r"\b(text|sign|signs|signage|written|writing|reads?|says|words?|letters|logo|menu|banner|"
                               r"label|poster|number plate)\b", re.I)),
    ("photo look", re.compile(r"\b(blurr?y|blurred|out of focus|motion blur|black[- ]and[- ]white|monochrome|sepia|"
                              r"overexposed|underexposed|washed out|close-?up|long exposure|silhouette|reflections?|aerial|"
                              r"panorama|bokeh|macro|wide|tilted|grainy|dark|bright|hazy|smog|foggy|fog|lit|lighting|light|"
                              r"lights|sunlight|sunset|night)\b", re.I)),
    ("colour", re.compile(rf"\b({COLOURS})\b", re.I)),
    ("person attribute", re.compile(r"\b(person|man|woman|men|women|boy|girl|child|children|kid|kids|people|someone|"
                                    r"bride|groom|baby|he|she|guy|lady|player|players)\b", re.I)),
    ("relation", re.compile(r"\b(next to|beside|behind|in front of|on top of|underneath|under|above|below|between|near|"
                            r"nearby|against|leaning|surrounded by|attached to|hanging|on (his|her|their|its))\b", re.I)),
    ("small / background detail", re.compile(r"\b(background|in the distance|distant|far away|tiny|small|little|corner|"
                                             r"edge|faint|partially|barely)\b", re.I)),
    ("or-composition", re.compile(r"\bor\b", re.I)),
]
# the grounding tail itself contains "or" ("... toy, model or picture of one") - not a composition
_TAIL = re.compile(r"\s*\(not a drawing, painting, statue, toy, model or picture of one\)", re.I)


def route(question: str, crop) -> str:
    """'' = scorer cascade; else the name of the rule that sends the question to the judge."""
    if str(crop) == "box" or re.search(r"(?i)\bred box\b", question):
        return "red-box person"
    q = _TAIL.sub("", question)
    for name, rx in RULES[1:]:
        if rx.search(q):
            return name
    return ""


# ---- RESULTS 59c (pre-registered 10-14 ~04:20, before any 59c number) ----
# TRIP / PLACE MEMBERSHIP questions ("Is this a photo taken during a trip to Bangladesh?", "...from our Maui trip?",
# "...taken in Burgos or León?") are answered from the photo's own place metadata when a capitalised name in the question
# matches the library's reverse-geocoded places (engine.place_mask, the product's place matcher): P = .99 for photos at
# that place, .01 elsewhere. A trip question whose names match no place ("during a safari", "a trip out west", "Europe")
# goes to the JUDGE under the same budget as the pixel kinds (RESULTS 59b: the scorer said no to most of those photos).
TRIP = re.compile(r"(?i)\b(trips?|vacation|holiday|journey|tour|visit|visiting|travel\w*)\b|\btaken (in|at|during|on)\b|"
                  r"\b(photo|picture|shot) from\b|\bduring (a|an|the|our|my)\b")
_CAND = re.compile(r"(?<![\w])([A-Z][\w'’.\-]*(?:\s+(?:(?:of|and|or|de|del|la|le|el|du|&)\s+)?[A-Z][\w'’.\-]*)*)")
_SKIP = {"Is", "Was", "Are", "Were", "Does", "Do", "Did", "Has", "Have", "I", "A", "An", "The", "This", "That"}


def place_candidates(question: str) -> list:
    """Capitalised names in the question (leading question word dropped), longest first within the order found."""
    q = _TAIL.sub("", question)
    out = []
    for m in _CAND.finditer(q):
        words = m.group(1).split()
        while words and words[0] in _SKIP:
            words = words[1:]
        if words:
            out.append(" ".join(words).rstrip(".?!,"))
    return [c for c in out if c]


def budget(n: int, frac: float = 0.05, floor: int = 50) -> int:
    """Judge budget for one pixel-kind question with n in-scope photos: the top max(50, 5% of n) by the millisecond score
    (RESULTS 59b: top 100 of ~1,900 = 5.3% kept the all-judge F1 with 27B facts; the floor keeps small scopes, where a
    trip's or a day's photos are often mostly matches, fully or mostly judged)."""
    return min(n, max(floor, -(-int(frac * n * 1000) // 1000)))


# ---- RESULTS 59d (pre-registered 10-15, before any 59d number; design read on TRAIN only) ----
# R59d = R59c (K = 0, judge budget, trip route) PLUS:
# (a) COMPOUND TRIP QUESTIONS. The trip route answers from place metadata ONLY when nothing is left of the question
#     once the place phrase is removed; otherwise P = metadata AND the remainder (remainder P at the place, .01 elsewhere),
#     the remainder routed like any question (pixel kinds -> judge under the budget, else the scorer) and split by (b).
#     "Nothing left" = no word beyond question / photo / trip frame words (FRAME, incl. a trip's kind: "a hiking trip in
#     Mendoza", "a hike on the Pacific Crest Trail") and date words (TIME: a date is the album's date window, not pixels;
#     R59c's behaviour for "taken in Chicago during the summer of 2010" is kept). Place names match TOLERANTLY
#     (PlaceIndex.mask): accents and case folded, every word of the name must equal an address token (or the name with
#     spaces removed one token: "Machu Picchu" ~ "Machupicchu"); only if that matches no photo of the library, a word of
#     >= 6 characters may match an address token of >= 6 characters within edit distance 2 ("Pagsanjan" ~ "Pinagsanjan"),
#     except when a plain word of a multiword name already matches exactly on its own ("London Olympics": london is a
#     place, "olympics" an event, not a misspelt "Olympic" park; read on TRAIN a0); only if that still matches nothing,
#     place-kind and natural-feature words (engine PLACE_KIND + FEATURE) become optional ("Pagsanjan Falls").
#     Candidates: route.place_candidates, the first one that matches >= 1 photo (as R59c).
# (b) OR / AND COMPOSITIONS SPLIT. "A or B" questions are split into parts (split_parts), each part routed separately
#     (pixel kinds -> judge under ITS OWN budget, else the scorer), P = max over parts (union). Forms: a clause joined by
#     ", or <is/are/was/...>" (grounding's _or_question), and word-level "PRE X or Y [or Z] POST" (opener, determiner and
#     terminator rules in _word_split; adjective alternatives share the head noun: "a maroon or white jersey"; no split
#     when an alternative ends in a function word: "taken with or showing"). "A and also B" and two question sentences
#     ("Is there a crowd? Is the ground muddy?") split the same way, P = min over parts. Plain " and " and ", and <aux>"
#     clauses are never split ("black and white", "Maine and Canada", the grounding's ", and was this photo taken at a X").
#     Parts are decomposed again (a part may itself be a trip question); a part's leaf route is route(), or the judge
#     when it is a trip question with no matching place (as R59c).
# (c) found in h on TRAIN failures only (eval/facts_scorer/r59d/diag_roles.py on train_ms_pairs.parquet; dev not read):
#   (c1) BUDGET EXTENSION. On TRAIN h, 54 judge-routed questions are budget-limited (scope > budget); they hold 5,098 9B
#        matches of which the budget reaches 2,151 (.42); in 35 of them the budget's last 50 judged photos are >= 50%
#        matches, and those 35 hold 5,017 of the 5,098 (follow-up filters inside a trip/place: "mountains in the
#        background" 377 matches, budget 50; "wearing glasses" 241 / 50). Rule: a judge-routed question is judged in
#        blocks of its budget M = max(50, 5% of scope) in millisecond-score order; after each block, if >= 50% of THAT
#        block are judge matches (P >= .7), the next block is judged too; stop at the first block below 50% or when the
#        scope is exhausted. Photos never judged keep the millisecond P.
#   (c2) EXCLUSIONS ARE JUDGED. A plan's exclude_question ("remove the ones on the kitchen counter") is judge-routed
#        whatever its kind (budget + extension, millisecond-score order = the photos the scorer would drop are checked
#        first). TRAIN: on scorer-routed exclude questions the scorer agrees with the 9B worst of all roles (F1 vs 9B: h
#        .80, g .69; album questions h .85), and the largest single h loss is one: "Is the bottle on a kitchen counter?"
#        scorer yes 24/27 vs 9B 2/27 -> R59c F1 .20 vs all-judge .96 (35032604@N00:h31).
# Acceptability as 59c: lower 95% bound of the ALL difference vs all-judge >= -.03 on TRAIN AND dev (4B phone facts);
# reported per category vs R59c (paired), judge calls; 27B facts as reference. Ablations r59d_a / _b / _c = R59c plus
# one item each (attribution only, not part of the decision).
# Disclosures: (c) and the wording details of (a)/(b) (FRAME/TIME words, the fuzzy guard, opener/terminator rules) were
# designed on TRAIN (question texts in design_train.txt, R59c TRAIN per-request scores, train_ms_pairs.parquet), so TRAIN
# is in-sample for them; dev (not read for any of this) is the clean confirmation. Same scorers / facts / plans / judge
# cache as 59c: 4B = towervec12B_q3vl4bq4r (TRAIN: fold _h{fold}), facts_q3vl4bq4rm; 27B = towervec12B_full (TRAIN:
# towervec12B_h{fold}), facts_27bm; new judge questions (parts, remainders) answered by a lazily loaded 9B.
# ALL-JUDGE stack with (a) + (b) (aj59d: every leaf judged on its whole scope, trip metadata AND judged remainder, parts
# judged, union / min), dev only: ADOPT in the product (Python + Swift) iff dev ALL diff vs the all-judge reference > 0
# with lower 95% bound >= -.01 and no category's mean down by more than .02.
import unicodedata  # noqa: E402

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
        from findpics.engine import PLACE_KIND
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
