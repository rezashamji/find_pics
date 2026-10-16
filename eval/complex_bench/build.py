"""Complex-query benchmark, step 4: truth programs + oracle answers -> CANDIDATE answer sets; drop degenerate or
ambiguous queries; write eval/complex_bench/queries/<split>.json.

Program semantics (an album): all metadata constraints AND every `need` question yes AND at least one `need_any` yes
AND every `not` question no (oracle P(yes) >= CUT) AND, for a person album, the person's face found AND every
`person_need` yes on the boxed person. An anchor's window is computed from the anchor photo ids the generator named;
the query is DROPPED if the anchor's own description (anchor.need, judged on every photo) does not find those ids or
also finds photos whose window differs (the owner's words would be ambiguous). Category f (one half-remembered photo)
= the target ids plus every photo that passes all of the target's precise questions; dropped if that is > 6 photos.
Usage: python eval/complex_bench/build.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, city, load_items, load_programs, whole_photo_q  # noqa: E402

import os as _os
CUT = float(_os.environ.get("FP_CB_CUT", "0.5"))
# per-category cut, e.g. FP_CB_CUTS="a=0.9,b=0.9" (oracle P(yes) needed for a yes; chosen on DEV-library audit labels)
# final per-category cuts (10-12): chosen on the EVEN-index audited requests, reported on the ODD half (audit/quality_*.txt)
# RESULTS 52 (10-13): h 0.9 -> 0.5 with the rewritten h programs (fresh blind audit3: cut 0.5 precision .93 / recall .94
# vs cut 0.9 .98 / .66; 0.5 wins on test-only, dev-only and both even/odd halves - the rule written before scoring)
# RESULTS 56 (10-13): b 0.8 -> 0.9 with the rewritten b programs (fresh blind audit5: F1 .786 vs .727 on test-only, .716 vs
# .710 on dev-only - the rule written before scoring; adopted version precision .73 / recall .79)
FINAL_CUTS = "a=0.7,b=0.9,c=0.5,d=0.8,e=0.5,f=0.9,g=0.5,h=0.5"
CUTS = {k: float(v) for k, v in (x.split("=") for x in _os.environ.get("FP_CB_CUTS", FINAL_CUTS if "FP_CB_CUT" not in _os.environ else "").split(",") if x)}
# categories whose candidate sets passed (or nearly passed) the audit bar precision ~0.8 / recall 0.7; b, c, g are provisional
# h VERIFIED since RESULTS 52 (rewritten programs, cut 0.5: audit3 precision .93 [.84, .97] / recall .94 [.89, .98]);
# b PROVISIONAL since RESULTS 56 (fresh audit5: old truth precision .56, adopted rewrite at cut 0.9 .73 / recall .79 < bar)
# RESULTS 63 (10-16, adopted by the coordinator): g VERIFIED on the corrected truth (removals only; queries/{dev,train}.json
# = t63/queries, the pre-63 files in queries_pre63/). NOTE: rebuilding with build.py does NOT re-apply the t63 removals.
VERIFIED = set(_os.environ.get("FP_CB_VERIFIED", "a,d,e,f,g,h").split(","))
QDIR = _os.environ.get("FP_CB_QDIR", "queries")
# FP_CB_ONLY_SPLITS="train": build (and write) only those splits' libraries, e.g. to add newly finished TRAIN libraries
# into a separate FP_CB_QDIR without touching (or reading) the other splits (10-12, planner training)
ONLY = set(_os.environ.get("FP_CB_ONLY_SPLITS", "test,dev,train").split(","))
# v2 (10-12, after reading the dev failures): the generator often narrowed a request with dates/places the owner never
# said ("the guy playing the red guitar on stage" -> only 2011-05-28 in Barcelona), so correct photos from other days
# counted as false positives. A date constraint is kept only if the words carry a time/event cue; a place constraint
# only if one of its words appears in the request.
TIME_CUE = re.compile(r"\d|\b(today|tonight|yesterday|ago|last|past|this|next|recent|recently|jan|feb|mar|apr|may|jun|jul|"
                      r"aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|"
                      r"december|spring|summer|fall|autumn|winter|christmas|xmas|thanksgiving|halloween|easter|new year|"
                      r"weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday|trip|vacation|holiday|honeymoon|"
                      r"when|during|day|night|morning|evening|afternoon|year|week|month|wedding|party|concert|gig|show|"
                      r"festival|fair|parade|game|match|race|marathon|hike|walk|visit|tour|cruise|conference|graduation|"
                      r"birthday|ceremony|rodeo|carnival|tailgate|before|after|since|until|then|that time|era)\b", re.I)
STRICT_WORDS = "--keep-hidden" not in sys.argv
# v4 (10-12, from audit2): dropping unstated time/place HURT a/b/c/g (v2-added answers were right 3/13, 3/18, 0/18,
# 1/4 by eye, while true matches outside the generator's scope were rare: 0/43, 2/36, 3/48, 4/49) and helped d/h
# (recall 0.68 -> 0.72, 0.75 -> 0.86). So the time/place stripping is applied only to these categories.
STRIP_CATS = set(_os.environ.get("FP_CB_STRIP", "d,h").split(","))
# RESULTS 52 (10-13): compile rules for category h (FP_CB_HFIX=0 = the 10-12 rules, for reproduction). Reading every
# dev + train h program showed the strip above removing the ENCODING of a place the owner names by another word:
# "photos from Laos" was a date window (no time word -> dropped -> the whole library), "Hampi" was "Hosapete taluk",
# "Bangladesh" was "Dhaka", "Machu Picchu" vs the address word "Machupicchu", "at the winery" was "Paso Robles". Now:
#   - a place constraint stays when the words name ANY place (an address name of the library at any level, a
#     landmark from the photo facts, or "the winery / hospital / zoo ..."), and a date/day/hour constraint stays when
#     the words carry a time cue (as before) OR name a place the album's place fields do not cover (then the dates are
#     how the program encodes it: "photos of Borobudur" = its days);
#   - place names match accent-, case- and space-insensitively ("Machu Picchu" = "Machupicchu", "León" = "Leon";
#     the old matcher turned "León" into the word "le"); new field place_any = ANY of several places;
#   - a turn may hold several albums: its answer is their UNION ("also add the ones from Zagreb");
#   - a follow-up whose words only narrow ("just / only / remove ...") must return a subset of the previous turn, and
#     one that only widens ("also / too / add / include") a superset; otherwise the request is dropped (the program
#     contradicts its own conversation).
H_FIX = _os.environ.get("FP_CB_HFIX", "1") == "1"
# RESULTS 53 (10-13): category g (two albums) gets the same place matcher (accent/case/space-insensitive, place_any) and
# a stated-year check (the words state a year but no album carries a date); FP_CB_GFIX=0 = the 10-12 rules for g.
G_FIX = _os.environ.get("FP_CB_GFIX", "1") == "1"
# RESULTS 56 (10-13): category b (attribute/colour/style) gets the same place matcher and the stated-year check;
# FP_CB_BFIX=0 = the 10-12 rules for b.
B_FIX = _os.environ.get("FP_CB_BFIX", "1") == "1"
_GENERIC_PLACE = set("""beach park lake river market center centre downtown harbour harbor airport station garden gardens
zoo university hospital museum church castle island islands bay port city north south east west new old town village
valley mountain mountains forest hill hills road street square bridge temple cathedral palace tower falls canyon national
state county district region province municipality quarter shopping mall food concert festival lighthouse""".split())
PLACE_NOUN = re.compile(r"\b(the|that|our|this)\s+(\w+\s+)?(winery|vineyard|hospital|zoo|aquarium|museum|restaurant|hotel|"
                        r"resort|airport|stadium|arena|campus|school|university|farm|ranch|cabin|lodge|camp|campground|"
                        r"brewery|theater|theatre|gallery|venue|club|pub|bar)s?\b", re.I)
NARROW = re.compile(r"\b(just|only|remove|exclude|excluding|without|take out|drop|get rid|except|make sure|but not|"
                    r"no more)\b", re.I)
WIDEN = re.compile(r"\b(too|also|add|include|including|as well|plus|back|even)\b", re.I)
# "actually" / "wait" open most follow-ups of every kind in these conversations, so they mark nothing by themselves
REPLACE = re.compile(r"\b(instead|undo|meant|rather|forget|no|not the|switch|all of them|all the|again|show me all|swap|want|"
                     r"change)\b", re.I)


def _nrm(s) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()


def _names(pl) -> list:
    return [str(x) for x in pl] if isinstance(pl, list) else [str(pl)] if pl else []
# v3 (10-12, from the blind audit: category d in-scope non-answers were real matches 46/56): for "photos from <time/
# place/event>" the generator added visual filters the owner never said ("photos from the conference in London" ->
# only photos with a microphone). In category d a yes/no question is kept only if one of its content words appears in
# the request.
_STOP = set("is are was were there this that these those the a an any of in on at it its photo photos image picture "
            "pictures visible real live not or and with someone somebody person people does do anything something "
            "scene shown showing show taken being some has have who which what where from into onto by for to as one "
            "two several many group".split())


FROM_MOMENT = re.compile(r"^\W*(?:(?:can you |please )?(?:show me|find|pull up|get|give me)\s+)?(?:all\s+)?(?:the |our |my )?"
                         r"(?:photos|pictures|pics|shots|images|everything|all)( we took| i took)? (from|during|on)\b", re.I)


def _content(q):
    from common import norm
    return {norm(w) for w in re.findall(r"[a-zA-Z]{3,}", q.lower()) if w not in _STOP}


def strip_hidden_visual(albums, words):
    from common import norm
    W = {norm(w) for w in re.findall(r"[a-zA-Z]{3,}", words.lower())}
    W4 = {w[:4] for w in W}
    keep = lambda q: bool({c for c in _content(q) if c in W or c[:4] in W4})
    out, gone = [], []
    for a in albums or []:
        if not isinstance(a, dict):
            out.append(a); continue
        a = dict(a)
        for k in ("need", "need_any"):
            qs = [q for q in (a.get(k) or []) if isinstance(q, str)]
            kept = [q for q in qs if keep(q)]
            gone += [f"{k}:{q}" for q in qs if q not in kept]
            a[k] = kept
        out.append(a)
    return out, gone
MAX_FRAC = 0.25
EVENT_GAP_H = 3.0


def events(t: pd.Series) -> np.ndarray:
    order = np.argsort(t.values)
    ts = t.values[order]
    gaps = np.r_[True, (np.diff(ts) / np.timedelta64(1, "h")) > EVENT_GAP_H]
    ev = np.empty(len(t), np.int64); ev[order] = np.cumsum(gaps)
    return ev


class Lib:
    def __init__(self, user, it):
        L = it[it.user == user].sort_values("t").reset_index(drop=True)
        self.L, self.user = L, user
        self.ids = L.item_id.to_numpy()
        self.pos = {i: k for k, i in enumerate(self.ids)}
        self.ev = events(L.t)
        self.city = L.place.map(city).to_numpy()
        self.addr = L.place.fillna("").str.lower()
        self.addr_n = L.place.fillna("").map(_nrm)                          # accent-free, casefolded (RESULTS 52)
        self.addr_ns = self.addr_n.str.replace(r"\W+", "", regex=True)
        self._place_names = None
        O = pd.read_parquet(ROOT / "oracle" / f"{user}.parquet")
        self.O = {q: dict(zip(g.item_id.astype(str), g.p)) for q, g in O.groupby("q")}
        pf = ROOT / "oracle" / f"{user}_persons.json"
        self.persons = json.loads(pf.read_text()) if pf.exists() else {}

    cut = CUT

    def yes(self, q):
        d = self.O.get(q.strip())
        if d is None:
            raise KeyError(f"no oracle answers for {q!r}")
        return np.array([d.get(i, 0.0) >= self.cut for i in self.ids])

    def pyes(self, key, ids_ok):
        d = self.O.get(key)
        if d is None:
            raise KeyError(f"no oracle answers for {key!r}")
        return np.array([ok and d.get(i, 0.0) >= self.cut for i, ok in zip(self.ids, ids_ok)])

    def window(self, rows, w):
        t = self.L.t
        rows = list(rows)
        m = np.zeros(len(t), bool)
        if w == "same_day":
            m = t.dt.date.isin(set(t.iloc[rows].dt.date)).to_numpy()
        elif w == "same_week":
            wk = t.dt.isocalendar(); key = list(zip(wk.year, wk.week)); keys = {key[r] for r in rows}
            m = np.array([k in keys for k in key])
        elif w == "same_event":
            m = np.isin(self.ev, self.ev[rows])
        elif w == "same_place":
            cs = {self.city[r] for r in rows} - {""}
            m = np.isin(self.city, list(cs))
        elif w in ("before_in_event", "after_in_event", "before", "after"):
            for e in set(self.ev[rows]):
                first = t.iloc[[r for r in rows if self.ev[r] == e]].min()
                same = self.ev == e
                m |= same & ((t < first) if w.startswith("before") else (t > first)).to_numpy()
        elif w in ("since", "until"):
            first = t.iloc[rows].min()
            m = ((t > first) if w == "since" else (t < first)).to_numpy()
        elif re.fullmatch(r"days_(before|after):\d+", w or ""):
            n = int(w.split(":")[1]); d = t.dt.normalize()
            for a in set(d.iloc[rows]):
                lo, hi = (a - pd.Timedelta(days=n), a - pd.Timedelta(days=1)) if "before" in w else \
                         (a + pd.Timedelta(days=1), a + pd.Timedelta(days=n))
                m |= ((d >= lo) & (d <= hi)).to_numpy()
        elif re.fullmatch(r"minutes_(before|after):\d+", w or ""):
            n = pd.Timedelta(minutes=int(w.split(":")[1]))
            for r in rows:
                a = t.iloc[r]
                m |= (((t >= a - n) & (t < a)) if "before" in w else ((t > a) & (t <= a + n))).to_numpy()
        else:
            raise ValueError(f"unknown window {w!r}")
        return m

    def place_mask(self, name):
        """RESULTS 52 matcher: every word of the name in the address (accent/case-insensitive, unicode words), or the
        name written without spaces ("Machu Picchu" = "Machupicchu"). A list = all of its names (as before)."""
        m = np.ones(len(self.ids), bool)
        for nm in _names(name):
            toks = [w for w in re.findall(r"\w+", _nrm(nm)) if len(w) > 1]
            mm = np.ones(len(self.ids), bool)
            for w in toks:
                mm &= self.addr_n.str.contains(rf"\b{re.escape(w)}\b", regex=True).to_numpy()
            joined = "".join(toks)
            if len(toks) >= 2 and len(joined) >= 6:
                mm |= self.addr_ns.str.contains(joined, regex=False).to_numpy()
            m &= mm
        return m

    def place_names(self):
        """Every address name of the library (>= 2 photos, any level, not generic words) + landmark names from the
        photo facts: what "the words name a place" is checked against (RESULTS 52)."""
        if self._place_names is None:
            from collections import Counter as _C
            from common import load_facts, terms
            c = _C()
            for p in self.L.place.fillna(""):
                c.update({_nrm(x.strip()) for x in p.split(",") if x.strip() and not re.search(r"\d", x)})
            names = {x for x, n in c.items() if n >= 2 and len(x) >= 4 and x not in _GENERIC_PLACE}
            for f in load_facts(self.user).values():
                names |= {_nrm(t.split(":", 1)[1]) for t in terms(f) if t.startswith("landmark:")}
            self._place_names = {x for x in names if len(x) >= 4 and x not in _GENERIC_PLACE}
        return self._place_names

    def named_places(self, words):
        """Place names (place_names) the words contain: as a phrase, or with a space the other way round ("Machu
        Picchu" in the words = "Machupicchu" in the addresses)."""
        w = _nrm(words)
        toks = re.findall(r"\w+", w)
        joined2 = {a + b for a, b in zip(toks, toks[1:])}
        out = set()
        for x in self.place_names():
            xs = re.sub(r"\W+", "", x)
            if re.search(r"\b" + re.escape(x) + r"\b", w) or (len(xs) >= 6 and (xs in joined2 or (" " in x and xs in toks))):
                out.add(x)
        return out

    def meta_mask(self, a, loose=False):
        t = self.L.t
        m = np.ones(len(t), bool)
        if a.get("place_any"):
            anym = np.zeros(len(t), bool)
            for nm in _names(a["place_any"]):
                anym |= self.place_mask(nm)
            m &= anym
        if a.get("date_from"):
            m &= (t >= pd.Timestamp(a["date_from"])).to_numpy()
        if a.get("date_to"):
            m &= (t < pd.Timestamp(a["date_to"])).to_numpy()
        if a.get("hours"):
            h0, h1 = [int(x[:2]) * 60 + int(x[3:5]) for x in a["hours"].split("-")]
            mins = (t.dt.hour * 60 + t.dt.minute).to_numpy()
            m &= ((mins >= h0) & (mins < h1)) if h0 < h1 else ((mins >= h0) | (mins < h1))
        if a.get("place") and loose:
            m &= self.place_mask(a["place"])
        elif a.get("place"):
            pl = " ".join(map(str, a["place"])) if isinstance(a["place"], list) else str(a["place"])
            for w in [w for w in re.sub(r"[^a-z0-9 ]+", " ", pl.lower()).split() if len(w) > 1]:
                m &= self.addr.str.contains(rf"\b{re.escape(w)}\b", regex=True).to_numpy()
        if a.get("days"):
            m &= t.dt.date.astype(str).isin(set(a["days"])).to_numpy()
        return m

    def album(self, a, cat):
        """-> (mask, problems list)."""
        probs = []
        m = self.meta_mask(a, loose=(cat == "h" and H_FIX) or (cat == "g" and G_FIX) or (cat == "b" and B_FIX))
        if cat == "f" and a.get("ids"):
            tgt = [str(i) for i in a["ids"] if str(i) in self.pos]
            if not tgt:
                return None, ["f: target ids not in library"]
            fit = np.ones(len(m), bool)
            for q in a.get("need") or []:
                fit &= self.yes(q)
            tm = np.isin(self.ids, tgt)
            if not fit[tm].all():
                probs.append("f: target fails its own description")
            if (fit & ~tm).sum() > 6 - len(tgt):
                probs.append(f"f: description fits {int((fit & ~tm).sum())} other photos")
            return tm | (fit & m), probs
        for q in a.get("need") or []:
            m &= self.yes(q)
        if a.get("need_any"):
            anym = np.zeros(len(m), bool)
            for q in a["need_any"]:
                anym |= self.yes(q)
            m &= anym
        for q in a.get("not") or []:
            m &= ~self.yes(q)
        if a.get("person"):
            ok = np.isin(self.ids, list(self.persons.get(a["person"], {})))
            if not ok.any():
                probs.append(f"person {a['person']} not found")
            m &= ok
            for q in a.get("person_need") or []:
                m &= self.pyes(f"PERSON[{a['person']}]::{q.strip()}", ok)
        else:
            # no named person ("the women in saris"): the person condition is asked of the whole photo. Before 10-12
            # it was skipped, so the truth was every photo of the album's (hidden) day/place (RESULTS 49b/49c).
            for q in a.get("person_need") or []:
                if isinstance(q, str) and q.strip():
                    m &= self.yes(whole_photo_q(q))
        anc = a.get("anchor")
        if isinstance(anc, dict) and anc.get("window"):
            ids = [str(i) for i in anc.get("ids") or [] if str(i) in self.pos]
            if not ids:
                return None, ["anchor ids not in library"]
            rows = [self.pos[i] for i in ids]
            W = self.window(rows, anc["window"])
            if anc.get("need"):
                am = np.ones(len(m), bool)
                for q in anc["need"]:
                    am &= self.yes(q)
                found = np.where(am)[0]
                if not am[rows].any():
                    probs.append("anchor description does not find the anchor photo")
                elif len(found) and not np.array_equal(self.window(found, anc["window"]), W):
                    probs.append(f"anchor ambiguous: description finds {len(found)} photos in other windows")
            else:
                probs.append("anchor has no description")
            m &= W
        return m, probs


def strip_hidden(albums, words, cat):
    """-> (albums, list of stripped constraints)."""
    out, gone = [], []
    for a in albums or []:
        if not isinstance(a, dict):
            out.append(a); continue
        a = dict(a)
        if cat not in ("e", "f"):
            pl = a.get("place")
            if pl:
                pl = " ".join(map(str, pl)) if isinstance(pl, list) else str(pl)
                if not any(w.lower() in words.lower() for w in re.findall(r"[A-Za-z]{3,}", pl)):
                    gone.append(f"place:{pl}"); a["place"] = None
            if cat in ("a", "b", "c", "g", "h") and not TIME_CUE.search(words):
                for k in ("date_from", "date_to", "days", "hours"):
                    if a.get(k):
                        gone.append(f"{k}:{a[k]}"); a[k] = None
        out.append(a)
    return out, gone


def strip_hidden_h(albums, words, lib):
    """RESULTS 52 strip for category h (H_FIX): -> (albums, stripped). A place constraint stays when the words name any
    place (lib.named_places, or "the winery / hospital / zoo ..."); dates/days/hours stay when the words carry a time cue
    or name a place that the album's place fields do not cover (the dates encode it)."""
    out, gone = [], []
    named = lib.named_places(words)
    noun = bool(PLACE_NOUN.search(words))
    wn = _nrm(words)
    wns = re.sub(r"\W+", "", wn)

    def stated(nm):
        toks = [t for t in re.findall(r"\w+", _nrm(nm)) if len(t) >= 3]
        return any(re.search(rf"\b{re.escape(t)}\b", wn) for t in toks) or (len(re.sub(r"\W+", "", _nrm(nm))) >= 6 and re.sub(r"\W+", "", _nrm(nm)) in wns)

    def covers(names, x):
        xs = re.sub(r"\W+", "", x)
        return any(xs in re.sub(r"\W+", "", _nrm(n)) for n in names)
    for a in albums or []:
        if not isinstance(a, dict):
            out.append(a); continue
        a = dict(a)
        covered = set()
        for key in ("place", "place_any"):
            names = _names(a.get(key))
            if not names:
                continue
            if any(stated(n) for n in names) or named or noun:
                covered |= {x for x in named if covers(names, x)}
            else:
                gone.append(f"{key}:{a.get(key)}"); a[key] = None
        if not TIME_CUE.search(words) and not ((named - covered) or noun):
            for k in ("date_from", "date_to", "days", "hours"):
                if a.get(k):
                    gone.append(f"{k}:{a[k]}"); a[k] = None
        out.append(a)
    return out, gone


YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")


def year_problems(turns, progs):
    """RESULTS 52: a message that states a year must compile to albums that all carry a date constraint (the 27B rewrite
    dropped "in 2008" from 3 of 171 requests)."""
    return [f"h: turn {k + 1} states a year but an album has no date" for k, (t, albums) in enumerate(zip(turns, progs))
            if YEAR.search(str(t)) and any(isinstance(a, dict) and not (a.get("date_from") or a.get("date_to") or
                                                                       a.get("days") or a.get("anchor")) for a in albums or [])]


def g_year_problems(words, albums):
    """RESULTS 53: the words state a year but no album carries any date constraint (the time was dropped)."""
    if YEAR.search(words) and not any(isinstance(a, dict) and (a.get("date_from") or a.get("date_to") or a.get("days")
                                                                or a.get("anchor")) for a in albums or []):
        return ["g: the words state a year but no album has a date"]
    return []


def edit_problems(turns, answers):
    """RESULTS 52: a follow-up whose words only NARROW must return (within 10%) a subset of the previous turn; one whose
    words only WIDEN, a superset. Mixed or replacing edits ("actually ... instead", "undo") are not checked."""
    flat = [set().union(*map(set, s)) if s else set() for s in answers]
    out = []
    for k in range(1, min(len(flat), len(turns))):
        t = str(turns[k]); prev, cur = flat[k - 1], flat[k]
        nar, wid, rep = NARROW.search(t), WIDEN.search(t), REPLACE.search(t)
        if nar and not wid and not rep and cur and len(cur - prev) > 0.1 * len(cur):
            out.append(f"h: turn {k + 1} narrows but adds photos ({len(cur - prev)} of {len(cur)})")
        if wid and not nar and not rep and prev and len(prev - cur) > 0.1 * len(prev):
            out.append(f"h: turn {k + 1} widens but drops photos ({len(prev - cur)} of {len(prev)})")
    return out


def main():
    it = load_items()
    splits = json.loads((ROOT / "splits.json").read_text())
    split_of = {u: s for s, us in splits.items() if s in ("test", "dev", "train") for u in us}
    allq, stats = {"test": [], "dev": [], "train": []}, Counter()
    drops = []
    for pf in sorted((ROOT / "programs").glob("*.json")):
        u = pf.stem
        if split_of.get(u, "train") not in ONLY or not (ROOT / "oracle" / f"{u}.parquet").exists():
            continue
        P = load_programs(u)          # h: RESULTS 52 rewrite overlay (programs_h2/) unless FP_CB_H2=""
        lib = Lib(u, it)
        n = len(lib.ids)
        for qi, r in enumerate(P["requests"]):
            cat = r.get("cat")
            qid = f"{u}:{cat}{qi}"
            lib.cut = CUTS.get(cat, CUT)
            probs = []
            try:
                turns = r.get("turns") or []
                progs = r.get("turn_programs") if cat == "h" else [r.get("albums")]
                stripped = []
                if STRICT_WORDS and progs:
                    new = []
                    for k, albums in enumerate(progs):
                        w = " ".join(map(str, turns[:k + 1])) if cat == "h" else " ".join(map(str, turns))
                        if cat == "h" and H_FIX and cat in STRIP_CATS:
                            al, gone = strip_hidden_h(albums, w, lib)
                        else:
                            al, gone = strip_hidden(albums, w, cat) if cat in STRIP_CATS else (albums, [])
                        if cat == "d" and FROM_MOMENT.search(w):
                            al, g2 = strip_hidden_visual(al, w); gone += g2
                        new.append(al); stripped += gone
                    progs = new
                if cat == "h" and (not progs or len(progs) != len(turns)):
                    raise ValueError("h: turn_programs do not match turns")
                answers = []
                for albums in progs:
                    sets = []
                    for a in albums or []:
                        mk, pb = lib.album(a, cat)
                        probs += pb
                        sets.append([] if mk is None else lib.ids[mk].tolist())
                    answers.append(sets)
                if not turns or not answers or not answers[-1]:
                    raise ValueError("empty request or program")
                fin = answers[-1]
                if cat == "g" and len(fin) != 2:
                    probs.append("g: not 2 albums")
                if cat not in (("g", "h") if H_FIX else ("g",)) and len(fin) != 1:   # h: a turn's albums = their union
                    probs.append(f"{len(fin)} albums for a one-album category")
                for s in ([sorted(set().union(*map(set, fin)))] if cat == "h" and H_FIX else fin):   # h: the union
                    if not s:
                        probs.append("empty answer")
                    if len(s) > max(150, MAX_FRAC * n) or len(s) > MAX_FRAC * n:
                        probs.append(f"answer too large ({len(s)})")
                if (cat == "g" and G_FIX) or (cat == "b" and B_FIX):
                    probs += g_year_problems(" ".join(map(str, turns)), progs[-1])
                if cat == "g" and len(fin) == 2 and fin[0] and fin[1]:
                    ov = len(set(fin[0]) & set(fin[1])) / min(len(fin[0]), len(fin[1]))
                    if ov > 0.2:
                        probs.append(f"g: albums overlap {ov:.0%}")
                if cat == "h":
                    flat = [sorted(set().union(*map(set, s))) for s in answers]
                    if any(not f for f in flat):
                        probs.append("h: a turn has an empty answer")
                    if len(flat) >= 2 and flat[-1] == flat[0]:
                        probs.append("h: edits change nothing")
                    if H_FIX:
                        probs += edit_problems(turns, answers) + year_problems(turns, progs)
            except Exception as e:
                probs.append(f"error: {type(e).__name__}: {str(e)[:120]}")
                answers = None
            prog_out = (progs if cat == "h" else progs[0]) if answers is not None and progs else \
                (r.get("turn_programs") if cat == "h" else r.get("albums"))
            rec = dict(qid=qid, user=u, split=split_of.get(u), cat=cat, today=P["today"], turns=r.get("turns"),
                       program=prog_out, why=r.get("why"), answers=answers, library_size=n,
                       stripped=stripped if answers is not None else [], verified=cat in VERIFIED,
                       cut=lib.cut)
            if cat == "h" and r.get("turn_programs_v1") is not None:   # RESULTS 52 rewrite (words unchanged)
                rec["program_version"] = "h2"
            if cat in ("g", "b") and r.get("albums_v1") is not None:   # RESULTS 53 (g) / 56 (b) rewrite (words unchanged)
                rec["program_version"] = cat + "2"
            if probs:
                stats[(cat, "dropped")] += 1
                drops.append(dict(qid=qid, cat=cat, turns=r.get("turns"), problems=probs))
            else:
                stats[(cat, "kept")] += 1
                allq[split_of.get(u, "train")].append(rec)
    (ROOT / QDIR).mkdir(exist_ok=True)
    for s, qs in allq.items():
        if s not in ONLY:
            continue
        (ROOT / QDIR / f"{s}.json").write_text(json.dumps(dict(split=s, cut=CUT, cuts=CUTS, queries=qs), indent=1))
    (ROOT / QDIR / "dropped.json").write_text(json.dumps(drops, indent=1))
    for c in "abcdefgh":
        print(c, "kept", stats[(c, "kept")], "dropped", stats[(c, "dropped")])
    print({s: len(q) for s, q in allq.items()})
    print(Counter(p.split(":")[0] if p.startswith("error") else re.sub(r"\d+", "N", p)[:50] for d in drops for p in d["problems"]).most_common(15))
    print(Counter((d["cat"], re.sub(r"\d+", "N", d["problems"][0])[:40]) for d in drops).most_common(25))


if __name__ == "__main__":
    main()
