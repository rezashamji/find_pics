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
from common import ROOT, city, load_items, whole_photo_q  # noqa: E402

import os as _os
CUT = float(_os.environ.get("FP_CB_CUT", "0.5"))
# per-category cut, e.g. FP_CB_CUTS="a=0.9,b=0.9" (oracle P(yes) needed for a yes; chosen on DEV-library audit labels)
# final per-category cuts (10-12): chosen on the EVEN-index audited requests, reported on the ODD half (audit/quality_*.txt)
FINAL_CUTS = "a=0.7,b=0.8,c=0.5,d=0.8,e=0.5,f=0.9,g=0.5,h=0.9"
CUTS = {k: float(v) for k, v in (x.split("=") for x in _os.environ.get("FP_CB_CUTS", FINAL_CUTS if "FP_CB_CUT" not in _os.environ else "").split(",") if x)}
# categories whose candidate sets passed (or nearly passed) the audit bar precision ~0.8 / recall 0.7; c, g, h are provisional
VERIFIED = set(_os.environ.get("FP_CB_VERIFIED", "a,b,d,e,f").split(","))
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

    def meta_mask(self, a):
        t = self.L.t
        m = np.ones(len(t), bool)
        if a.get("date_from"):
            m &= (t >= pd.Timestamp(a["date_from"])).to_numpy()
        if a.get("date_to"):
            m &= (t < pd.Timestamp(a["date_to"])).to_numpy()
        if a.get("hours"):
            h0, h1 = [int(x[:2]) * 60 + int(x[3:5]) for x in a["hours"].split("-")]
            mins = (t.dt.hour * 60 + t.dt.minute).to_numpy()
            m &= ((mins >= h0) & (mins < h1)) if h0 < h1 else ((mins >= h0) | (mins < h1))
        if a.get("place"):
            pl = " ".join(map(str, a["place"])) if isinstance(a["place"], list) else str(a["place"])
            for w in [w for w in re.sub(r"[^a-z0-9 ]+", " ", pl.lower()).split() if len(w) > 1]:
                m &= self.addr.str.contains(rf"\b{re.escape(w)}\b", regex=True).to_numpy()
        if a.get("days"):
            m &= t.dt.date.astype(str).isin(set(a["days"])).to_numpy()
        return m

    def album(self, a, cat):
        """-> (mask, problems list)."""
        probs = []
        m = self.meta_mask(a)
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
        P = json.loads(pf.read_text())
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
                if cat != "g" and len(fin) != 1:
                    probs.append(f"{len(fin)} albums for a one-album category")
                for s in fin:
                    if not s:
                        probs.append("empty answer")
                    if len(s) > max(150, MAX_FRAC * n) or len(s) > MAX_FRAC * n:
                        probs.append(f"answer too large ({len(s)})")
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
            except Exception as e:
                probs.append(f"error: {type(e).__name__}: {str(e)[:120]}")
                answers = None
            prog_out = (progs if cat == "h" else progs[0]) if answers is not None and progs else \
                (r.get("turn_programs") if cat == "h" else r.get("albums"))
            rec = dict(qid=qid, user=u, split=split_of.get(u), cat=cat, today=P["today"], turns=r.get("turns"),
                       program=prog_out, why=r.get("why"), answers=answers, library_size=n,
                       stripped=stripped if answers is not None else [], verified=cat in VERIFIED,
                       cut=lib.cut)
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
