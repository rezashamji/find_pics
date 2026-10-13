"""RESULTS 55 (dev only): per-photo evidence for every dev category-b request, for classifying failures by cause.

Replays the shipped stack's dev plans (runs/m54/dev_s1_plan, the 10-05 adapter on the phone's 4-bit weights, code at
origin/main) with the CACHED 9B judge only (no model; a cache miss is an error), exactly as run_system.py --judge=cache,
and records for each request and each photo of its library:
  ret_first / ret_final   returned in the first answer / final round (replay; checked against the m54 run's sets)
  judged, p_main          judged in the final round, and the 9B's P(yes) for the album question
  p_main_any              the 9B's P for the album question from ANY cached run (also photos outside the plan's scope)
  p_filter / p_excl       the 9B's P for the album's filter / exclude question (cache lookup; NaN = never asked)
  fast, fast_rank         PE-Core look score (engine.look_scores) and its rank among the in-scope photos (0 = best)
  in_truth, truth_scope   in the candidate answer set; inside the truth program's date/day/place scope
  o<k>                    the 27B oracle's P(yes) for the truth program's k-th need question (truth = all >= 0.8)
plus per-photo pixel statistics (pix.parquet): colourfulness, saturation, hue fractions, blur, exposure.
Usage (CPU is enough): PYTHONPATH=src python eval/complex_bench/b55/dump.py [cat=b] -> eval/complex_bench/b55/dump_<cat>/
"""
import glob
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

CAT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("cat=")), "b")
OUT = ROOT / "b55" / f"dump_{CAT}"
RUN = ROOT / "runs" / "m54" / "m54_dev_s1"
PLANS = ROOT / "runs" / "m54" / "dev_s1_plan" / "plans.json"
JC = ROOT / "runs" / "cmp" / "jcache"
NAMES = ["Sam", "Alex"]


def pixel_stats(im) -> dict:
    """Colour / blur / exposure statistics of one RGB image at the pixels the judge sees (DISBench <= 500 px)."""
    a = np.asarray(im, np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    chroma = mx - mn                                   # 0 = grey
    v = mx
    s = np.where(mx > 1e-6, chroma / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # hue in degrees
    h = np.zeros_like(mx)
    m = chroma > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = (60 * ((g - b)[rm] / chroma[rm])) % 360
    h[gm] = 60 * ((b - r)[gm] / chroma[gm]) + 120
    h[bm] = 60 * ((r - g)[bm] / chroma[bm]) + 240
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Hasler-Suesstrunk colourfulness
    rg, yb = r - g, 0.5 * (r + g) - b
    colourful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    # blur: variance of the Laplacian of the luminance (4-neighbour), on the image as loaded
    L = lum * 255
    lap = (-4 * L[1:-1, 1:-1] + L[:-2, 1:-1] + L[2:, 1:-1] + L[1:-1, :-2] + L[1:-1, 2:])
    gx = np.abs(np.diff(L, axis=1)).mean(); gy = np.abs(np.diff(L, axis=0)).mean()
    sat_px = (s > 0.35) & (v > 0.25)
    out = dict(w=im.width, h=im.height, chroma_mean=float(chroma.mean()), chroma_p95=float(np.percentile(chroma, 95)),
               frac_chroma_gt10=float((chroma > 10 / 255).mean()), frac_chroma_gt30=float((chroma > 30 / 255).mean()),
               sat_mean=float(s.mean()), colourful=colourful, lum_mean=float(lum.mean()),
               frac_clip_hi=float((lum > 250 / 255).mean()), frac_clip_lo=float((lum < 8 / 255).mean()),
               lap_var=float(lap.var()), grad_mean=float((gx + gy) / 2), frac_sat=float(sat_px.mean()))
    for k in range(12):   # fraction of ALL pixels that are saturated and in hue bin [30k, 30k+30)
        out[f"hue{k * 30:03d}"] = float((sat_px & (h >= 30 * k) & (h < 30 * k + 30)).mean())
    return out


def main():
    from findpics import store
    from findpics.converse import CachedJudge, Plan, stream_plan
    from findpics.engine import Thresholds, look_scores, _judge_rows
    from findpics.media import load_image
    from findpics.models import ImageTextEncoder
    from findpics.people import expand_refs
    import build
    OUT.mkdir(parents=True, exist_ok=True)
    Q = [q for q in json.loads((ROOT / "queries" / "dev.json").read_text())["queries"] if q["cat"] == CAT]
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("users=")), None)
    if only:
        Q = [q for q in Q if q["user"] in only.split(",")]
    plans = json.loads(PLANS.read_text())["plans"]
    runs = {r["qid"]: r for f in sorted(RUN.glob("part*.json")) for r in json.loads(f.read_text())}
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    from common import load_items
    it = load_items()

    class CacheOnly:
        def p_yes(self, images, question):
            raise RuntimeError(f"judge cache miss ({len(images)} images): {question[:80]}")

        def text(self, prompt, **kw):
            raise RuntimeError("cache only")

    log = open(OUT / "log.txt", "a")
    for u in sorted({q["user"] for q in Q}):
        sub = store.subset(idx, np.where(user_of == u)[0])
        cache = {}
        for f in sorted(glob.glob(str(JC / "*" / f"{u}.json"))):
            try:
                cache.update(json.loads(Path(f).read_text()))
            except json.JSONDecodeError:
                pass
        J = CachedJudge(CacheOnly(), None)
        J.cache = cache
        ids = sub.items.item_id.astype(str).to_numpy()
        # every photo once: pixel stats + the raw bytes' md5 state for cache keys
        ims = {}
        pix = []
        for k, p in enumerate(sub.items.path):
            try:
                im = load_image(p)
            except Exception:
                im = None
            ims[k] = im
            if im is not None:
                pix.append(dict(item_id=ids[k], **pixel_stats(im)))
        pd.DataFrame(pix).to_parquet(OUT / f"pix_{u}.parquet")

        def lookup(question):
            if not question:
                return np.full(len(ids), np.nan)
            out = np.full(len(ids), np.nan)
            for k, im in ims.items():
                if im is None:
                    continue
                key = hashlib.md5(question.encode() + im.tobytes() + str(im.size).encode()).hexdigest()
                if key in cache:
                    out[k] = cache[key]
            return out

        lib = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        people = NAMES[:len(G)]

        def refs_for(person, G=G, sub=sub):
            for nm, g in zip(NAMES, G):
                if person and person.lower() == nm.lower():
                    refs = expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3)
                    return nm, refs, int(g["rep"]), len(g["item_ids"])
            raise ValueError(f"unknown person {person}")
        L = build.Lib(u, it)
        L.cut = build.CUTS.get(CAT, build.CUT)
        for q in [q for q in Q if q["user"] == u]:
            sp = plans[q["qid"]]
            P = Plan.model_validate(sp["turns"][-1]["plan"])
            for a in P.albums:
                if a.person and a.person not in people:
                    a.person = None
                a.with_people = [w for w in a.with_people if w in people]
            rounds = list(stream_plan(sub, P, enc, J, refs_for, th=Thresholds(tail_budget=300, stream=True)))
            first = set().union(*[set(r.returned.item_id.astype(str)) for r in rounds[0]])
            final = set().union(*[set(r.returned.item_id.astype(str)) for r in rounds[-1]])
            run_final = set().union(*map(set, runs[q["qid"]]["turns"][-1]["final"]["albums"]))
            run_first = set().union(*map(set, runs[q["qid"]]["turns"][-1]["first"]["albums"]))
            same = (final == run_final, first == run_first)
            truth = set().union(*map(set, q["answers"][-1]))
            d = pd.DataFrame(dict(item_id=ids))
            d["ret_first"] = d.item_id.isin(first); d["ret_final"] = d.item_id.isin(final)
            d["in_truth"] = d.item_id.isin(truth)
            for ai, (a, r) in enumerate(zip(P.albums, rounds[-1])):
                j = r.judged.copy(); j["item_id"] = j.item_id.astype(str)
                pm = dict(zip(j.item_id, j.p_attr))
                d[f"judged{ai}"] = d.item_id.isin(pm)
                d[f"p_main{ai}"] = d.item_id.map(pm)
                d[f"p_main_any{ai}"] = lookup(a.judge_question)
                d[f"p_filter{ai}"] = lookup(a.filter_question)
                d[f"p_excl{ai}"] = lookup(a.exclude_question)
                fast = look_scores(sub, enc, a.looks, a.avoid)
                d[f"fast{ai}"] = fast
                sc = d[f"judged{ai}"].to_numpy()
                rk = np.full(len(d), np.nan)
                o = np.argsort(-fast[sc]); rr = np.empty(len(o)); rr[o] = np.arange(len(o)); rk[sc] = rr
                d[f"fast_rank{ai}"] = rk
            # truth program evidence (one album for b)
            tp = q["program"][0]
            Lpos = np.array([L.pos.get(i, -1) for i in ids])
            mm = L.meta_mask(tp)
            d["truth_scope"] = [bool(mm[p]) if p >= 0 else False for p in Lpos]
            for k, nq in enumerate(tp.get("need") or []):
                o = L.O.get(nq.strip(), {})
                d[f"o{k}"] = d.item_id.map(o)
            d["day"] = d.item_id.map(dict(zip(L.ids, L.L.t.dt.date.astype(str))))
            d["place"] = d.item_id.map(dict(zip(L.ids, L.L.place.fillna(""))))
            d.to_parquet(OUT / f"{q['qid'].replace(':', '_')}.parquet")
            meta = dict(qid=q["qid"], words=q["turns"], plan=json.loads(P.model_dump_json()), truth=tp,
                        replay_same_final=same[0], replay_same_first=same[1], n_truth=len(truth), n_final=len(final))
            (OUT / f"{q['qid'].replace(':', '_')}.json").write_text(json.dumps(meta, indent=1))
            print(q["qid"], "replay identical final/first:", same, len(final), len(truth), file=log, flush=True)
            print(q["qid"], same, flush=True)


if __name__ == "__main__":
    main()
