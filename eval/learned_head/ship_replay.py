"""RESULTS 48: the cosine + learned-head BLEND as fast mode's order, replayed with the shipped stop rule on stored judge
answers (no model runs; extends rank_replay.py / blend.py).

  python eval/learned_head/ship_replay.py SETTING SOURCE [SOURCE ...]  -> scale/ship_replay_{SETTING}_{TAG}.tsv/.txt
SETTING  m48    audit4 exactly as the M48-scale replay ran it: beach "Is this a photo of a beach?" at 0.999, food at
                0.9999 (the default since b2eccec), stored P (eval/recall_audit/scores_q3vl4b_real); head texts as trained
         pre47  the same with food at 0.999 too (the plan before RESULTS 47)
         r47    the plan shipped in RESULTS 47: beach asks "Is a beach clearly visible in this photo?" (judge P from
                eval/scene_questions/p.tsv g_vis via FINDPICS_P_OVERRIDE, AND the head reads the new question's text:
                scores_{RUN}_r47.parquet from head_ship.py rescore), food at 0.9999. ev16 stores only 0.7 yes sets, so
                its judge labels do not change; only the head's beach text does.
         r47i8  r47 with every head text from the int8-simulated text tower (the phone's pe_core_text is int8)
SOURCE   RUN:COL, e.g. JDonly:mlp (3-seed mean, = the scale report), JDonly:mlp_s0 (the shipped single seed),
         JDonly-na:mlp_s1 (no-|img-txt| variant, seed 1)
Blend = z(look) + z(logit head), equal weights (blend.py), with z computed two ways:
  pooled  per search over all its libraries (the M48-scale replay, blend.py)
  perlib  per search per original library
The PHONE computes z over the photos of the library it searches = one replay case. audit4 replays its 4 libraries as
ONE 7,886-photo library (one case per search), so there the phone's z is `pooled` (perlib would z-score sub-libraries
that the ranking then mixes); ev16 cases are single libraries, so there the phone's z is `perlib`. Column `phone` marks
those rows.
Baselines: shipped = the order the earlier replays used (audit4: PE-Core-B-16 on the original files, looks_b16;
ev16: PE-Core-L-14 server index, ev16_ranked.parquet); look = the phone's B-16 cosine on phone-simulated vectors (the
same `look` the blend uses). ev16 is reported WITHOUT selfies (no look score there; the M46/M50 detector handles them).
Metrics (rank_replay.py): (a) judge calls in rank order to find 80% / 90% of the judge's yes; (b) shipped fast mode, 20
tail seeds: first-answer recall and calls, and calls for the blend's order to reach the number the baseline's first
answer found (same seed) -> saved %.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval"); sys.path.insert(0, "eval/learned_head")
import rank_replay as RR  # noqa: E402
import score_apple_photos as SAP  # noqa: E402
import tune_stop_rule as T  # noqa: E402

S = Path("eval/learned_head/scale")


def setting(name):
    SAP.SUBJECT_CUT = 0.999
    if name in ("pre47", "m48"):
        SAP.FOOD_CUT = 0.999 if name == "pre47" else 0.9999; os.environ["FINDPICS_P_OVERRIDE"] = ""
    else:
        SAP.FOOD_CUT = 0.9999; os.environ["FINDPICS_P_OVERRIDE"] = "beach photos=g_vis"
    a4 = T.audit4()
    for c in a4:
        c["qkey"] = c["key"]; c["lib"] = None
    ev = [c for c in T.ev16() if c["query"] != "selfies"]
    for c in ev:
        c["qkey"] = (c["query"], c["user"])
    return a4, ev


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def blend_scores(X, col, mode):
    """{(qkey, item): blended score} for A4 (qkey = query) and EV16 (qkey = (query, lib))."""
    X = X[X.src.isin(["A4", "EV16"])].reset_index(drop=True)
    p = np.clip(X[col].to_numpy(float), 1e-6, 1 - 1e-6)
    X = X.assign(_lg=np.log(p / (1 - p)), _b=np.nan)
    keys = ["src", "query"] + (["lib"] if mode == "perlib" else [])
    out = np.full(len(X), np.nan)
    for _, g in X.groupby(keys, sort=False):
        if g.look.isna().any():
            continue                                        # selfies: no cosine; not replayed
        out[g.index] = z(g.look) + z(g._lg)
    X["_b"] = out
    sc = {}
    for src, d in X.groupby("src"):
        if src == "A4":
            sc.update(zip(zip(d["query"], d.item.astype(str)), d._b))
        else:
            sc.update(zip(zip(zip(d["query"], d.lib), d.item.astype(str)), d._b))
    return sc


def look_scores(X):
    X = X[X.src.isin(["A4", "EV16"]) & X.look.notna()]
    sc = {}
    for src, d in X.groupby("src"):
        k = d["query"] if src == "A4" else list(zip(d["query"], d.lib))
        sc.update(zip(zip(k, d.item.astype(str)), d.look))
    return sc


def main():
    st, sources = sys.argv[1], sys.argv[2:]
    tag = os.environ.get("SR_TAG", "all")
    a4, ev = setting(st)
    say = []
    rows = []
    first_X = None
    base = {}
    for src_ in sources:
        run, col = src_.split(":")
        suf = "" if st in ("pre47", "m48") else "_" + st
        X = pd.read_parquet(S / f"scores_{run}{suf}.parquet")
        if first_X is None:
            first_X = X
            lk = look_scores(X)
            for setname, cases in (("audit4", a4), ("ev16", ev)):
                for bname, bc in (("shipped", cases), ("look", RR.reorder(cases, lk))):
                    b, per = RR.shipped(bc)
                    base[(setname, bname)] = (bc, b, per)
                    yes = sum(int(c["y"].sum()) for c in bc)
                    say.append(f"[{st}] {setname} baseline {bname}: {len(bc)} cases, {yes} judge-yes; first answer recall "
                               f"{b['recall']:.3f} with {b['calls']:.0f} calls; 80%/90% in rank order "
                               f"{RR.calls_to(bc, .8)} / {RR.calls_to(bc, .9)}")
                    print(say[-1], flush=True)
        for mode in ("pooled", "perlib"):
            sc = blend_scores(X, col, mode)
            for setname, cases in (("audit4", a4), ("ev16", ev)):
                hc = RR.reorder(cases, sc)
                for bname in ("shipped", "look"):
                    bc, b, per = base[(setname, bname)]
                    h, _ = RR.shipped(hc, base=per)
                    r = dict(setting=st, source=src_, z=mode, set=setname, baseline=bname,
                             phone=(setname == "audit4") == (mode == "pooled"),
                             base_recall=b["recall"], base_calls=b["calls"], blend_recall=h["recall"],
                             blend_calls=h["calls"], match_calls=h["match_calls"],
                             saved=1 - h["match_calls"] / b["calls"],
                             a80_base=RR.calls_to(bc, .8), a80_blend=RR.calls_to(hc, .8),
                             a90_base=RR.calls_to(bc, .9), a90_blend=RR.calls_to(hc, .9))
                    rows.append(r)
                    say.append(f"[{st}] {src_:18s} z={mode:6s}{'*' if r['phone'] else ' '} {setname:6s} vs {bname:7s}: first answer recall "
                               f"{b['recall']:.3f} -> {h['recall']:.3f}; calls to match {b['calls']:.0f} -> "
                               f"{h['match_calls']:.0f} ({100 * r['saved']:+.1f}% saved); 80% {r['a80_base']} -> "
                               f"{r['a80_blend']}, 90% {r['a90_base']} -> {r['a90_blend']}")
                    print(say[-1], flush=True)
                if os.environ.get("SR_PER_SEARCH") == src_ and (setname == "audit4") == (mode == "pooled"):
                    bc, b, per = base[(setname, "look")]
                    for q in sorted({c["key"] if setname == "audit4" else c["query"] for c in cases}):
                        sel = [k for k, c in enumerate(bc) if (c["key"] if setname == "audit4" else c["query"]) == q]
                        cq = [bc[k] for k in sel]; hq = [hc[k] for k in sel]
                        bq, pq = RR.shipped(cq); hq_, _ = RR.shipped(hq, base=pq)
                        say.append(f"      per search {setname} {q:24s} judge-yes {sum(int(c['y'].sum()) for c in cq):5d}; "
                                   f"first answer recall {bq['recall']:.3f} -> {hq_['recall']:.3f}; calls to match "
                                   f"{bq['calls']:.0f} -> {hq_['match_calls']:.0f}")
                        print(say[-1], flush=True)
    pd.DataFrame(rows).to_csv(S / f"ship_replay_{st}_{tag}.tsv", sep="\t", index=False, float_format="%.4f")
    (S / f"ship_replay_{st}_{tag}.txt").write_text("\n".join(say) + "\n")
    print("SHIP_REPLAY_DONE", st, tag, flush=True)


if __name__ == "__main__":
    main()
