"""STRICTER JUDGE CUT: precision AND recall per search (no new model runs; public DISBench data only).

Judge = the phone's Qwen3-VL-4B at its real 4-bit weights, stored P(yes) on all 7,886 photos x 6 searches
(eval/recall_audit/scores_q3vl4b_real, float32). Truth = RESULTS 34's estimator (eval_recall.score_systems): blind eye
labels drawn uniformly inside the strata of the RESULTS 34 judge (eval/recall_audit/scores = the OLD simulated weights at
P >= 0.7; kept / A doubt / B top-5% vector / C rest), every earlier eye label exact. A stricter cut on the REAL judge's P
is just another "system": score_systems splits every stratum into cells by which systems returned the photo (nested cuts
-> cells = P bins inside a stratum); a uniform draw inside a stratum is uniform inside each cell, so each cell's real
count = cell size x its labelled match rate (the stratum's rate when the cell holds no labelled photo: reported below as
"imputed"). All cuts are scored jointly, so every cut's recall has the same estimated total of real photos.

Two label sets:
  V1  RESULTS 34 labels only (old + blind eye) = the estimator exactly as RESULTS 34/45 ran it.
  V2  V1 + the RESULTS 45 census (eval/vector_enough/labels_*.txt, source audit4). The census covers a FIXED set of
      photos (each search: vector top 50 and judged top 50 of the pooled library, each library's vector top 20 and
      judged top 20; judged = real P >= judgeCutoff) and every photo in it has a label. So that set is made an exact
      stratum (label order = eval_vector_enough.truth_audit4), and the uniform blind labels of each RESULTS 34 stratum
      are used only for the photos OUTSIDE it (a uniform draw restricted to a fixed subset is uniform on that subset).
Intervals: 95% bootstrap inside each stratum (score_systems; as RESULTS 34); Bayesian (Jeffreys) for recall too.

  python eval/eval_judge_cut_sweep.py      -> eval/judge_cut_sweep/report.txt + sweep.tsv
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
from eval_recall import QUERIES, _labels_eye, load_scores, old_labels, score_systems  # noqa: E402

OUT = Path("eval/judge_cut_sweep")
SH = {"photos with a dog": "dog", "photos with a car": "car", "photos with a bicycle": "bicycle",
      "beach photos": "beach", "sunset photos": "sunset", "food photos": "food"}
CUTS = [("0.7", 0.7), ("0.99", 0.99), ("0.999", 0.999), ("0.9999", 0.9999), ("0.99999", 0.99999),
        ("0.999999", 0.999999), ("==1.0", 1.0)]
F32_ONE_GAP = 25 * math.log(2)        # float32 P = 1/(1+e^-g) rounds to exactly 1.0 once 1-P < 2^-25 (half an ulp)


def gap_of(c):
    return F32_ONE_GAP if c >= 1.0 else math.log(c / (1 - c))


def real_scores():
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores_q3vl4b_real").glob("part*.parquet"))])
    sc["item_id"] = sc.item_id.astype(str)
    return sc


def keep_at(p, c):
    """The phone's test: Double(Float P) >= cut; '==1.0' = P is exactly 1.0 in float32."""
    p = p.astype(np.float64)
    return p == 1.0 if c >= 1.0 else p >= c


def census_exact(df, sc, old, eye):
    """V2 exact set: every photo of the RESULTS 45 census screens with its truth_audit4 label."""
    from eval_vector_enough import new_labels, screen_sets, truth_audit4
    from score_apple_photos import judge_cutoff
    new, recs = new_labels()
    a4 = {(r["query"], r["item_id"]) for r in recs if r["source"] == "audit4"}
    new = {k: v for k, v in new.items() if k in a4}
    lk = pd.read_parquet("eval/apple_photos/looks_b16.parquet").set_index("item_id")
    lk.index = lk.index.astype(str)
    st = df.set_index(["query", "item_id"]).stratum
    ex, nscreen, unl = {}, {}, {}
    for q in QUERIES:
        d = sc[sc["query"] == q].set_index("item_id")
        cut = judge_cutoff(d.question.iloc[0])
        d = d.assign(s=lk.loc[d.index, q].to_numpy(), y=d.p >= cut).sort_values("s", ascending=False, kind="stable")
        need = set()
        for S, per_lib in ((50, False), (20, True)):
            for _, g in (d.groupby("user") if per_lib else [("all", d)]):
                v, j = screen_sets(g, S)
                need |= set(v) | set(j)
        nscreen[q] = len(need)
        unl[q] = 0
        for i in need:
            v = truth_audit4(q, i, st[(q, i)], old, eye, new)
            if v is None:
                unl[q] += 1
                continue
            ex[(q, i)] = v
    return ex, nscreen, unl, len(new)


def cell_diag(df, P, exact, eye):
    """Per search and P bin: photos, exact-labelled, uniform-labelled, photos in (stratum, bin) cells with no uniform label
    (their rate is the stratum's: imputed)."""
    edges = [c for _, c in CUTS]
    rows = []
    for q in QUERIES:
        d = df[df["query"] == q].copy()
        p = P[q].reindex(d.item_id).to_numpy().astype(np.float64)
        b = np.zeros(len(d), int)
        for k, c in enumerate(edges):
            b[keep_at(p, c)] = k + 1
        d["bin"] = b
        d["ex"] = [(q, i) in exact for i in d.item_id]
        d["uni"] = [(not e) and (q, i) in eye for i, e in zip(d.item_id, d.ex)]
        for k in range(len(edges) + 1):
            g = d[d.bin == k]
            imp = 0
            for s, gs in g[~g.ex].groupby("stratum"):
                if gs.uni.sum() == 0 and len(gs):
                    imp += len(gs)
            rows.append(dict(query=q, bin=k, N=len(g), exact=int(g.ex.sum()), uniform=int(g.uni.sum()), imputed=imp))
    return pd.DataFrame(rows)


def bin_name(k):
    lo = "0" if k == 0 else CUTS[k - 1][0]
    hi = CUTS[k][0] if k < len(CUTS) else None
    if lo == "==1.0":
        return "P == 1.0"
    if hi == "==1.0":
        return f"[{lo}, 1.0)"
    return f"[{lo}, {hi})"


def other_libraries():
    """RESULTS 35 (2): earlier eye labels of OTHER libraries (biased: mostly judge disagreements), question 'Is this a
    photo of X?' re-judged by the OLD simulated weights (models/qwen3vl_4b_mlx4sim; no real-weight P exists for them)."""
    from eval_question_variants import labels
    lab = labels()
    sf = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/scene_fix").glob("part*.parquet"))], ignore_index=True)
    sf = sf[sf.qname == "q0"]
    out = []
    for q, d in sf.groupby("query"):
        L = np.array([lab.get((q, str(i))) for i in d.item_id], object)
        p = d.p.to_numpy()
        for name, c in CUTS:
            k = keep_at(p, c)
            out.append(dict(query=q, cut=name, **{f"{v}_kept": int((k & (L == v)).sum()) for v in ("right", "wrong", "unsure")},
                            **{f"{v}_all": int((L == v).sum()) for v in ("right", "wrong", "unsure")}))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_scores()
    df["item_id"] = df.item_id.astype(str)
    sc = real_scores()
    P = {q: sc[sc["query"] == q].set_index("item_id").p for q in QUERIES}
    for q in QUERIES:
        assert set(P[q].index) == set(df[df["query"] == q].item_id), q
    old, eye = old_labels(), _labels_eye()
    cex, nscreen, unl, n_new = census_exact(df, sc, old, eye)
    old2 = dict(cex)
    old2.update({k: v for k, v in old.items()})               # earlier labels keep priority (truth_audit4 order)
    # systems: nested cuts + disjoint P bins (bins are unions of the same cells: no change to the estimate)
    systems = {}
    for q in QUERIES:
        p = P[q]
        for name, c in CUTS:
            systems.setdefault(f"cut {name}", {})[q] = set(p.index[keep_at(p.to_numpy(), c)])
        for k in range(len(CUTS), -1, -1):
            hi = keep_at(p.to_numpy(), CUTS[k][1]) if k < len(CUTS) else np.zeros(len(p), bool)
            lo = keep_at(p.to_numpy(), CUTS[k - 1][1]) if k > 0 else np.ones(len(p), bool)
            systems.setdefault(f"bin {bin_name(k)}", {})[q] = set(p.index[lo & ~hi])
    R = {}
    for tag, o, um in (("V1", old, False), ("V2", old2, False), ("V2u", old2, True)):
        r = score_systems(df, o, eye, systems, unsure_match=um)
        r["est"] = tag
        R[tag] = r
        print(tag, "done", flush=True)
    A = pd.concat(R.values(), ignore_index=True)
    A.to_csv(OUT / "sweep.tsv", sep="\t", index=False, float_format="%.5g")
    D1 = cell_diag(df, P, set(old), eye)       # V1 exact = earlier labels
    D2 = cell_diag(df, P, set(old2), eye)
    O = other_libraries()

    L = ["JUDGE CUT SWEEP: precision AND recall of a stricter P(yes) cut, per search (public DISBench data; 2026-10-11)",
         "Code: eval/eval_judge_cut_sweep.py (no model runs). Judge = Qwen3-VL-4B real phone 4-bit weights, stored float32",
         "P(yes) on all 7,886 photos of the 4 RESULTS 34 libraries (eval/recall_audit/scores_q3vl4b_real), the planner's",
         "questions: dog/car/bicycle = object form 'Is there a real X anywhere in this photo ...?' (phone cut today 0.7);",
         "beach/sunset/food = subject form 'Is this a photo of X?' (phone cut today 0.99, JudgeCutoff.swift).",
         "",
         "METHOD. RESULTS 34's estimator (eval_recall.score_systems). Its strata were NOT defined by the 0.99 cut: they are",
         "the OLD simulated-weight judge (eval/recall_audit/scores) at P >= 0.7 (kept) / doubt / top-5% vector / rest, with",
         "uniform blind draws inside each. Any keep rule on the REAL judge's P is scored by splitting every stratum into",
         "cells by which cuts keep the photo (nested cuts -> cells = real-P bins within a stratum). A uniform draw inside a",
         "stratum is uniform inside each cell, so each cell's real count = size x its labelled match rate; a cell with no",
         "labelled photo takes its stratum's rate ('imputed' below). All cuts scored jointly = one shared total of real",
         "photos, so recall at every cut has the same denominator. Earlier (non-uniform) eye labels count exactly.",
         "  V1 = RESULTS 34 labels only (old + 1,950 blind) - the estimator exactly as RESULTS 34/45 ran it.",
         f"  V2 = V1 + RESULTS 45 census ({n_new} audit4 census labels). The census covers a FIXED photo set per search (vector",
         "       top 50 + judged top 50 of the pooled library, each library's vector top 20 + judged top 20), all labelled",
         "       -> made an exact stratum; the uniform blind labels then estimate only the photos outside it. Main table = V2.",
         "  'unsure' = not a match (as RESULTS 34/45); V2u = unsure counted as a match (sensitivity).",
         "  Intervals: 95% stratified bootstrap (RESULTS 34; optimistic where a stratum shows 0 matches in its sample),",
         "  recall also 95% Bayesian (Jeffreys per cell; the honest one for misses in the big rest stratum).",
         f"  Census screens per search: " + ", ".join(f"{SH[q]} {nscreen[q]} photos ({unl[q]} unlabelled)" for q in QUERIES),
         "",
         "Cut -> yes-minus-no logit gap g = ln(P/(1-P)) (P = py/(py+pn), py/pn summed over yes/Yes/' yes'/' Yes' and no",
         "variants): " + ", ".join(f"{n} -> {gap_of(c):.2f}" for n, c in CUTS) + " (float32 P is exactly 1.0 for g > ~17.3).",
         ""]
    for q in QUERIES:
        L.append(f"== {SH[q]}  ({sc[sc['query'] == q].question.iloc[0]})")
        g2 = R["V2"].set_index(["system", "query"]); g1 = R["V1"].set_index(["system", "query"])
        gu = R["V2u"].set_index(["system", "query"])
        L.append(f"   real photos in the 7,886 (est.): V2 {g2.loc[('cut 0.7', q)].real_est:.1f}, V1 {g1.loc[('cut 0.7', q)].real_est:.1f}")
        L.append("   | cut (gap) | kept | V2 precision [95% boot] | V2 recall [95% boot] [95% Bayes] | V1 precision | V1 recall [boot] |"
                 " V2u (unsure=match) P / R |")
        for name, c in CUTS:
            a, b, u = g2.loc[(f"cut {name}", q)], g1.loc[(f"cut {name}", q)], gu.loc[(f"cut {name}", q)]
            L.append(f"   | {name:>8s} ({gap_of(c):5.2f}) | {a.returned_in_library:4d} | {a.precision_est:.2f} [{a.precision_boot_lo:.2f}-"
                     f"{a.precision_boot_hi:.2f}] | {a.recall:.3f} [{a.recall_boot_lo:.3f}-{a.recall_boot_hi:.3f}] "
                     f"[{a.recall_bayes_lo:.3f}-{a.recall_bayes_hi:.3f}] | {b.precision_est:.2f} [{b.precision_boot_lo:.2f}-"
                     f"{b.precision_boot_hi:.2f}] | {b.recall:.3f} [{b.recall_boot_lo:.3f}-{b.recall_boot_hi:.3f}] | "
                     f"{u.precision_est:.2f} / {u.recall:.3f} |")
        L.append("   per P bin (V2): photos, est. precision of the bin [95% boot], labels: exact / uniform, photos imputed at stratum rate")
        dq = D2[D2["query"] == q].set_index("bin")
        for k in range(len(CUTS) + 1):
            nm = bin_name(k)
            a = g2.loc[(f"bin {nm}", q)]
            dd = dq.loc[k]
            if a.returned_in_library == 0:
                continue
            L.append(f"     {nm:22s} {a.returned_in_library:5d} photos  precision {a.precision_est:.2f} [{a.precision_boot_lo:.2f}-"
                     f"{a.precision_boot_hi:.2f}]  real {a.real_returned_est:7.1f}  labels {dd.exact}/{dd.uniform}  imputed {dd.imputed}")
        L.append("")
    L.append("OTHER LIBRARIES, other subjects (RESULTS 35 (2) labels: biased toward judge disagreements; P from the OLD simulated")
    L.append("weights, question 'Is this a photo of X?'). right kept / right, wrong kept / wrong, unsure kept / unsure:")
    for q, d in O.groupby("query", sort=False):
        L.append(f"  {q}: " + "; ".join(f"{r.cut}: {r.right_kept}/{r.right_all} r, {r.wrong_kept}/{r.wrong_all} w, "
                                        f"{r.unsure_kept}/{r.unsure_all} u" for r in d.itertuples()))
    L.append("")
    L.append("FLOAT RESOLUTION of the stored P (float32), per search over 7,886 photos:")
    for q in QUERIES:
        p = P[q].to_numpy().astype(np.float64)
        lt = p[p < 1]
        gmax = np.log(lt / (1 - lt)).max()
        L.append(f"  {SH[q]:8s} P == 1.0: {int((p == 1.0).sum()):4d}; P in [0.9999999, 1): {int(((p >= 0.9999999) & (p < 1)).sum()):3d}; "
                 f"P in [0.999999, 0.9999999): {int(((p >= 0.999999) & (p < 0.9999999)).sum()):3d}; distinct P values >= 0.99999: "
                 f"{len(np.unique(p[p >= 0.99999]))}; largest gap below 1.0: {gmax:.2f}")
    txt = "\n".join(L)
    (OUT / "report_tables.txt").write_text(txt)
    print(txt)
    D1.to_csv(OUT / "cells_v1.tsv", sep="\t", index=False)
    D2.to_csv(OUT / "cells_v2.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
