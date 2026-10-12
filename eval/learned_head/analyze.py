"""M48 analysis of eval/learned_head/scores_*.parquet (CPU, no model runs) -> eval/learned_head/tables.txt

Teacher-relative metrics per search (pooled over its libraries): AUC, AP, precision at the teacher's recall 0.8 / 0.9
(rank by the score, go down until 80% / 90% of the teacher's yes are in, precision of that prefix), R-precision.
Eye metrics: A4 = RESULTS 34's stratified estimator (eval_recall.score_systems; census screens exact, RESULTS 46 'V2';
unsure = no match), EV16 = the labelled photos only (biased samples, same photos for every system).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/learned_head")
L = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def auc(y, s):
    """ROC AUC by ranks (ties = average rank), = P(score of a random yes > score of a random no)."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap(y, s):
    """Average precision: mean over the yes photos of the precision of the ranking down to each of them."""
    y = np.asarray(y, bool)
    if not 0 < y.sum() < len(y):
        return np.nan
    o = np.argsort(-np.asarray(s, float), kind="stable"); t = y[o]
    return float((np.cumsum(t)[t] / (np.where(t)[0] + 1)).mean())


def p_at_r(y, s, R):
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-np.asarray(s, float), kind="stable"); tp = np.cumsum(y[o])
    k = int(np.argmax(tp >= R * y.sum()))
    return tp[k] / (k + 1)


def rprec(y, s):
    y = np.asarray(y, bool); n = int(y.sum())
    return y[np.argsort(-np.asarray(s, float), kind="stable")[:n]].mean() if n else np.nan


def metrics(d, cols, y="y"):
    r = {}
    for c in cols:
        if c not in d or d[c].isna().all():
            continue
        r[c] = dict(auc=auc(d[y], d[c]), ap=ap(d[y], d[c]), p80=p_at_r(d[y], d[c], .8), p90=p_at_r(d[y], d[c], .9),
                    rp=rprec(d[y], d[c]))
    return r


def table(S, cols, by="query", y="y", title="", per_lib=False):
    rows = []
    for q, d in S.groupby(by, sort=True):
        m = metrics(d, cols, y)
        r = dict(query=q, kind=d.kind.iloc[0], n=len(d), yes=int(d[y].sum()))
        for c, v in m.items():
            for k, x in v.items():
                r[f"{c}.{k}"] = x
        if per_lib:
            for c in cols:
                if c in d and not d[c].isna().all():
                    r[f"{c}.auc_lib"] = np.nanmean([auc(g[y], g[c]) for _, g in d.groupby("lib")])
        rows.append(r)
    T = pd.DataFrame(rows)
    say(f"\n{title}")
    hdr = f"{'search':28s} {'kind':7s} {'photos':>7s} {'yes':>6s} | " + " | ".join(f"{c:>5s} AUC/AP/P@R.8/P@R.9/Rp" for c in cols if f"{c}.auc" in T)
    say(hdr)
    for _, r in T.iterrows():
        cells = []
        for c in cols:
            if f"{c}.auc" in T:
                cells.append("  ".join("  -  " if pd.isna(r[f"{c}.{k}"]) else f"{r[f'{c}.{k}']:.2f}" for k in ("auc", "ap", "p80", "p90", "rp")))
        say(f"{str(r['query'])[:28]:28s} {r['kind']:7s} {r['n']:7d} {r['yes']:6d} | " + " | ".join(cells))
    for k, g in list(T.groupby("kind")) + [("ALL", T)]:
        cells = []
        for c in cols:
            if f"{c}.auc" in T:
                cells.append("  ".join(f"{g[f'{c}.{m}'].mean():.2f}" for m in ("auc", "ap", "p80", "p90", "rp")))
        say(f"{'mean of ' + str(len(g)) + ' searches':28s} {k:7s} {int(g.n.sum()):7d} {int(g.yes.sum()):6d} | " + " | ".join(cells))
    return T


def seed_spread(S, model, by="query"):
    cs = sorted(c for c in S.columns if c.startswith(model + "_s"))
    if len(cs) < 2:
        return ""
    v = [np.nanmean([auc(d.y, d[c]) for _, d in S.groupby(by)]) for c in cs]
    w = [np.nanmean([p_at_r(d.y, d[c], .8) for _, d in S.groupby(by)]) for c in cs]
    return f"{model}: mean AUC per seed {', '.join(f'{x:.3f}' for x in v)}; mean P@R.8 per seed {', '.join(f'{x:.3f}' for x in w)}"


# ------------------------------------------------------------------------------------------------------------- eye
_EYE = {}


def eye_ctx():
    if not _EYE:
        from eval_recall import load_scores, old_labels, _labels_eye
        from eval_judge_cut_sweep import census_exact, real_scores
        df = load_scores(); df["item_id"] = df.item_id.astype(str)
        old, eye = old_labels(), _labels_eye()
        cex = census_exact(df, real_scores(), old, eye)[0]
        old2 = dict(cex); old2.update(old)
        _EYE.update(df=df, old=old2, eye=eye)
    return _EYE


def eye_a4(S, systems, title):
    """systems: {name: boolean column or array over S rows}."""
    from eval_recall import score_systems
    E = eye_ctx()
    sysd = {nm: {q: set(S.item[(S["query"] == q).to_numpy() & np.asarray(k)]) for q in S["query"].unique()}
            for nm, k in systems.items()}
    R = score_systems(E["df"], E["old"], E["eye"], sysd)
    say(f"\n{title}")
    say(f"{'search':22s} {'system':34s} {'kept':>6s} {'eye precision est [95%]':>26s} {'eye recall est [95%]':>24s} {'labelled kept: match/n':>24s}")
    lab = S.eye.to_numpy()
    for q in sorted(S["query"].unique()):
        for nm, k in systems.items():
            r = R[(R.system == nm) & (R["query"] == q)].iloc[0]
            m = (S["query"] == q).to_numpy() & np.asarray(k) & pd.notna(lab)
            say(f"{q[:22]:22s} {nm[:34]:34s} {r.returned:6d} {r.precision_est:8.2f} [{r.precision_boot_lo:.2f}-{r.precision_boot_hi:.2f}]"
                f"      {r.recall:6.2f} [{r.recall_boot_lo:.2f}-{r.recall_boot_hi:.2f}]   {int((lab[m] == 'match').sum()):5d}/{int(m.sum())}")
    for nm in systems:
        g = R[R.system == nm]
        say(f"{'ALL 6 (pooled)':22s} {nm[:34]:34s} {int(g.returned.sum()):6d} {g.real_returned_est.sum() / max(g.returned_in_library.sum(), 1):8.2f}"
            f"                   {g.real_returned_est.sum() / g.real_est.sum():6.2f}")
    return R


def eye_auc(S, cols, title, judge=None):
    cols = [c for c in cols if c in S]
    d = S[S.eye.isin(["match", "no_match"])]
    say(f"\n{title}: photos with an eye label (unsure dropped): {len(d)} ({int((d.eye == 'match').sum())} match)")
    say(f"{'search':22s} {'labelled':>8s} {'match':>6s} | " + " ".join(f"{c:>7s}" for c in cols) + "   (AUC of each score vs the eye label)")
    for q, g in list(d.groupby("query")) + [("ALL (pooled)", d)]:
        say(f"{q[:22]:22s} {len(g):8d} {int((g.eye == 'match').sum()):6d} | " +
            " ".join(f"{auc(g.eye == 'match', g[c]):7.3f}" for c in cols))
    if judge:
        for nm, k in judge.items():
            m = d.eye == "match"
            say(f"  {nm}: keeps {int((k[d.index] & m).sum())}/{int(m.sum())} labelled matches, {int((k[d.index] & ~m).sum())}/{int((~m).sum())} labelled non-matches")


# ------------------------------------------------------------------------------------------------------------ main
def main():
    A = pd.read_parquet(OUT / "pairs.parquet")
    say("DATA (public only)")
    g = A.groupby("src").agg(pairs=("y", "size"), searches=("query", "nunique"), photos=("item", "nunique"),
                             libraries=("lib", "nunique"), teacher_yes=("y", "sum"), eye_labelled=("eye", lambda s: int(s.notna().sum())))
    say(g.to_string())
    jd = A[A.src == "JD"]
    k = jd.drop_duplicates("query").kind.value_counts()
    say("JD questions by kind: " + ", ".join(f"{a} {b}" for a, b in k.items()) +
        f"; JD teacher-yes by kind: " + ", ".join(f"{a} {int(b)}" for a, b in jd.groupby("kind").y.sum().items()))
    cols = ["look", "cos_q", "lr", "mlp", "att"]
    # E1
    S = pd.read_parquet(OUT / "scores_E1.parquet")
    for src, ttl in (("EV16", "E1 HELD-OUT QUERIES, EV16 (16 libraries pooled per search; teacher = Qwen3-VL-4B 16-bit, cut 0.7)"),
                     ("A4", "E1 HELD-OUT QUERIES, A4 (4 libraries pooled; teacher = Qwen3-VL-4B REAL phone weights, 0.7 / 0.999)"),
                     ("OI", "E1 HELD-OUT QUERIES, Open Images everyday-5 (19,218 items; teacher = Qwen3.5-9B 16-bit, 0.7)")):
        table(S[S.src == src], cols, title=ttl, per_lib=src != "OI")
        for mdl in ("mlp", "att", "lr"):
            say("  seed spread " + seed_spread(S[S.src == src], mdl))
    # E1b
    Sb = pd.read_parquet(OUT / "scores_E1b.parquet") if (OUT / "scores_E1b.parquet").exists() else S.iloc[:0]
    for src in (("EV16", "A4") if len(Sb) else ()):
        table(Sb[Sb.src == src], ["look", "cos_q", "lr", "mlp"],
              title=f"E1b HELD-OUT QUERY AND LIBRARY ({src}; JD photos of the 20 test libraries removed from training)")
    # E2
    S2 = pd.read_parquet(OUT / "scores_E2.parquet") if (OUT / "scores_E2.parquet").exists() else None
    if S2 is None:
        S2 = pd.DataFrame(columns=["kind", "query", "y", "jd_part", "cos_q", "lr", "mlp", "att"])
    say("\nE2 HELD-OUT JD QUESTIONS (5 folds by question cluster; teacher = Qwen3.5-9B 16-bit, cut 0.7). Macro mean over "
        "questions with both classes. 'all' = the question's 48 photos (24 best PE-Core-L matches + 24 random); "
        "'top24' = only the 24 best matches (random photos are trivially 'no': top24 is the real test).")
    say(f"{'kind':8s} {'questions':>9s} {'yes/pairs':>12s} | " + " | ".join(f"{c:>5s} AUC all/top24  Rp top24" for c in ("cos_q", "lr", "mlp", "att")))
    for kd, g in list(S2.groupby("kind")) + [("ALL", S2)]:
        cells = []
        for c in ("cos_q", "lr", "mlp", "att"):
            a1 = [auc(d.y, d[c]) for _, d in g.groupby("query")]
            t = g[g.jd_part == "top"]
            a2 = [auc(d.y, d[c]) for _, d in t.groupby("query")]
            r2 = [rprec(d.y, d[c]) for _, d in t.groupby("query") if 0 < d.y.sum() < len(d)]
            cells.append(f"{np.nanmean(a1):.3f} / {np.nanmean(a2):.3f}  {np.nanmean(r2):.3f}")
        say(f"{kd:8s} {g['query'].nunique():9d} {int(g.y.sum()):5d}/{len(g):6d} | " + " | ".join(cells))
    # E3
    S3 = pd.read_parquet(OUT / "scores_E3.parquet") if (OUT / "scores_E3.parquet").exists() else None
    if S3 is not None:
        table(S3, cols, title="E3 OPEN IMAGES LEAVE-CONCEPTS-OUT (full 19,218-item library per concept; teacher = "
                              "Qwen3.5-9B 16-bit 0.7; JD questions naming a held-out concept removed from its fold's training)")
    # EYE, A4 (E1 head)
    A4 = S[S.src == "A4"].reset_index(drop=True)
    A4["eye"] = A[A.src == "A4"].set_index(["query", "item"]).eye.reindex(list(zip(A4["query"], A4.item))).to_numpy()
    sysA = {"judge (phone 4B real, 0.7/0.999)": A4.y.to_numpy(),
            "look >= 0.2036 (likelyAtOnce)": (A4.look >= 0.2036).to_numpy(),
            "E1 mlp P >= 0.5": (A4.mlp >= 0.5).to_numpy()}
    kk = {}
    for c in ("look", "mlp"):
        k = np.zeros(len(A4), bool)
        for q, d in A4.groupby("query"):
            n = int(d.y.sum()); k[d.sort_values(c, ascending=False).index[:n]] = True
        kk[c] = k
    sysA["look, top-k = judge's count"] = kk["look"]; sysA["E1 mlp, top-k = judge's count"] = kk["mlp"]
    eye_a4(A4, sysA, "EYE, A4, HELD-OUT-QUERY HEAD (E1 mlp, 3-seed mean) vs the phone judge at its cuts. RESULTS 34 estimator "
                     "(cells impute stratum rates where a system's photos have no label: read the labelled column too)")
    eye_auc(A4, ["look", "cos_q", "lr", "mlp", "att"] + ["p"], "EYE AUC, A4 (E1 heads; p = phone judge P)")
    # EYE, EV16
    EV = S[S.src == "EV16"].reset_index(drop=True)
    eye_auc(EV, ["look", "cos_q", "lr", "mlp", "att"], "EYE AUC, EV16 (E1 heads)",
            judge={"Qwen3-VL-4B 16-bit (teacher)": EV.y, "Qwen3.5-9B 16-bit": EV.y9.astype(bool),
                   "E1 mlp P >= 0.5": EV.mlp >= 0.5,
                   "E1 mlp top-k = teacher count (per search x library)": pd.Series(
                       np.concatenate([np.isin(np.arange(len(g)), np.argsort(-g.mlp.to_numpy())[:int(g.y.sum())]) for _, g in EV.groupby(["query", "lib"], sort=False)]),
                       index=np.concatenate([g.index for _, g in EV.groupby(["query", "lib"], sort=False)])).sort_index()})
    # E4
    say("\nE4 TEACHER-ERROR TRAP: queries SEEN in training (EV16, 16 libraries), tested on A4's 4 unseen libraries.")
    E4 = {}
    for tag in ("E4a_raw4b", "E4b_agree", "E4c_raw9b"):
        f = OUT / f"scores_{tag}.parquet"
        if not f.exists():
            continue
        X = pd.read_parquet(f).reset_index(drop=True)
        X["eye"] = A[A.src == "A4"].set_index(["query", "item"]).eye.reindex(list(zip(X["query"], X.item))).to_numpy()
        E4[tag] = X
        table(X, ["look", "cos_q", "mlp", "att"], title=f"{tag}: teacher-relative on A4 (teacher = phone real weights)")
        eye_auc(X, ["look", "mlp", "att", "p"], f"{tag}: EYE AUC, A4")
    if E4:
        base = next(iter(E4.values()))
        sysE = {"judge (phone 4B real, 0.7/0.999)": base.y.to_numpy()}
        for tag, X in E4.items():
            k = np.zeros(len(X), bool)
            for q, d in X.groupby("query"):
                k[d.sort_values("mlp", ascending=False).index[:int(d.y.sum())]] = True
            sysE[f"{tag} mlp P>=0.5"] = (X.mlp >= 0.5).to_numpy()
            sysE[f"{tag} mlp top-k=judge"] = k
        eye_a4(base, sysE, "E4 EYE, A4: raw teacher vs agreement-filtered teacher vs 9B teacher (mlp)")
    (OUT / "tables.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
