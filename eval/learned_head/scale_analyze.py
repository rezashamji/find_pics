"""M48-scale analysis of eval/learned_head/scale/scores_{run}.parquet (CPU, no model runs)
-> eval/learned_head/scale/tables.txt (full per-search tables) + curve.txt (the learning curve summary).
Metrics exactly as eval/learned_head/report.txt (analyze.py functions): teacher-relative AUC / AP / P@R.8 per search
pooled over its libraries, macro mean over searches; A4 eye precision / recall at the judge's own count (RESULTS 34
estimator, unsure = no match); held-out planner (JD) questions: AUC among each question's 24 most similar photos.
  python eval/learned_head/scale_analyze.py [RUN ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval"); sys.path.insert(0, "eval/learned_head")
import analyze as AN  # noqa: E402

S = Path("eval/learned_head/scale")
C = []


def cs(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); C.append(s)


def macro(X, cols, y="y"):
    """Macro means over the searches every column can score (EV16 selfies have no look score: left out of all)."""
    if "look" in X and X.look.notna().any():
        X = X[X.groupby("query").look.transform(lambda v: v.notna().any())]
    r = {}
    for c in cols:
        if c not in X or X[c].isna().all():
            continue
        m = [v[c] for _, d in X.groupby("query") for v in [AN.metrics(d, [c], y)] if c in v]   # skips selfies' missing look
        r[c] = tuple(np.nanmean([v[k] for v in m]) for k in ("auc", "ap", "p80"))
    return r


def topk_judge(X, col):
    k = np.zeros(len(X), bool)
    for q, d in X.groupby("query"):
        k[d.sort_values(col, ascending=False).index[:int(d.y.sum())]] = True
    return k


def eye_pooled(X, systems):
    from eval_recall import score_systems
    E = AN.eye_ctx()
    sysd = {nm: {q: set(X.item[(X["query"] == q).to_numpy() & np.asarray(k)]) for q in X["query"].unique()}
            for nm, k in systems.items()}
    R = score_systems(E["df"], E["old"], E["eye"], sysd)
    out = {}
    for nm in systems:
        g = R[R.system == nm]
        out[nm] = (g.real_returned_est.sum() / max(g.returned_in_library.sum(), 1), g.real_returned_est.sum() / g.real_est.sum(),
                   {q: (r.precision_est, r.recall) for q, r in g.set_index("query").iterrows()})
    return out


def jd_eval(X, test_libs):
    """Held-out planner questions: per question AUC among its 24 'top' photos (macro), all / by kind / by nearness of
    the question to this run's training questions; plus the test-library photos only."""
    J = X[X.src == "JD"]
    if not len(J) or "near_train" not in J:
        return None
    T = J[J.jd_part == "top"]
    res = {}
    for c in ("cos_q", "lr", "mlp"):
        a = T.groupby("query").apply(lambda d: AN.auc(d.y, d[c]), include_groups=False)
        res[c] = a
    nt = T.groupby("query").near_train.first(); kd = T.groupby("query").kind.first()
    D = pd.DataFrame(dict(cos_q=res["cos_q"], lr=res["lr"], mlp=res["mlp"], near=nt, kind=kd)).dropna()
    U = J[J.lib.isin(test_libs)]                 # unseen library: only the photos of the 20 test libraries
    ul = U.groupby("query").apply(lambda d: pd.Series({c: AN.auc(d.y, d[c]) for c in ("cos_q", "mlp")}), include_groups=False).dropna()
    return D, ul


def main():
    runs = sys.argv[1:] or ["E1port", "JDonly", "N2400", "N5000", "N10000", "ALL"]
    runs = [r for r in runs if (S / f"scores_{r}.parquet").exists()]
    A = pd.read_parquet("eval/learned_head/pairs.parquet", columns=["src", "lib"])
    test_libs = set(A[A.src.isin(["EV16", "A4"])].lib)
    nq_train = {}
    sp = json.load(open(S / "split.json")) if (S / "split.json").exists() else None
    curve = []
    lab = sorted((S / "labels").glob("part*.parquet"))
    if lab:
        from score_apple_photos import judge_cutoff
        Q = json.load(open(S / "questions.json"))
        Lb = pd.concat([pd.read_parquet(f) for f in lab], ignore_index=True)
        cut = np.array([judge_cutoff(q["q"]) for q in Q])
        Lb["y"] = Lb.p.to_numpy() >= cut[Lb.qid.to_numpy()]
        per = Lb.groupby("qid").y.sum()
        cs(f"DATA: {len(Lb)} new pairs, {Lb.qid.nunique()} questions, {Lb.item_id.nunique()} photos; teacher-yes at the phone cut "
           f"{int(Lb.y.sum())} ({Lb.y.mean():.3f}); yes rate among the 250 most similar {Lb[Lb.how == 'top'].y.mean():.3f}, among "
           f"the 150 random {Lb[Lb.how == 'rand'].y.mean():.3f}; questions with 0 yes {int((per == 0).sum())}, with >= 200 yes "
           f"{int((per >= 200).sum())}; yes per question p10/p50/p90 {np.percentile(per, [10, 50, 90]).round(0).tolist()}; "
           f"subject-cut (0.999) questions {int((cut > 0.9).sum())} of {len(Q)}")
        kinds = pd.Series([q["jkind"] for q in Q])
        cs("question kinds (build.jd_kind regex): " + str(kinds.value_counts().to_dict()) + "; writer's kinds: " +
           str(pd.Series([q["kind"] for q in Q]).value_counts().to_dict()))
    for run in runs:
        X = pd.read_parquet(S / f"scores_{run}.parquet").reset_index(drop=True)
        AN.L.clear()
        cols = ["look", "cos_q", "lr", "mlp"]
        for src in ("EV16", "A4", "OI"):
            AN.table(X[X.src == src], cols, title=f"[{run}] {src} teacher-relative (unseen query; unseen library except OI for "
                                                     f"runs trained on OI15)", per_lib=src != "OI")
        A4 = X[X.src == "A4"].reset_index(drop=True)
        ey = eye_pooled(A4, {"judge": A4.y.to_numpy(), "look": topk_judge(A4, "look"), "mlp": topk_judge(A4, "mlp")})
        AN.say(f"\n[{run}] A4 EYE at the judge's count (pooled P / R): " + "; ".join(f"{k} {v[0]:.3f} / {v[1]:.3f}" for k, v in ey.items()))
        for q in sorted(ey["judge"][2]):
            AN.say(f"   {q:22s} " + "  ".join(f"{k} {ey[k][2][q][0]:.2f}/{ey[k][2][q][1]:.2f}" for k in ey))
        AN.eye_auc(A4, ["look", "cos_q", "lr", "mlp", "p"], f"[{run}] EYE AUC, A4")
        ev = macro(X[X.src == "EV16"], cols); a4 = macro(X[X.src == "A4"], cols); oi = macro(X[X.src == "OI"], cols)
        row = dict(run=run, ev_look=ev["look"], ev_mlp=ev["mlp"], ev_lr=ev["lr"], a4_look=a4["look"], a4_mlp=a4["mlp"],
                   oi_look=oi.get("look"), oi_mlp=oi.get("mlp"), eye_judge=ey["judge"][:2], eye_look=ey["look"][:2],
                   eye_mlp=ey["mlp"][:2])
        jd = jd_eval(X, test_libs)
        if jd is not None:
            D, ul = jd
            AN.say(f"\n[{run}] HELD-OUT PLANNER QUESTIONS (JD, never in training): AUC among the 24 most similar photos, macro")
            AN.say(f"   {'subset':44s} {'questions':>9s}  cos_q    lr    mlp")
            subsets = [("all", D)] + [(f"kind {k}", g) for k, g in D.groupby("kind")]
            for lo, hi in ((0, .80), (.80, .88), (.88, .93), (.93, 1.01)):
                subsets.append((f"nearest training question cos [{lo:.2f},{hi:.2f})", D[(D.near >= lo) & (D.near < hi)]))
            for nm, g in subsets:
                AN.say(f"   {nm:44s} {len(g):9d}  {g.cos_q.mean():.3f}  {g.lr.mean():.3f}  {g.mlp.mean():.3f}")
            AN.say(f"   test-library photos only (unseen library; all their photos), {len(ul)} questions: cos_q {ul.cos_q.mean():.3f}  mlp {ul.mlp.mean():.3f}")
            nn = D[D.near < .93]
            row.update(jd_all=(D.cos_q.mean(), D.mlp.mean()), jd_nopara=(nn.cos_q.mean(), nn.mlp.mean(), len(nn)),
                       jd_unseenlib=(ul.cos_q.mean(), ul.mlp.mean()))
        HN = X[X.src == "NEW"]
        if len(HN):
            a_all = HN.groupby("query").apply(lambda d: pd.Series({c: AN.auc(d.y, d[c]) for c in ("cos_q", "mlp")}), include_groups=False)
            Tp = HN[HN.how == "top"]
            a_top = Tp.groupby("query").apply(lambda d: pd.Series({c: AN.auc(d.y, d[c]) for c in ("cos_q", "mlp")} |
                                                                   {f"ap_{c}": AN.ap(d.y, d[c]) for c in ("cos_q", "mlp")}), include_groups=False)
            AN.say(f"\n[{run}] HELD-OUT GENERATED QUESTIONS ({HN['query'].nunique()}; whole paraphrase clusters held out; photos "
                   f"from the training libraries): AUC all 400 cos_q {a_all.cos_q.mean():.3f} mlp {a_all.mlp.mean():.3f}; "
                   f"among the 250 most similar: AUC cos_q {a_top.cos_q.mean():.3f} mlp {a_top.mlp.mean():.3f}, "
                   f"AP cos_q {a_top.ap_cos_q.mean():.3f} mlp {a_top.ap_mlp.mean():.3f}")
            row.update(hn=(a_top.cos_q.mean(), a_top.mlp.mean(), a_top.ap_cos_q.mean(), a_top.ap_mlp.mean()))
        curve.append(row)
        (S / f"tables_{run}.txt").write_text("\n".join(AN.L) + "\n")
    cs("LEARNING CURVE (unseen queries; EV16/A4 also unseen libraries). Teacher-relative = macro over searches, AUC / AP / P@R.8.")
    cs("Eye = A4 pooled eye precision / recall when each system keeps as many photos per search as the phone judge.")
    for r in curve:
        f = lambda t: "  -  " if t is None else " / ".join(f"{x:.3f}" for x in t[:3])  # noqa: E731
        cs(f"\n{r['run']}")
        cs(f"   EV16  look {f(r['ev_look'])} | lr {f(r['ev_lr'])} | mlp {f(r['ev_mlp'])}")
        cs(f"   A4    look {f(r['a4_look'])} | mlp {f(r['a4_mlp'])}      OI5 look {f(r['oi_look'])} | mlp {f(r['oi_mlp'])}")
        cs(f"   EYE   judge {f(r['eye_judge'])} | look {f(r['eye_look'])} | mlp {f(r['eye_mlp'])}")
        if "jd_all" in r:
            cs(f"   JD held-out top24 AUC: all cos {r['jd_all'][0]:.3f} mlp {r['jd_all'][1]:.3f}; no paraphrase in training "
               f"({r['jd_nopara'][2]}) cos {r['jd_nopara'][0]:.3f} mlp {r['jd_nopara'][1]:.3f}; test-library photos cos "
               f"{r['jd_unseenlib'][0]:.3f} mlp {r['jd_unseenlib'][1]:.3f}")
        if "hn" in r:
            cs(f"   held-out generated (top250): AUC cos {r['hn'][0]:.3f} mlp {r['hn'][1]:.3f}; AP cos {r['hn'][2]:.3f} mlp {r['hn'][3]:.3f}")
    (S / __import__("os").environ.get("CURVE_OUT", "curve.txt")).write_text("\n".join(C) + "\n")


if __name__ == "__main__":
    main()
