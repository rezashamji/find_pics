"""Reza's demo, the ranking step: within ONE person, order photos from heaviest to leanest.
Single-photo judge ("Does this person look heavier...?") ranks heavy-era above lean-era photos with AUC 0.81-0.87
(eval/results_transformation_judge.json). Hypothesis: comparing two photos of the same person side by side is easier
than an absolute judgement (no per-person calibration, same face for reference).
Data: data/public/transformation/rows.csv + <key>.jpg (IMDB-WIKI face crops; Chris Pratt, Jonah Hill, Seth Rogen, eras
from published reports). For each photo: K random partners of the same person, both orders (cancels left/right bias):
  P(left heavier) from "Two photos of the same person. Does the person look heavier in the LEFT panel than in the RIGHT?"
Score(photo) = mean over its comparisons of P(it is the heavier one). Within-person AUC heavy-era vs lean-era.
Also: the single-photo difference score P(heavier) - P(fit) if eval/../scored_fit.csv exists.
Usage (vLLM env, GPU): python eval/eval_transformation_pairwise.py [K]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OUT = Path("data/public/transformation")
Q = ("Two photos of the same person. Does the person look heavier (fuller face, more body weight) in the LEFT panel "
     "than in the RIGHT panel?")


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(s); n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    from findpics.engine import side_by_side
    from findpics.vlm import VLLMJudge
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rows = pd.read_csv(OUT / "rows.csv", index_col=0)
    ims = {k: Image.open(OUT / f"{k}.jpg").convert("RGB") for k in rows.index}
    J = VLLMJudge(gpu_mem=0.8)
    rng = np.random.default_rng(0)
    pairs = []
    for name, g in rows.groupby("name"):
        keys = list(g.index)
        for k in keys:
            for o in rng.choice([x for x in keys if x != k], min(K, len(keys) - 1), replace=False):
                pairs += [(name, k, o), (name, o, k)]
    p = J.p_yes([side_by_side(ims[a], ims[b], H=448) for _, a, b in pairs], Q)
    df = pd.DataFrame(pairs, columns=["name", "left", "right"]); df["p_left_heavier"] = p
    df.to_csv(OUT / "pairwise.csv", index=False)
    win = pd.concat([df.assign(k=df.left, w=df.p_left_heavier), df.assign(k=df.right, w=1 - df.p_left_heavier)])
    score = win.groupby("k").w.mean()
    rows["pairwise"] = score.reindex(rows.index)
    left_bias = float((df.p_left_heavier >= 0.5).mean())
    res = dict(pairs=len(df), K=K, says_left_heavier_rate=round(left_bias, 3))
    single = json.load(open("eval/results_transformation_judge.json")) if Path("eval/results_transformation_judge.json").exists() else {}
    fit = pd.read_csv(OUT / "scored_fit.csv", index_col=0) if (OUT / "scored_fit.csv").exists() else None
    for name, g in rows.groupby("name"):
        y = (g.era == "heavy").to_numpy()
        r = dict(heavy=int(y.sum()), lean=int((~y).sum()), auc_pairwise=round(auc(y, g.pairwise), 3),
                 auc_single_heavier=(single.get(name, {}).get("heavier", {}) or {}).get("auc_heavy_vs_lean"))
        if fit is not None:
            f = fit.reindex(g.index)
            r["auc_single_heavier_minus_fit"] = round(auc(y, f.p_heavier - f.p_fit), 3)
        # what the product would put in "heavier": top half by score within the person
        top = g.pairwise >= g.pairwise.median()
        r["top_half_heavy_era"] = f"{int((top & (g.era == 'heavy')).sum())}/{int(top.sum())}"
        res[name] = r
    print(json.dumps(res, indent=1))
    json.dump(res, open("eval/results_transformation_pairwise.json", "w"), indent=1)
    rows.to_csv(OUT / "scored_pairwise.csv")


if __name__ == "__main__":
    main()
