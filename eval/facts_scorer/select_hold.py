"""RESULTS 59 model selection on held-out TRAIN fold(s) only (oracle match >= .8 / non-match < .2): pooled AUC, macro
AUC over questions, F1 at .7 and the F1-best cut. Usage: python eval/facts_scorer/select_hold.py <model dir> ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

print("model | pairs | AUC | macro AUC | P/R/F1 @.7 | best cut (F1)")
PLANQ = set()
if Path("eval/facts_scorer/data/pairs_trainplan.parquet").exists():
    _E = pd.read_parquet("eval/facts_scorer/data/pairs_trainplan.parquet", columns=["user", "q"]).drop_duplicates()
    PLANQ = set(zip(_E.user, _E.q))
part = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--part=")), "truth")   # truth | plan | all
for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
    f = Path(d) / "pred_hold.parquet"
    if not f.exists():
        continue
    X = pd.read_parquet(f)
    isplan = pd.Series([(u, q) in PLANQ for u, q in zip(X.user, X.q)], index=X.index)
    X = X[~isplan] if part == "truth" else X[isplan] if part == "plan" else X
    if not len(X):
        continue
    X = X[(X.p >= .8) | (X.p < .2)]
    y = (X.p >= .8).to_numpy().astype(int); s = X.pred.to_numpy()
    a = []
    for _, g in X.assign(y=y).groupby(["user", "q"]):
        if 0 < g.y.sum() < len(g):
            a.append(roc_auc_score(g.y, g.pred))
    yes = s >= .7; tp = (yes & (y == 1)).sum(); P = tp / max(1, yes.sum()); R = tp / y.sum()
    pr, rc, th = precision_recall_curve(y, s); f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9); k = np.argmax(f1[:-1])
    print(f"{Path(d).name} | {len(X)} | {roc_auc_score(y, s):.3f} | {np.mean(a):.3f} ({len(a)} q) | "
          f"{P:.3f}/{R:.3f}/{2 * P * R / max(P + R, 1e-9):.3f} | {th[k]:.3f} ({f1[k]:.3f})")
