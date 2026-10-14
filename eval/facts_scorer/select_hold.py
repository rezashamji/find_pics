"""RESULTS 59 model selection on held-out TRAIN fold(s) only (oracle match >= .8 / non-match < .2): pooled AUC, macro
AUC over questions, F1 at .7 and the F1-best cut. Usage: python eval/facts_scorer/select_hold.py <model dir> ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

print("model | pairs | AUC | macro AUC | P/R/F1 @.7 | best cut (F1)")
for d in sys.argv[1:]:
    f = Path(d) / "pred_hold.parquet"
    if not f.exists():
        continue
    X = pd.read_parquet(f)
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
