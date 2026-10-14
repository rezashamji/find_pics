"""RESULTS 59: eye-labelled everyday searches (eye_sets.py; TRAIN libraries), the 9B judge vs scorers, all at cut .7
(unsure = no match, as RESULTS 34). A scorer's P for a photo comes from the fold model that did NOT train on that
photo's library (<name>_h0..h3; fold k = sorted TRAIN users [k::4]).
  E1 (RESULTS 45 sample, not uniform): precision / recall / F1 among the labelled photos.
  E2 (RESULTS 34 stratified sample): weighted (stratum size / labelled in stratum) estimates for the 4-library pool.
Usage: python eval/facts_scorer/eval_eye.py <scorer name> ... [--out=file]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M = Path("eval/facts_scorer/models")


def folds():
    tr = sorted(json.loads(Path("eval/complex_bench/splits.json").read_text())["train"])
    return {u: k for k in range(4) for u in tr[k::4]}


def prf(y, yes, w):
    tp = (w * (yes & y)).sum(); P = tp / max(1e-9, (w * yes).sum()); R = tp / max(1e-9, (w * y).sum())
    return P, R, 2 * P * R / max(1e-9, P + R)


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    T = pd.read_parquet("eval/facts_scorer/data/eye_sets.parquet")
    J = pd.read_parquet("eval/facts_scorer/data/eye_9b.parquet").rename(columns={"p": "p9b"})
    J["item_id"] = J.item_id.astype(str)
    T = T.merge(J, on=["item_id", "question"], how="left")
    fo = folds()
    T["fold"] = T.user.map(fo)
    for n in names:
        col = np.full(len(T), np.nan)
        for k in range(4):
            f = M / f"{n}_h{k}" / "pred_eye.parquet"
            if not f.exists():
                continue
            X = pd.read_parquet(f).drop_duplicates(["item_id", "q"])
            d = dict(zip(zip(X.item_id.astype(str), X.q), X.pred))
            m = (T.fold == k).to_numpy()
            col[m] = [d.get((i, q), np.nan) for i, q in zip(T.item_id[m], T.question[m])]
        T[n] = col
    meths = ["p9b", "p_phone"] + names
    lines = [f"Eye-labelled everyday searches, cut .7, unsure = no match. Scorers: fold models (library never trained on)."]
    for s in ("E1", "E2"):
        X = T[T.set == s]
        y = (X.eye == "match").to_numpy(); w = X.weight.to_numpy() if s == "E2" else np.ones(len(X))
        lines.append(f"-- {s}: {len(X)} labelled photos ({int(y.sum())} match), "
                     + ("weighted population estimate" if s == "E2" else "labelled-sample counts"))
        lines.append("method | P | R | F1 | coverage")
        for m in meths:
            v = X[m].to_numpy(); ok = np.isfinite(v)
            if not ok.any():
                continue
            P, R, F = prf(y[ok], v[ok] >= .7, w[ok])
            lines.append(f"{'9B judge' if m == 'p9b' else 'phone judge (stored, Qwen3-VL-4B 4-bit)' if m == 'p_phone' else m}"
                         f" | {P:.3f} | {R:.3f} | {F:.3f} | {ok.sum()}/{len(X)}")
        lines.append("per search (F1): " + "; ".join(
            f"{q}: " + " / ".join(f"{prf(y[(X['query'] == q).to_numpy() & np.isfinite(X[m].to_numpy())], X[m].to_numpy()[(X['query'] == q).to_numpy() & np.isfinite(X[m].to_numpy())] >= .7, w[(X['query'] == q).to_numpy() & np.isfinite(X[m].to_numpy())])[2]:.2f}"
                              for m in ["p9b"] + names)
            for q in sorted(X["query"].unique())) + f"   (order: 9B / {' / '.join(names)})")
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")
        T.to_parquet(Path(out_f).with_suffix(".parquet"))


if __name__ == "__main__":
    main()
