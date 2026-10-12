"""M48-scale: cosine + head BLEND as the ranking (no training). Per search (pooled over its libraries, as the metrics pool): z-score of the phone's look score
plus z-score of logit(head P) (equal weights, fixed in advance, not tuned on the test sets); a search with no look
score (EV16 selfies) keeps the head alone. Writes scale/scores_{RUN}b.parquet with the blend in the `mlp` column (and
the head in `head`), so scale_analyze.py and rank_replay.py score it unchanged.
  python eval/learned_head/blend.py RUN [RUN ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path("eval/learned_head/scale")


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def main():
    for run in sys.argv[1:]:
        X = pd.read_parquet(S / f"scores_{run}.parquet").reset_index(drop=True)
        X["head"] = X.mlp
        lg = np.log(np.clip(X.mlp.to_numpy(float), 1e-6, 1 - 1e-6) / (1 - np.clip(X.mlp.to_numpy(float), 1e-6, 1 - 1e-6)))
        X["_lg"] = lg
        out = np.full(len(X), np.nan)
        for _, g in X.groupby(["src", "query"], sort=False):    # pooled over libraries, as every metric pools
            h = z(g._lg)
            out[g.index] = h + z(g.look) if g.look.notna().all() else h
        X["mlp"] = out.astype(np.float32)
        X.drop(columns=["_lg"]).to_parquet(S / f"scores_{run}b.parquet")
        print("BLEND_DONE", run, flush=True)


if __name__ == "__main__":
    main()
