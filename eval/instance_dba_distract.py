"""CPU: does neighbour smoothing (DBA) still help "find this specific thing/place" when the library also holds
everyday photos? instance_qe.py found DBA k=1-2 +1.7 (products) / +7.3 (landmarks) points on libraries made only of
the identities. Here 20,000 DISBench everyday photo vectors (same PE-Core-L14-336 index vectors) are added as
distractors: they are never targets, they take part in the smoothing, and they compete in the ranking."""
import glob
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
D = np.concatenate([np.load(f).astype(np.float32) for f in sorted(glob.glob("data/public/index_disbench/shards/*/clip.npy"))])
D = D[rng.choice(len(D), 20000, replace=False)]; D /= np.linalg.norm(D, axis=1, keepdims=True)


def dba(X, k):
    out = np.empty_like(X)
    for s in range(0, len(X), 2000):                       # chunked k-NN (each row includes itself)
        S = X[s:s + 2000] @ X.T; nn = np.argpartition(-S, k, axis=1)[:, :k + 1]
        out[s:s + 2000] = X[nn].mean(1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def rprec(X, L, ids, n_id):
    r = np.random.default_rng(0); res = []
    for l in ids:
        mine = np.where(L[:n_id] == l)[0]
        ref = r.choice(mine, min(3, len(mine) - 1), replace=False); tgt = np.setdiff1d(mine, ref)
        s = X @ X[ref].mean(0); s[ref] = -np.inf
        o = np.argsort(-s); res.append(np.isin(o[:len(tgt)], tgt).mean())
    return round(float(np.mean(res)), 3)


out = {}
for kind in ("things", "places"):
    L = np.load(f"eval/instance_{kind}_labels.npy", allow_pickle=True)
    X = np.load(f"eval/instance_{kind}_PE-Core_full.npy").astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = [l for l in pd.unique(L) if (L == l).sum() >= 2]
    A = np.concatenate([X, D]); n = len(X)
    r = {"identity vectors": n, "distractors": len(D), "no distractors, plain": rprec(X, L, ids, n),
         "+20k everyday, plain": rprec(A, L, ids, n)}
    for k in (1, 2, 3):
        r[f"+20k everyday, DBA {k}"] = rprec(dba(A, k), L, ids, n)
    out[kind] = r; print(kind, r, flush=True)
json.dump(out, open("eval/instance_dba_distract.json", "w"), indent=1)
