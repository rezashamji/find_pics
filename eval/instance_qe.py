"""CPU: query expansion for "find this specific thing/place" (RESULTS 19: PE-Core mean of 3 refs = 0.731 products,
0.693 landmarks). Classic instance-retrieval add-ons, no new model:
  AQE k : add the k best matches of the first search to the references (mean), search again
  aQE   : same, matches weighted by similarity^3
  DBA k : every library vector replaced by the mean of itself and its k nearest library neighbours (computed once at
          indexing; the references use the same transform)
Same protocol as instance_combos.py (3 random refs per identity, seed 0; R-precision)."""
import json
import numpy as np
import pandas as pd


def rprec(X, L, ids, qe=0, w=0.0, seed=0):
    r = np.random.default_rng(seed); out = []
    for l in ids:
        mine = np.where(L == l)[0]
        ref = r.choice(mine, min(3, len(mine) - 1), replace=False); tgt = np.setdiff1d(mine, ref)
        q = X[ref].mean(0); s = X @ q; s[ref] = -np.inf
        if qe:
            top = np.argsort(-s)[:qe]
            wt = np.maximum(s[top], 0) ** w if w else np.ones(qe)
            q2 = (X[ref].sum(0) + (wt[:, None] * X[top]).sum(0) / max(wt.mean(), 1e-6)) / (len(ref) + qe)
            s = X @ q2; s[ref] = -np.inf
        o = np.argsort(-s); out.append(np.isin(o[:len(tgt)], tgt).mean())
    return round(float(np.mean(out)), 3)


def dba(X, k):
    S = X @ X.T; nn = np.argsort(-S, axis=1)[:, :k + 1]          # includes itself
    Y = X[nn].mean(1); return Y / np.linalg.norm(Y, axis=1, keepdims=True)


res = {}
for kind in ("things", "places"):
    L = np.load(f"eval/instance_{kind}_labels.npy", allow_pickle=True)
    X = np.load(f"eval/instance_{kind}_PE-Core_full.npy").astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = [l for l in pd.unique(L) if (L == l).sum() >= 2]
    r = {"n_vectors": len(X), "identities": len(ids), "PE mean (now)": rprec(X, L, ids)}
    for k in (2, 3, 5, 10):
        r[f"AQE {k}"] = rprec(X, L, ids, qe=k)
        r[f"aQE {k}"] = rprec(X, L, ids, qe=k, w=3.0)
    for k in (1, 2, 3):
        Y = dba(X, k); r[f"DBA {k}"] = rprec(Y, L, ids); r[f"DBA {k} + aQE 3"] = rprec(Y, L, ids, qe=3, w=3.0)
    res[kind] = r; print(kind, r, flush=True)
json.dump(res, open("eval/instance_qe.json", "w"), indent=1)
