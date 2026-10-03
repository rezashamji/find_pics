"""CPU: can combining the saved PE-Core and DINOv2 vectors, or aggregating the 3 references differently, beat the best
single model on 'find this specific thing/place'? Same protocol as eval_instance.py (refs = 3 random photos per
identity, seed 0; R-precision = fraction of the top-T that are the identity's other photos, T = #other photos)."""
import json
import numpy as np
import pandas as pd

out = {}
for kind in ("things", "places", "dogs", "copies"):
    try:
        L = np.load(f"eval/instance_{kind}_labels.npy", allow_pickle=True)
        A = np.load(f"eval/instance_{kind}_PE-Core_full.npy").astype(np.float32)
        B = np.load(f"eval/instance_{kind}_DINOv2_full.npy").astype(np.float32)
    except FileNotFoundError:
        continue
    A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
    ids = [l for l in pd.unique(L) if (L == l).sum() >= 2]
    variants = {"PE max": (A, None, "max"), "DINO max": (B, None, "max"), "PE+DINO max": (A, B, "max"),
                "PE mean": (A, None, "mean"), "PE+DINO mean": (A, B, "mean")}
    res = {}
    for name, (X, Y, agg) in variants.items():
        r = np.random.default_rng(0); rp = []
        for l in ids:
            mine = np.where(L == l)[0]
            ref = mine[:1] if kind == "copies" else r.choice(mine, min(3, len(mine) - 1), replace=False)
            tgt = np.setdiff1d(mine, ref)
            S = X @ X[ref].T + (Y @ Y[ref].T if Y is not None else 0)
            s = S.max(1) if agg == "max" else S.mean(1); s[ref] = -np.inf
            o = np.argsort(-s); rp.append(np.isin(o[:len(tgt)], tgt).mean())
        res[name] = round(float(np.mean(rp)), 3)
    out[kind] = res; print(kind, len(ids), "identities:", res, flush=True)
json.dump(out, open("eval/results_instance_combos.json", "w"), indent=1)
