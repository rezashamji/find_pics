"""Subject search ("this dog / thing / place") without library x library smoothing: does neighbour smoothing (DBA k=2)
restricted to a candidate set keep RESULTS 30's quality? (FindPicsCore.subjectScoresCandidates.)

Exact (subjectScores): every library unit = normalized sum of its k+1 nearest library units (itself included); every
reference = normalized (itself + its k nearest library units); score = mean over smoothed refs of the cosine.
Candidates (K): the references are smoothed exactly (one pass); q = cosine of each RAW unit to the mean smoothed ref;
the top K units by q are smoothed with their k+1 nearest units WITHIN those K and scored like exact; every other unit
keeps q. Cost O(n R + K^2) instead of O(n^2).
Protocol = eval/pet_dba.py (3 random reference photos per identity, seed 0, targets = its other photos, references
excluded from the ranking): R-precision, top-5, and how many of the exact method's top 300 (what the phone judges)
the candidate method also has in its top 300.
Sets (PE-Core-L-14-336 vectors): products and landmarks (eval/instance_*), 40 DogFaceNet dogs; libraries as in
RESULTS 30 (+20k everyday / + the 19,218-photo test library) and with ALL 109k DISBench everyday photos added.
GPU (torch). `python eval/dba_candidates.py`;  `python eval/dba_candidates.py fixture` (CPU golden file for Swift).
"""
import glob
import json
import sys

import numpy as np

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics"
KS = (1000, 2000, 5000, 10000)


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


# ---------------- numpy reference (also the golden-fixture generator) ----------------
def topk_np(Q, X, m):
    S = Q @ X.T
    return np.argsort(-S, axis=1, kind="stable")[:, :m]


def smooth_refs_np(X, R, k):
    nn = topk_np(R, X, min(k, len(X)))
    return norm(R + X[nn].sum(1))


def exact_np(X, R, k=2):
    nn = topk_np(X, X, min(k + 1, len(X)))
    Xs = norm(X[nn].sum(1))
    V = smooth_refs_np(X, R, k)
    return (Xs @ V.T).mean(1)


def cand_np(X, R, k=2, K=1000):
    if len(X) <= K:
        return exact_np(X, R, k)
    V = smooth_refs_np(X, R, k)
    q = (X @ V.T).mean(1)
    C = np.argsort(-q, kind="stable")[:K]
    Xc = X[C]
    nn = topk_np(Xc, Xc, min(k + 1, K))
    xs = norm(Xc[nn].sum(1))
    out = q.copy(); out[C] = (xs @ V.T).mean(1)
    return out


def fixture():
    rng = np.random.default_rng(5)
    d = 24; cent = norm(rng.normal(size=(30, d)))
    X = norm(cent[rng.integers(0, 30, 1500)] + 0.6 * rng.normal(size=(1500, d)) / np.sqrt(d)).astype(np.float32)
    X = X.round(6).astype(np.float32)
    R = X[[3, 10, 77]].copy() + 0.01
    R = norm(R).round(6).astype(np.float32)
    out = {"units": X.tolist(), "refs": R.tolist(), "k": 2}
    for K in (200, 600):
        out[f"cand{K}"] = cand_np(X, R, 2, K).astype(np.float64).tolist()
    out["exact"] = exact_np(X, R, 2).astype(np.float64).tolist()
    json.dump(out, open(ROOT + "/ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/subject_candidates.json", "w"))
    print("fixture written")


# ---------------- GPU evaluation ----------------
def main():
    import torch
    dev = "cuda"
    rng = np.random.default_rng(0)
    D = np.concatenate([np.load(f).astype(np.float32) for f in sorted(glob.glob(ROOT + "/data/public/index_disbench/shards/*/clip.npy"))])
    D = norm(D)
    D20 = D[rng.choice(len(D), 20000, replace=False)]

    def T(a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)

    def topk(Q, X, m, block=4096):
        out = []
        for s in range(0, len(Q), block):
            out.append(torch.topk(Q[s:s + block] @ X.T, m, dim=1).indices)
        return torch.cat(out)

    def nrm(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-9)

    def lib_smooth(X, k=2):
        nn = topk(X, X, k + 1)
        return nrm(X[nn].sum(1))

    def ref_smooth(X, R, k=2):
        return nrm(R + X[topk(R, X, k)].sum(1))

    def run(name, X, labels, n_id, queries):
        Xt = T(X); Xs = lib_smooth(Xt)
        res = {"library": len(X), "plain": [], "exact": []}
        for K in KS:
            res[f"K{K}"] = []
        overlap = {K: [] for K in KS}
        for refs, tg in queries:
            R = Xt[refs]; V = ref_smooth(Xt, R); vm = V.mean(0)
            plain = (Xt @ R.mean(0)); ex = Xs @ vm
            q = Xt @ vm
            scores = {"plain": plain, "exact": ex}
            for K in KS:
                C = torch.topk(q, K).indices; Xc = Xt[C]
                nn = topk(Xc, Xc, 3)
                s = q.clone(); s[C] = nrm(Xc[nn].sum(1)) @ vm
                scores[f"K{K}"] = s
            top_ex = None
            for m, s in scores.items():
                s = s.clone(); s[torch.as_tensor(refs, device=dev)] = -float("inf")
                o = torch.argsort(s, descending=True).cpu().numpy()
                res[m].append((np.isin(o[:len(tg)], tg).mean(), np.isin(o[:5], tg).sum()))
                if m == "exact":
                    top_ex = set(o[:300].tolist())
                elif m.startswith("K"):
                    overlap[int(m[1:])].append(len(top_ex & set(o[:300].tolist())))
            del scores
        out = {"library": len(X), "queries": len(queries)}
        for m in ["plain", "exact"] + [f"K{K}" for K in KS]:
            a = np.array(res[m]); out[m] = {"rprec": round(float(a[:, 0].mean()), 4), "top5": f"{int(a[:, 1].sum())}/{5 * len(queries)}"}
        for K in KS:
            out[f"K{K}"]["top300_overlap_with_exact"] = f"{np.mean(overlap[K]):.1f}/300 (min {min(overlap[K])})"
        print(name, json.dumps(out), flush=True)
        return out

    out = {}
    for kind in ("things", "places"):
        L = np.load(f"{ROOT}/eval/instance_{kind}_labels.npy", allow_pickle=True)
        X = norm(np.load(f"{ROOT}/eval/instance_{kind}_PE-Core_full.npy").astype(np.float32))
        ids = [l for l in np.unique(L) if (L == l).sum() >= 4]
        r = np.random.default_rng(0)
        ids = list(r.choice(ids, min(300, len(ids)), replace=False))
        qs = []
        for l in ids:
            mine = np.where(L == l)[0]; refs = r.choice(mine, 3, replace=False); qs.append((refs, np.setdiff1d(mine, refs)))
        out[f"{kind} +20k everyday"] = run(kind, np.concatenate([X, D20]), L, len(X), qs)
        out[f"{kind} +109k everyday"] = run(kind, np.concatenate([X, D]), L, len(X), qs)
    # dogs: same 40 dogs / refs as eval/pet_dba.py (vectors cached by it in .cache/pet_dba_vectors.npz)
    z = np.load(ROOT + "/.cache/pet_dba_vectors.npz", allow_pickle=True)
    V, BG, lab = z["V"], z["BG"], z["lab"]
    r = np.random.default_rng(0)
    import pandas as pd
    dogs = [d for d in pd.unique(lab) if (lab == d).sum() >= 6]
    dogs = list(r.choice(dogs, min(40, len(dogs)), replace=False))
    qs = []
    for d in dogs:
        mine = np.where(lab == d)[0]; refs = r.choice(mine, 3, replace=False); qs.append((refs, np.setdiff1d(mine, refs)))
    out["dogs + test library"] = run("dogs", np.concatenate([V, BG]), lab, len(V), qs)
    out["dogs + test library + 109k everyday"] = run("dogs", np.concatenate([V, BG, D]), lab, len(V), qs)
    json.dump(out, open(ROOT + "/eval/dba_candidates.json", "w"), indent=1)


if __name__ == "__main__":
    fixture() if len(sys.argv) > 1 and sys.argv[1] == "fixture" else main()
