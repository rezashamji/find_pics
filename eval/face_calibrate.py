"""Recalibrate every face threshold for the shipped phone face model, AuraFace-v1 + flip averaging (Reza, 10-07).

Every cut in people.py / FindPicsCore was set for buffalo_l. Same criteria that set them:
  - grouping cut (face_groups, assignedGroups; buffalo 0.55): groups must be as pure as buffalo_l's (RESULTS 24: groups
    pure by eye). Measured: purity = share of a group's faces that belong to its majority identity, on CelebA + DigiFace.
  - expansion cut (expand_refs; buffalo 0.55: the lowest cut that added recall without new wrong matches, JOURNAL
    10-02 04:22) and person accept (buffalo 0.40) and the "closer to someone else" cut (other_identities; buffalo 0.40,
    the same number as person accept by design): tuned jointly at PRODUCT level (people.py exactly, 3 and 8 refs, every
    CelebA image in one library) for the most recall with no more wrong items than buffalo_l at its shipped cuts
    (62 with 3 refs, 180 with 8 refs; eval/face_free_deepdive.md operating point b), ONE setting for both ref counts.
  - pairwise cuts with no product-level criterion (possible band 0.30, pickRefFaces' "nothing matches" 0.30,
    refs_from_items consensus floor 0.20): mapped to the cut with the same number of different-person pairs above it
    as buffalo_l on CelebA (eval_face_free.map_thresholds idea).
Embeddings: data/public/derived/face_free_{celeba,digiface}.npz (eval_face_free.py embed; public data only).
Usage: python eval/face_calibrate.py run      (CPU, 8 processes) -> eval/results_face_calibrate.json
"""
import json
import os
import sys
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics"
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src")
import eval_face_free as F          # noqa: E402
from findpics import people as P    # noqa: E402

NEW = "auraface+flip"
OUT = f"{ROOT}/eval/results_face_calibrate.json"
BUDGET = {3: 62, 8: 180}            # buffalo_l's wrong items at its shipped cuts (CelebA, deep dive)
_D = {}


def data(ds):
    if ds not in _D:
        z = np.load(F.OUT / f"face_free_{ds}.npz")       # same variants as eval_face_free.variants
        _D[ds] = (z["who"], {"buffalo_l": F.l2(z["buffalo_l"]), NEW: F.l2(F.l2(z["auraface"]) + F.l2(z["auraface_flip"]))})
    return _D[ds]


def lib(E):
    import pandas as pd
    n = len(E)
    return SimpleNamespace(face_emb=E.astype(np.float16), n_items=n,
                           faces=pd.DataFrame({"item_row": np.arange(n), "face_px": np.full(n, 100.0), "det_score": np.ones(n)}))


def purity(args):
    ds, model, G = args
    who, V = data(ds); idx = lib(V[model])
    gs = P.face_groups(idx, top=12, accept=G)
    rows = []
    for g in gs:
        ids, c = np.unique(who[np.array(g["faces"])], return_counts=True)
        rows.append(dict(size=len(g["faces"]), majority=int(ids[np.argmax(c)]), purity=float(c.max() / c.sum()),
                         majority_share_of_person=float(c.max() / (who == ids[np.argmax(c)]).sum())))
    maj = [r["majority"] for r in rows]
    return dict(ds=ds, model=model, cut=G, groups=len(rows), mean_purity=round(float(np.mean([r["purity"] for r in rows])), 4) if rows else None,
                min_purity=round(float(min(r["purity"] for r in rows)), 4) if rows else None,
                impure_faces=int(sum(r["size"] * (1 - r["purity"]) for r in rows)), faces=int(sum(r["size"] for r in rows)),
                split_people=len(maj) - len(set(maj)),
                mean_share_of_person=round(float(np.mean([r["majority_share_of_person"] for r in rows])), 4) if rows else None)


def scores(ds, model, G, Ecut, O, n_refs, seed=0):
    """people.py pipeline (as eval_face_free.product) with separate cuts: groups G, expansion Ecut (None = off),
    other-identities O. -> (target item scores, wrong item scores >= floor)."""
    who, V = data(ds); E = V[model]; idx = lib(E); E16 = idx.face_emb.astype(np.float32)
    G_ = P.face_groups(idx, top=12, accept=G)
    groups = [E16[np.array(g["faces"])] for g in G_]
    GM = np.full((len(E), len(groups)), -1.0, np.float32)
    for gi, g in enumerate(groups):
        S = E16 @ g.T; S[S > 0.999] = -1.0; GM[:, gi] = S.max(1)
    rng = np.random.default_rng(seed); T, I = [], []; checked = 0
    for q in np.unique(who):
        mine = np.where(who == q)[0]
        if len(mine) < 8 + 5:
            continue
        ref_rows = rng.choice(mine, n_refs, replace=False)
        refs = idx.face_emb[ref_rows] if Ecut is None else P.expand_refs(idx, idx.face_emb[ref_rows], accept=Ecut, rounds=3)
        R = refs.astype(np.float32).T
        keep = [gi for gi, g in enumerate(groups) if float((g @ R).max(1).mean()) < O]
        s = P.face_sims(idx, refs)
        if keep:
            s = np.where(GM[:, keep].max(1) >= s, -1.0, s)
        if checked < 2:
            others = np.concatenate([groups[k] for k in keep]) if keep else np.zeros((0, E.shape[1]), np.float32)
            assert np.allclose(P.item_person_scores(idx, refs, others=others)[0], s, atol=1e-5)
            checked += 1
        tgt = np.setdiff1d(mine, ref_rows)
        T.append(s[tgt]); so = s[who != q]; I.append(so[so >= 0.2])
    return np.concatenate(T), np.sort(np.concatenate(I))[::-1]


def solve(args):
    """Smallest accept cut with wrong items <= budget, per ref count; one cut for both = the larger one."""
    ds, model, G, Ecut, O = args
    r = dict(ds=ds, model=model, group_cut=G, expand_cut=Ecut, other_cut=O)
    TI = {n: scores(ds, model, G, Ecut, O, n) for n in (3, 8)}
    a = {n: (float(TI[n][1][BUDGET[n]]) + 1e-6 if len(TI[n][1]) > BUDGET[n] else 0.2) for n in (3, 8)}
    A = max(a.values())
    for n in (3, 8):
        T, I = TI[n]
        r[f"accept_alone_{n}"] = round(a[n], 4)
        r[f"recall_{n}"] = round(float((T >= A).mean()), 4); r[f"hits_{n}"] = int((T >= A).sum()); r[f"targets_{n}"] = int(len(T))
        r[f"wrong_{n}"] = int((I >= A).sum())
    r["accept"] = round(A, 4); r["recall_sum"] = r["recall_3"] + r["recall_8"]
    return r


def at(args):
    """Fixed cuts -> recall / wrong items / possible-list size (any dataset)."""
    ds, model, G, Ecut, O, A, M = args
    r = dict(ds=ds, model=model, group_cut=G, expand_cut=Ecut, other_cut=O, accept=A, maybe=M)
    for n in (3, 8):
        T, I = scores(ds, model, G, Ecut, O, n)
        r[f"recall_{n}"] = round(float((T >= A).mean()), 4); r[f"hits_{n}"] = int((T >= A).sum()); r[f"targets_{n}"] = int(len(T))
        r[f"wrong_{n}"] = int((I >= A).sum()); r[f"possible_right_{n}"] = int(((T >= M) & (T < A)).sum())
        r[f"possible_wrong_{n}"] = int(((I >= M) & (I < A)).sum())
    return r


def pair_map(ds, cuts=(0.2, 0.3, 0.4, 0.55)):
    """buffalo_l cut -> NEW cut with the same number of different-person pairs at or above it; plus the NEW model's
    pair counts at a given cut (for the chosen accept)."""
    who, V = data(ds)
    E = V["buffalo_l"]; cnt = {c: 0 for c in cuts}
    for s in range(0, len(E), 2048):
        S = E[s:s + 2048] @ E.T
        S[who[s:s + 2048, None] == who[None, :]] = -2
        S[np.arange(S.shape[1])[None, :] <= np.arange(s, s + S.shape[0])[:, None]] = -2
        for c in cuts:
            cnt[c] += int((S >= c).sum())
    top = F.impostor_top(V[NEW], who, k=max(cnt.values()) + 10)[0]
    return dict(buffalo_pairs=cnt, mapped={c: round(float(top[min(max(n, 1), len(top)) - 1]), 4) for c, n in cnt.items()},
                pairs=int(sum(int((who != w).sum()) for w in who) // 2))


def pair_count(ds, model, c):
    who, V = data(ds); E = V[model]; n = 0
    for s in range(0, len(E), 2048):
        S = E[s:s + 2048] @ E.T
        S[who[s:s + 2048, None] == who[None, :]] = -2
        S[np.arange(S.shape[1])[None, :] <= np.arange(s, s + S.shape[0])[:, None]] = -2
        n += int((S >= c).sum())
    return n


def run():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in ("celeba", "digiface"):
        data(ds)                     # loaded before the pool forks: workers share it
    def save():
        json.dump(res, open(OUT, "w"), indent=1)
    with Pool(8) as pool:
        if "pairs" not in res:
            res["pairs"] = {ds: pair_map(ds) for ds in ("celeba", "digiface")}; save(); print("pairs", res["pairs"], flush=True)
        if "purity" not in res:
            jobs = [(ds, "buffalo_l", 0.55) for ds in ("celeba", "digiface")]
            jobs += [(ds, NEW, g) for ds in ("celeba", "digiface") for g in (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68)]
            res["purity"] = pool.map(purity, jobs); save()
            for r in res["purity"]:
                print("purity", r, flush=True)
        if "sweep" not in res:
            grid = [("celeba", NEW, G, Ec, O) for G in (0.62,) for Ec in (None, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70)
                    for O in (0.46, 0.48, 0.50, 0.52, 0.54)]
            grid += [("celeba", "buffalo_l", 0.55, 0.55, 0.40)]
            res["sweep"] = pool.map(solve, grid); save()
            for r in sorted(res["sweep"], key=lambda r: -r["recall_sum"])[:15]:
                print("sweep", r, flush=True)
    print("FACE_CALIBRATE_OK", flush=True)


def final(G, Ec, O, A, M):
    """The chosen setting on both datasets vs buffalo_l at its shipped cuts."""
    res = json.load(open(OUT))
    Ec = None if Ec in (None, "none") else Ec
    key = f"final_G{G}_E{Ec}_O{O}_A{A}"
    jobs = [(ds, NEW, G, Ec, O, A, M) for ds in ("celeba", "digiface")]
    jobs += [(ds, "buffalo_l", 0.55, 0.55, 0.40, 0.40, 0.30) for ds in ("celeba", "digiface")]
    for ds in ("celeba", "digiface"):
        data(ds)
    with Pool(4) as pool:
        res[key] = pool.map(at, jobs)
    res[key + "_pairs"] = {ds: {"new_at_accept": pair_count(ds, NEW, A), "buffalo_at_0.40": pair_count(ds, "buffalo_l", 0.40)}
                          for ds in ("celeba", "digiface")}
    json.dump(res, open(OUT, "w"), indent=1)
    for r in res[key]:
        print("final", r, flush=True)
    print("final pairs", res[key + "_pairs"], flush=True)
    print("FACE_FINAL_OK", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "final":
        final(*[None if a == "none" else float(a) for a in sys.argv[2:7]])
