"""Bounded "Who is X?" grouping (FindPicsCore.faceGroups with `cap`): does grouping a capped face sample (the newest
half + a deterministic hash sample of the rest), then assigning every other face to the first group whose seed face
it matches (cosine >= accept, O(F x groups)), give the same top groups as greedy grouping of ALL faces (O(F^2))?

Library (public, buffalo_l vectors, group cut 0.55): the test library's faces (data/public/index_testlib: 6 recurring
people with 49-245 tagged photos + IMDB scene crowds + Pexels clips) + CelebA and DigiFace faces with a Zipf
frequency skew (identity of rank r keeps ~72 r^-0.8 of its faces). Faces without a photo date get spread dates.
Agreement: each of the FULL top-10 groups is matched to the sampled top-10 group with the largest Jaccard overlap of
face sets; reported per cap.
`python eval/face_groups_sampled.py` (CPU), `python eval/face_groups_sampled.py fixture` (golden file for the Swift test).
"""
import json
import sys

import numpy as np
import pandas as pd

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics"
M64 = (1 << 64) - 1


def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def sample_rows(rows, taken, cap, seed=0, newest_frac=0.5):
    """FindPicsCore.faceSampleRows: newest cap/2 faces (taken desc, undated last, then index), the rest by hash rank."""
    rows = list(rows)
    if len(rows) <= cap:
        return rows
    newest = sorted(rows, key=lambda r: (taken[r] is None, -(taken[r] or 0), r))[:int(cap * newest_frac)]
    chosen = set(newest)
    rest = sorted((r for r in rows if r not in chosen), key=lambda r: (splitmix((r + seed) & M64), r))
    chosen.update(rest[:cap - len(newest)])
    return sorted(chosen)


def greedy(E, accept, top):
    """faceGroups' greedy core on normalized rows E: returns [(seed, members)] in creation order."""
    n = len(E)
    nb = []
    for s in range(0, n, 2048):
        S = E[s:s + 2048] @ E.T
        nb += [np.where(r >= accept)[0] for r in S]
    alive = np.ones(n, bool); out = []
    while len(out) < top and alive.any():
        d = np.array([np.count_nonzero(alive[x]) if alive[i] else -1 for i, x in enumerate(nb)])   # recounted (as Swift)
        c = int(np.argmax(d))
        if d[c] < 2:
            break
        mem = nb[c][alive[nb[c]]]
        alive[mem] = False; alive[c] = False
        out.append((c, mem))
    return out


def face_groups(emb, item, px, det, taken, top=12, accept=0.55, min_px=40, min_det=0.7, cap=None, seed=0,
                newest_frac=0.5, over=3):
    rows = [i for i in range(len(emb)) if px[i] >= min_px and det[i] >= min_det]
    sampled = bool(cap) and len(rows) > cap
    samp = sample_rows(rows, taken, cap, seed, newest_frac) if sampled else rows
    E = emb[samp].astype(np.float32)
    E = E / (np.sqrt((E * E).sum(1, keepdims=True)) + 1e-8)
    gs = greedy(E, accept, top * over if sampled else top)   # a sample: more greedy groups, cut to `top` after assignment
    groups = []
    for c, mem in gs:
        mean = E[mem].mean(0); rep = samp[mem[int(np.argmax(E[mem] @ mean))]]
        groups.append({"seed": samp[c], "faces": [samp[m] for m in mem], "rep": rep})
    if sampled and groups:
        insamp = set(samp); rest = np.array([r for r in rows if r not in insamp])
        R = emb[rest].astype(np.float32); R = R / (np.sqrt((R * R).sum(1, keepdims=True)) + 1e-8)
        Sd = np.stack([E[samp.index(g["seed"])] for g in groups])
        S = R @ Sd.T
        hit = S >= accept
        first = np.where(hit.any(1), hit.argmax(1), -1)
        for r, g in zip(rest, first):
            if g >= 0:
                groups[g]["faces"].append(int(r))
    for g in groups:
        g["faces"] = sorted(g["faces"]); g["items"] = sorted(set(item[f] for f in g["faces"]))
    order = sorted(range(len(groups)), key=lambda k: (-len(groups[k]["items"]), k))
    return [groups[k] for k in order][:top]


def library(crowd_all=False):
    import glob
    sh = sorted(glob.glob(ROOT + "/data/public/index_testlib/shards/*"))
    E = np.concatenate([np.load(s + "/face_emb.npy").astype(np.float32) for s in sh])
    F = pd.concat([pd.read_parquet(s + "/faces.parquet") for s in sh], ignore_index=True)
    it = pd.read_parquet(ROOT + "/data/public/index_testlib/items.parquet")
    tk = dict(zip(it.item_id, pd.to_datetime(it.taken, utc=True, errors="coerce")))
    item = list(F.item_id); px = F.face_px.to_numpy(); det = F.det_score.to_numpy()
    taken = [None if pd.isna(tk.get(i)) else tk[i].timestamp() for i in item]
    who = ["testlib"] * len(item)
    rng = np.random.default_rng(0)
    dated = np.array([t for t in taken if t is not None])          # the crowd shares the library's date range
    for name in ("celeba", "digiface"):
        z = np.load(f"{ROOT}/data/public/derived/face_free_{name}.npz")
        V = z["buffalo_l"].astype(np.float32); w = z["who"]
        ids = rng.permutation(np.unique(w))
        for r, i in enumerate(ids, 1):
            mine = np.where(w == i)[0]
            keep = mine if crowd_all else mine[:max(2, int(round(72 * r ** -0.8)))]
            sel = keep
            E = np.concatenate([E, V[sel]]); item += [f"{name}_{j}" for j in sel]
            px = np.concatenate([px, np.full(len(sel), 100.0)]); det = np.concatenate([det, np.full(len(sel), 0.95)])
            taken += list(rng.choice(dated, len(sel))); who += [f"{name}:{i}"] * len(sel)
    return E, item, px, det, taken, who


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main():
    import time
    out = {}
    for name, crowd, caps in (("test library + Zipf-skewed CelebA/DigiFace", False, (10000, 5000, 3000, 1500)),
                              ("test library + ALL CelebA/DigiFace faces", True, (20000, 10000, 5000))):
        E, item, px, det, taken, who = library(crowd)
        t = time.time(); full = face_groups(E, item, px, det, taken)[:10]; tf = time.time() - t
        elig = sum(1 for i in range(len(E)) if px[i] >= 40 and det[i] >= 0.7)
        o = {"faces": len(E), "eligible": int(elig), "full_seconds": round(tf, 1),
             "full_top10_sizes": [len(g["items"]) for g in full]}
        for cap in caps:
            for nf, over in ((0.5, 1), (0.5, 3), (0.0, 3)):
                t = time.time(); smp = face_groups(E, item, px, det, taken, cap=cap, newest_frac=nf, over=over)[:10]; ts = time.time() - t
                best = [max((jaccard(g["faces"], h["faces"]), k) for k, h in enumerate(smp)) for g in full]
                key = f"cap{cap} newest{nf} over{over}"
                o[key] = {"sample_fraction": round(cap / elig, 3), "seconds": round(ts, 1),
                          "sizes": [len(g["items"]) for g in smp],
                          "jaccard_per_full_group": [round(b[0], 3) for b in best],
                          "matched_jaccard_ge_0.5": int(sum(b[0] >= 0.5 for b in best)),
                          "same_rank": int(sum(b[1] == k for k, b in enumerate(best))),
                          "distinct_matches": len(set(b[1] for b in best if b[0] >= 0.5))}
                print(name, key, o[key], flush=True)
        out[name] = o
    json.dump(out, open(ROOT + "/eval/face_groups_sampled.json", "w"), indent=1)


def fixture():
    """Small clustered set where the cap bites: golden output for FindPicsCore's faceGroups(cap:)."""
    rng = np.random.default_rng(3)
    d, cent = 16, rng.normal(size=(9, 16))
    sizes = [220, 150, 90, 60, 40, 25, 15, 8, 4]
    emb, item, px, det, taken = [], [], [], [], []
    k = 0
    for c, n in enumerate(sizes):
        for _ in range(n):
            v = cent[c] + rng.normal(scale=0.35, size=d); emb.append(v / np.linalg.norm(v))
            item.append(f"p{k // 2}"); px.append(float(rng.choice([30, 60, 120]))); det.append(float(rng.choice([0.6, 0.9])))
            taken.append(None if k % 11 == 0 else float(1.6e9 + rng.integers(0, 10 ** 7))); k += 1
    for _ in range(200):
        v = rng.normal(size=d); emb.append(v / np.linalg.norm(v)); item.append(f"x{k}"); px.append(80.0); det.append(0.9)
        taken.append(float(1.6e9 + rng.integers(0, 10 ** 7))); k += 1
    emb = np.array(emb).round(6).astype(np.float32)
    perm = rng.permutation(len(emb))
    emb = emb[perm]; item = [item[i] for i in perm]; px = [px[i] for i in perm]; det = [det[i] for i in perm]
    taken = [taken[i] for i in perm]
    res = {}
    for key, cap, nf in (("full", None, 0.0), ("cap300_newest0.5", 300, 0.5), ("cap300_newest0", 300, 0.0)):
        g = face_groups(emb, item, px, det, taken, accept=0.6, cap=cap, newest_frac=nf)
        res[key] = [{"faces": x["faces"], "rep": x["rep"]} for x in g]
    sm = sample_rows([i for i in range(len(emb)) if px[i] >= 40 and det[i] >= 0.7], taken, 300)
    json.dump({"emb": [[float(x) for x in r] for r in emb], "item": item, "px": px, "det": det, "taken": taken, "accept": 0.6,
               "sample300": sm, "groups": res},
              open(ROOT + "/ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/face_groups_cap.json", "w"))
    print({k: [len(x["faces"]) for x in v] for k, v in res.items()})


if __name__ == "__main__":
    fixture() if len(sys.argv) > 1 and sys.argv[1] == "fixture" else main()
