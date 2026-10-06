"""Free face recognizers we may ship: accuracy boosts and product-level effect (deep dive, 2026-10-06).

Stage `embed` (GPU): the SAME detector + alignment as eval_face_commercial.py (buffalo_l SCRFD + insightface norm_crop,
112 px ArcFace template, which is also the template OpenCV's FaceRecognizerSF.alignCrop uses), then every recognizer
embeds the aligned crop AND its horizontal mirror. Channel-order checks: AuraFace fed BGR, SFace fed RGB.
Embeddings (public datasets only) -> data/public/derived/face_free_{digiface,celeba}.npz (git-ignored).

Stage `analyze` (CPU):
  (1) single-image protocol of eval_face_commercial.py (8 references, max over references, recall at buffalo_l's
      wrong-match rate at 0.40 and 10x lower) for every variant: plain, flip-TTA (normalize(e + e_mirror)),
      z-normalized score fusion of two models (weighted concatenation; weights 1/std of random-pair scores,
      no labels used).
  (2) product protocol: src/findpics/people.py exactly (references -> expand_refs(0.55, 3 rounds) -> face_groups
      "other identities" rule -> item_person_scores -> accept at 0.40), 3 or 8 references per identity, every image
      of every identity in one library. For non-buffalo variants the 0.30/0.40/0.55 cuts are mapped to the threshold
      with the same pairwise wrong-match rate on that dataset as buffalo_l at those cuts.
Usage: python eval/eval_face_free.py embed     (fp env, GPU)
       python eval/eval_face_free.py analyze   (fp env, CPU ok)
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics"
OUT = Path(f"{ROOT}/data/public/derived"); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src")


# ----------------------------------------------------------------------------------------------------------- embed
def embed_all():
    import cv2
    import torch
    import eval_face_commercial as C
    R = C.Recognizers()
    hf = R.torch["hyperface_10k_ldm"]

    @torch.no_grad()
    def emb(crops):
        o = {}
        for flip in (False, True):
            cs = [cv2.flip(c, 1) for c in crops] if flip else crops
            sfx = "_flip" if flip else ""
            o["buffalo_l" + sfx] = R.onnx["buffalo_l"].get_feat(cs)
            o["auraface" + sfx] = R.onnx["auraface"].get_feat(cs)
            o["sface" + sfx] = np.concatenate([R.sface.feature(c) for c in cs])
            x = torch.from_numpy(((np.stack(cs).astype(np.float32) / 255.0) - 0.5) / 0.5).permute(0, 3, 1, 2).contiguous().to(C.DEV)
            o["hyperface10k" + sfx] = hf(x)[0].float().cpu().numpy()
            if not flip:  # channel-order sanity checks
                o["auraface_bgrcheck"] = R.onnx["auraface"].get_feat([c[:, :, ::-1].copy() for c in cs])
                o["sface_rgbcheck"] = np.concatenate([R.sface.feature(c[:, :, ::-1].copy()) for c in cs])
        return {k: v.astype(np.float32) for k, v in o.items()}

    for ds, items in [("digiface", C.digiface_items(300)), ("celeba", C.celeba_items(10 ** 6))]:
        E, who, batch, bwho, miss, n = {}, [], [], [], 0, 0
        def flush():
            for k, v in emb(batch).items():
                E.setdefault(k, []).append(v)
            who.extend(bwho); batch.clear(); bwho.clear()
        for i, im in items:
            n += 1
            c = R.align(C.canvas(im, resize=(ds == "digiface")))
            if c is None:
                miss += 1; continue
            batch.append(c); bwho.append(i)
            if len(batch) == 256:
                flush()
        if batch:
            flush()
        np.savez(OUT / f"face_free_{ds}.npz", who=np.array(who), images=n, not_detected=miss,
                 **{k: np.concatenate(v) for k, v in E.items()})
        print(ds, "images", n, "not detected", miss, "embedded", len(who), flush=True)


# --------------------------------------------------------------------------------------------------------- analyze
def l2(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def rand_pair_std(E, n=3000, seed=1):  # spread of scores between random images (almost all different people)
    r = np.random.default_rng(seed).choice(len(E), min(n, len(E)), replace=False)
    S = E[r] @ E[r].T
    return float(S[np.triu_indices(len(r), 1)].std())


def variants(z):
    base = {k: l2(z[k]) for k in ["buffalo_l", "auraface", "sface", "hyperface10k", "auraface_bgrcheck", "sface_rgbcheck"]}
    V = dict(base)
    for k in ["buffalo_l", "auraface", "sface", "hyperface10k"]:
        V[k + "+flip"] = l2(l2(z[k]) + l2(z[k + "_flip"]))
    def fuse(names):
        parts = [V[n] / np.sqrt(rand_pair_std(V[n])) for n in names]   # score = sum_i s_i / std_i
        return l2(np.concatenate(parts, 1))
    V["auraface+sface (z-fused)"] = fuse(["auraface", "sface"])
    V["auraface+sface (z-fused) +flip"] = fuse(["auraface+flip", "sface+flip"])
    V["auraface+hyperface10k (z-fused) +flip"] = fuse(["auraface+flip", "hyperface10k+flip"])
    V["auraface+sface+hyperface10k (z-fused) +flip"] = fuse(["auraface+flip", "sface+flip", "hyperface10k+flip"])
    return V


def single_image(V, who):
    import eval_face_commercial as C
    res = {k: C.scores(E, who) for k, E in V.items()}
    fmr0 = float((res["buffalo_l"][1] >= 0.40).mean())
    out = {"buffalo_rate_at_0_40": fmr0}
    for k, (pos, neg) in res.items():
        row = dict(n_pos=int(len(pos)), n_neg=int(len(neg)))
        for lbl, fmr in [("at_buffalo_0.40_rate", fmr0), ("at_10x_lower_rate", fmr0 / 10)]:
            t = float(np.quantile(neg, 1 - fmr))
            row[lbl] = dict(threshold=round(t, 3), recall=round(float((pos >= t).mean()), 4), hits=int((pos >= t).sum()))
        out[k] = row
    return out


def impostor_top(E, who, k=200_000, chunk=2048):
    """Largest k scores over ALL pairs of images of different people (i<j) -> sorted descending."""
    keep = np.empty(0, np.float32)
    for s in range(0, len(E), chunk):
        S = E[s:s + chunk] @ E.T
        S[who[s:s + chunk, None] == who[None, :]] = -2         # same person: not an impostor pair
        rows = np.arange(s, s + S.shape[0])[:, None]
        S[np.arange(S.shape[1])[None, :] <= rows] = -2       # count each pair once (j > i)
        v = S[S > -2].ravel()
        if len(v) > k:
            v = np.partition(v, len(v) - k)[-k:]
        keep = np.concatenate([keep, v])
        if len(keep) > 4 * k:
            keep = np.partition(keep, len(keep) - k)[-k:]
    return np.sort(keep)[::-1], None


def map_thresholds(V, who):
    """buffalo_l cut c -> per-variant cut with the same pairwise wrong-match COUNT (same dataset)."""
    E = V["buffalo_l"]; counts = {c: 0 for c in (0.30, 0.40, 0.55)}
    for s in range(0, len(E), 2048):                      # exact buffalo_l impostor-pair counts at each cut
        S = E[s:s + 2048] @ E.T
        S[who[s:s + 2048, None] == who[None, :]] = -2
        S[np.arange(S.shape[1])[None, :] <= np.arange(s, s + S.shape[0])[:, None]] = -2
        for c in counts:
            counts[c] += int((S >= c).sum())
    kk = max(counts.values()) + 10
    n_pairs = int(sum(int((who != w).sum()) for w in who) // 2)
    M = {"_pairs": n_pairs, "_buffalo_wrong_pairs": counts}
    for k, E in V.items():
        top = impostor_top(E, who, k=kk)[0]
        M[k] = {c: (c if k == "buffalo_l" else float(top[min(max(cnt, 1), len(top)) - 1])) for c, cnt in counts.items()}
    return M


def product(E, who, th, n_refs, seed=0, keep_scores=False, floor=None):
    """people.py pipeline on a library of every image in the dataset (1 face per image, item = image)."""
    from findpics import people as P
    import pandas as pd
    n = len(E)
    idx = SimpleNamespace(face_emb=E.astype(np.float16), n_items=n,
                          faces=pd.DataFrame({"item_row": np.arange(n), "face_px": np.full(n, 100.0), "det_score": np.ones(n)}))
    E16 = idx.face_emb.astype(np.float32)          # what the product sees (float16 storage)
    G = P.face_groups(idx, top=12, accept=th[0.55])
    groups = [E16[np.array(g["faces"])] for g in G]
    # per face, max similarity to each group's faces (self-match ignored, as item_person_scores does)
    GM = np.full((n, len(groups)), -1.0, np.float32)
    for gi, g in enumerate(groups):
        S = E16 @ g.T; S[S > 0.999] = -1.0; GM[:, gi] = S.max(1)
    rng = np.random.default_rng(seed)
    hit = tot = wrong = maybe_hit = 0; checked = 0; T, I = [], []
    for q in np.unique(who):
        mine = np.where(who == q)[0]
        if len(mine) < 8 + 5:
            continue
        ref_rows = rng.choice(mine, n_refs, replace=False)
        refs = P.expand_refs(idx, idx.face_emb[ref_rows], accept=th[0.55], rounds=3)
        R = refs.astype(np.float32).T
        keep = [gi for gi, g in enumerate(groups) if float((g @ R).max(1).mean()) < th[0.40]]
        s = P.face_sims(idx, refs)
        if keep:
            s = np.where(GM[:, keep].max(1) >= s, -1.0, s)
        if checked < 3:   # the shortcut must equal people.item_person_scores exactly
            others = np.concatenate([groups[k] for k in keep]) if keep else np.zeros((0, E.shape[1]), np.float32)
            ref_s, _ = P.item_person_scores(idx, refs, others=others)
            assert np.allclose(ref_s, s, atol=1e-5), "shortcut != item_person_scores"
            checked += 1
        tgt = np.setdiff1d(mine, ref_rows); other = who != q
        hit += int((s[tgt] >= th[0.40]).sum()); tot += len(tgt)
        maybe_hit += int(((s[tgt] >= th[0.30]) & (s[tgt] < th[0.40])).sum())
        wrong += int((s[other] >= th[0.40]).sum())
        if keep_scores:
            T.append(s[tgt]); so = s[other]; I.append(so[so >= floor])
    if keep_scores:
        return np.concatenate(T), np.concatenate(I)
    return dict(recall=round(hit / tot, 4), hits=hit, targets=tot, wrong_items=wrong,
                precision=round(hit / max(1, hit + wrong), 4), in_possible_list=maybe_hit,
                thresholds={str(k): round(v, 3) for k, v in th.items()}, groups=len(groups))


def analyze():
    out = {}
    PRODUCT = ["buffalo_l", "buffalo_l+flip", "auraface", "auraface+flip", "sface", "sface+flip",
               "auraface+sface (z-fused) +flip", "hyperface10k+flip", "auraface+hyperface10k (z-fused) +flip",
               "auraface+sface+hyperface10k (z-fused) +flip"]
    for ds in ["digiface", "celeba"]:
        z = np.load(OUT / f"face_free_{ds}.npz")
        who = z["who"]; V = variants(z)
        r = dict(images=int(z["images"]), not_detected=int(z["not_detected"]), identities=int(len(np.unique(who))))
        r["single_image"] = single_image(V, who); print(ds, "single-image done", flush=True)
        M = map_thresholds({k: V[k] for k in PRODUCT}, who)
        r["thresholds"] = {k: (v if k.startswith("_") else {str(c): round(t, 4) for c, t in v.items()}) for k, v in M.items()}
        r["product"] = {}
        for k in PRODUCT:
            for nr in (3, 8):
                r["product"][f"{k} | {nr} refs"] = product(V[k], who, M[k], nr)
                print(ds, k, nr, r["product"][f"{k} | {nr} refs"], flush=True)
        out[ds] = r
        json.dump(out, open(f"{ROOT}/eval/results_face_free.json", "w"), indent=1)


def genuine_quantiles(E, who, qs, seed=0):
    """Quantiles of same-person pair scores (scale-free grid for the expansion cut)."""
    rng = np.random.default_rng(seed); v = []
    for q in np.unique(who):
        m = np.where(who == q)[0]
        S = E[m] @ E[m].T; v.append(S[np.triu_indices(len(m), 1)])
    v = np.concatenate(v)
    return {q: float(np.quantile(v, q)) for q in qs}


def sweep(ds):
    """Operating point tuned per model (also buffalo_l): for each expansion cut on a grid (quantiles of same-person
    scores, plus no expansion) and every final accept cut, the best recall whose wrong-item count is <= buffalo_l's at
    the shipped cuts (0.55 / 0.40). Uses dataset labels to tune: an upper bound for every model alike."""
    base = json.load(open(f"{ROOT}/eval/results_face_free.json"))[ds]
    z = np.load(OUT / f"face_free_{ds}.npz"); who = z["who"]; V = variants(z)
    names = ["buffalo_l", "auraface", "auraface+flip", "sface+flip", "auraface+sface (z-fused) +flip",
             "auraface+sface+hyperface10k (z-fused) +flip"]
    res = {}
    for k in names:
        E = V[k]; th0 = {float(c): t for c, t in base["thresholds"][k].items()}
        gq = genuine_quantiles(E, who, [0.2, 0.4, 0.6, 0.8])
        for nr in (3, 8):
            W = base["product"][f"buffalo_l | {nr} refs"]["wrong_items"]
            best = None
            for lbl, ce in [("none", 1.01), ("mapped_0.55", th0[0.55])] + [(f"genuine_q{q}", c) for q, c in gq.items()]:
                th = dict(th0); th[0.55] = ce
                T, I = product(E, who, th, nr, keep_scores=True, floor=th0[0.30] - 0.1)
                Is = np.sort(I)[::-1]
                t = Is[W] + 1e-6 if len(Is) > W else th0[0.30] - 0.1   # smallest cut with <= W wrong items
                r = dict(expand_cut=lbl, expand_value=round(ce, 3), accept_cut=round(float(t), 4),
                         recall=round(float((T >= t).mean()), 4), hits=int((T >= t).sum()), targets=int(len(T)),
                         wrong_items=int((I >= t).sum()), wrong_budget=W)
                print(ds, k, nr, r, flush=True)
                if best is None or r["recall"] > best["recall"]:
                    best = r
            res[f"{k} | {nr} refs"] = best
        json.dump(res, open(f"{ROOT}/eval/results_face_free_sweep_{ds}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep(sys.argv[2])
    else:
        {"embed": embed_all, "analyze": analyze}[sys.argv[1]]()
