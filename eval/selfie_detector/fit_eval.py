"""M46 step 3: fit + evaluate selfie (front camera) detectors on the public Commons set, splits BY UPLOADER.

Methods
  geo_rule : one threshold on the largest face's area fraction (score = that fraction; 0 when no face).
  geo_lr   : logistic regression on 6 face-geometry features (largest area frac, its centre offset, its centre y,
             n faces, 2nd/1st area ratio, has-face flag).
  vec_lr   : logistic regression on the 1024-d PE-Core-B-16 image vector.
  combo_lr : logistic regression on [vector, geometry features].
Evaluation: 5-fold GroupKFold by uploader -> out-of-fold scores for every photo. AUC + 95% CI by uploader-cluster
bootstrap. Operating points at precision 0.90 / 0.95 chosen HONESTLY: the threshold is picked on each outer-train fold
(from inner grouped CV scores) and applied to the held-out fold. Precision is also re-weighted to a phone-like
prevalence (P = pi*TPR / (pi*TPR + (1-pi)*FPR)).
Then: final weights fitted on all data -> weights.json; and a learning curve: AUC / recall when only N positives
(e.g. 267, like the phone's tagged set) are available for training.

Usage: python eval/selfie_detector/fit_eval.py LABELS.jsonl FEATURES.npz OUT_DIR
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

RNG = np.random.default_rng(0)
PHONE_PI = 267 / 17000   # ~ front-tagged / originals on the device (187k - ~170k in iCloud)
GEO_NAMES = ["area_frac", "centre_off", "centre_y", "n_faces", "second_ratio", "has_face"]


def geometry(faces, W, H, min_score=0.5):
    fs = [f for f in faces if f[4] >= min_score]
    if not fs:
        return [0.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    areas = sorted((((f[2] - f[0]) * (f[3] - f[1])) / (W * H), f) for f in fs)[::-1]
    a, f = areas[0]
    cx, cy = (f[0] + f[2]) / 2 / W, (f[1] + f[3]) / 2 / H
    off = float(np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5))
    second = areas[1][0] / a if len(areas) > 1 else 0.0
    return [float(a), off, float(cy), float(min(len(fs), 10)), float(second), 1.0]


def geo_design(G):
    """Geometry features as fed to the logistic models (sqrt area makes it ~linear in face size)."""
    return np.column_stack([np.sqrt(G[:, 0]), G[:, 1], G[:, 2], np.log1p(G[:, 3]), G[:, 4], G[:, 5]])


class Std:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


def fit_predict(method, Xv, G, y, tr, te, C=None):
    if method == "geo_rule":
        return G[te, 0], None
    if method == "geo_lr":
        X = geo_design(G)
        sc = Std().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc(X[tr]), y[tr])
        return m.decision_function(sc(X[te])), (m, sc)
    if method == "vec_lr":
        X = Xv
    else:
        X = np.hstack([Xv, geo_design(G)])
    sc = Std().fit(X[tr])
    m = LogisticRegression(C=C or 0.05, max_iter=5000).fit(sc(X[tr]), y[tr])
    return m.decision_function(sc(X[te])), (m, sc)


def thr_at_precision(s, y, p):
    o = np.argsort(-s)
    tp = np.cumsum(y[o]); k = np.arange(1, len(o) + 1)
    prec = tp / k
    ok = np.where(prec >= p)[0]
    if len(ok) == 0:
        return np.inf
    j = ok[np.argmax(tp[ok])]           # most recall among cut points meeting the precision
    return s[o][j]


def boot_ci(fn, users, n=1000):
    u = np.unique(users)
    idx = {x: np.where(users == x)[0] for x in u}
    vals = []
    for _ in range(n):
        pick = RNG.choice(u, len(u), replace=True)
        ii = np.concatenate([idx[x] for x in pick])
        try:
            vals.append(fn(ii))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])


def main():
    labels_fn, feat_fn, out_dir = sys.argv[1:4]
    rows = {json.loads(l)["pageid"]: json.loads(l) for l in open(labels_fn)}
    F = np.load(feat_fn)
    faces = json.loads(str(F["faces_json"]))
    pid = F["pageid"]
    keep = [i for i, p in enumerate(pid) if int(p) in rows]
    pid = pid[keep]
    Xv = F["vec"][keep].astype(np.float32)
    G = np.array([geometry(faces[i], F["W"][i], F["H"][i]) for i in keep])
    y = np.array([rows[int(p)]["label"] for p in pid])
    users = np.array([rows[int(p)]["user"] for p in pid])
    hasface = G[:, 5] > 0
    print(f"N={len(y)} pos={y.sum()} neg={(1 - y).sum()} users={len(np.unique(users))} "
          f"with face: pos {y[hasface].sum()}/{y.sum()}, neg {(1 - y)[hasface].sum()}/{(1 - y).sum()}", flush=True)

    # C for the vector models: grouped CV on a small grid (cheap; chosen once, reported)
    gkf = GroupKFold(5)
    folds = list(gkf.split(Xv, y, users))
    Cs = {}
    for method in ["vec_lr", "combo_lr"]:
        best = None
        for C in [0.005, 0.02, 0.05, 0.2]:
            oof = np.zeros(len(y))
            for tr, te in folds:
                oof[te] = fit_predict(method, Xv, G, y, tr, te, C)[0]
            a = roc_auc_score(y, oof)
            print(method, "C", C, "oof AUC", round(a, 4), flush=True)
            if best is None or a > best[0]:
                best = (a, C)
        Cs[method] = best[1]

    methods = ["geo_rule", "geo_lr", "vec_lr", "combo_lr"]
    res, oofs = {}, {}
    for method in methods:
        oof = np.zeros(len(y))
        ops = {0.9: [], 0.95: []}
        for tr, te in folds:
            oof[te] = fit_predict(method, Xv, G, y, tr, te, Cs.get(method))[0]
            # inner grouped CV on the outer-train part -> threshold for each target precision
            inner = np.zeros(len(tr))
            for itr, ite in GroupKFold(4).split(tr, y[tr], users[tr]):
                inner[ite] = fit_predict(method, Xv, G, y, tr[itr], tr[ite], Cs.get(method))[0]
            for p in ops:
                ops[p].append((te, thr_at_precision(inner, y[tr], p)))
        oofs[method] = oof
        r = {"auc": roc_auc_score(y, oof),
             "auc_ci": boot_ci(lambda ii: roc_auc_score(y[ii], oof[ii]), users).tolist(),
             "auc_face": roc_auc_score(y[hasface], oof[hasface]),
             "auc_noface": (roc_auc_score(y[~hasface], oof[~hasface]) if 0 < y[~hasface].sum() < (~hasface).sum() else None)}
        for p, lst in ops.items():
            pred = np.zeros(len(y), bool)
            for te, t in lst:
                pred[te] = oof[te] >= t
            def stats(ii, pred=pred):
                tp = (pred[ii] & (y[ii] == 1)).sum(); fp = (pred[ii] & (y[ii] == 0)).sum()
                P = (y[ii] == 1).sum(); Nn = (y[ii] == 0).sum()
                tpr, fpr = tp / max(P, 1), fp / max(Nn, 1)
                prec = tp / max(tp + fp, 1)
                pw = PHONE_PI * tpr / max(PHONE_PI * tpr + (1 - PHONE_PI) * fpr, 1e-12)
                return dict(tp=int(tp), fp=int(fp), pos=int(P), neg=int(Nn), precision=prec, recall=tpr, fpr=fpr,
                            precision_at_phone_prevalence=pw)
            allv = stats(np.arange(len(y)))
            allv["precision_ci"] = boot_ci(lambda ii: stats(ii)["precision"], users, 500).tolist()
            allv["recall_ci"] = boot_ci(lambda ii: stats(ii)["recall"], users, 500).tolist()
            allv["face_subset"] = stats(np.where(hasface)[0])
            allv["noface_subset"] = stats(np.where(~hasface)[0])
            r[f"op_p{p}"] = allv
        res[method] = r
        print(method, json.dumps(r, default=float)[:600], flush=True)

    # learning curve: train on only n positives (all negatives of those users' folds kept), test on held-out users
    lc = {}
    for n_pos in [100, 267, 500, 1000]:
        aucs = {"vec_lr": [], "combo_lr": [], "geo_lr": []}
        for tr, te in folds:
            ptr = tr[y[tr] == 1]
            if len(ptr) < n_pos:
                continue
            for rep in range(3):
                sub = np.concatenate([RNG.choice(ptr, n_pos, replace=False),
                                      RNG.choice(tr[y[tr] == 0], min((y[tr] == 0).sum(), 20 * n_pos), replace=False)])
                for m in aucs:
                    s = fit_predict(m, Xv, G, y, sub, te, Cs.get(m))[0]
                    aucs[m].append(roc_auc_score(y[te], s))
        lc[n_pos] = {m: (float(np.mean(v)), float(np.std(v)), len(v)) for m, v in aucs.items() if v}
        print("learning curve", n_pos, lc[n_pos], flush=True)

    # final weights on all data (public only)
    out = {"model": "PE-Core-B-16 image vector (L2-normalized, open_clip squash-224) + SCRFD face boxes",
           "trained_on": f"Wikimedia Commons iPhone photos, EXIF front/back label, N={len(y)}, pos={int(y.sum())}",
           "geometry_features": GEO_NAMES, "geo_design": "sqrt(area_frac), centre_off, centre_y, log1p(n_faces), second_ratio, has_face",
           "score": "logit = w . ((x - mean) / std) + b ; selfie if logit >= threshold"}
    all_idx = np.arange(len(y))
    for method in ["geo_lr", "vec_lr", "combo_lr"]:
        _, (m, sc) = fit_predict(method, Xv, G, y, all_idx, all_idx[:1], Cs.get(method))
        oof = oofs[method]
        out[method] = {"C": Cs.get(method, 1.0), "mean": sc.m.tolist(), "std": sc.s.tolist(),
                       "w": m.coef_[0].tolist(), "b": float(m.intercept_[0]),
                       "threshold_p0.9": float(thr_at_precision(oof, y, 0.9)),
                       "threshold_p0.95": float(thr_at_precision(oof, y, 0.95)),
                       "note": "thresholds = from out-of-fold scores at Commons prevalence; re-pick on the phone's tagged set"}
    out["geo_rule"] = {"feature": "largest face area / frame area (SCRFD box)",
                       "threshold_p0.9": float(thr_at_precision(oofs["geo_rule"], y, 0.9)),
                       "threshold_p0.95": float(thr_at_precision(oofs["geo_rule"], y, 0.95))}
    json.dump(out, open(f"{out_dir}/weights.json", "w"))
    json.dump({"results": res, "learning_curve": lc, "C": Cs,
               "n": int(len(y)), "pos": int(y.sum()), "users": int(len(np.unique(users)))},
              open(f"{out_dir}/results.json", "w"), default=float, indent=1)
    np.savez(f"{out_dir}/oof_scores.npz", pageid=pid, y=y, users=users, G=G, **oofs)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
