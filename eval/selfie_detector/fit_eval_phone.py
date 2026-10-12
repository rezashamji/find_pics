"""M50 follow-up: does the 1,280 px fit's threshold break on PHONE-SIMULATED vectors, and does a refit fix it?

Inputs: labels.jsonl, features.npz (1,280 px vectors + SCRFD boxes; extract_features.py), phone_sim.npz
(extract_phone_sim.py: r448q80 = primary phone simulation, r480q80, r448q95, r448raw).
Same 5-fold GroupKFold BY UPLOADER as fit_eval.py (same row order -> same folds).

  A  old      : the shipped weights.json vec_lr (fitted on ALL 1,280 px photos) scored on every variant
                (in-sample photos, so slightly optimistic; B is the honest version).
  B  cross    : per fold, fit on 1,280 px train vectors, score the held-out users' vectors of every variant.
  C  phone    : per fold, fit on r448q80 train vectors (C by grouped CV), score held-out users on every variant;
                honest operating points (threshold for precision >= 0.90 / 0.95 picked by inner grouped CV on the
                outer-train part) on r448q80.
  D  mix      : per fold, fit on ALL five renderings of the train photos stacked (1,280 + the four phone variants;
                one photo's renderings never straddle a fold), score held-out users on every variant; honest
                operating points on r448q80.
  combo/geo   : refit on r448q80 for weights_phone.json's layout (geometry = SCRFD boxes from the 1,280 px image,
                as before; the phone's Vision boxes differ - the recommended vec_lr does not use them).
Writes OUT_DIR/results_phone.json, OUT_DIR/weights_phone.json (same layout as weights.json), oof_scores_phone.npz.
Usage: python eval/selfie_detector/fit_eval_phone.py LABELS.jsonl FEATURES.npz PHONE_SIM.npz OUT_DIR
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "eval/selfie_detector")
from fit_eval import GEO_NAMES, Std, boot_ci, geo_design, geometry, thr_at_precision  # noqa: E402

VARS = ["r448q80", "r480q80", "r448q95", "r448raw"]
PHONE_PI = 267 / 17000


def lr(X, y, C):
    sc = Std().fit(X)
    m = LogisticRegression(C=C, max_iter=5000).fit(sc(X), y)
    return m, sc


def score(model, X):
    m, sc = model
    return m.decision_function(sc(X))


def rates(s, y, t):
    p = s >= t
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    P = int((y == 1).sum()); N = int((y == 0).sum())
    return dict(tp=tp, fp=fp, pos=P, neg=N, recall=tp / P, fpr=fp / N, precision=tp / max(tp + fp, 1))


def summarize(s, y, hasface, users, t=0.5):
    r = dict(auc=roc_auc_score(y, s), at05=rates(s, y, t),
             at05_face=rates(s[hasface], y[hasface], t),
             mean_logit_neg=float(s[y == 0].mean()), mean_logit_pos=float(s[y == 1].mean()))
    r["auc_ci"] = boot_ci(lambda ii: roc_auc_score(y[ii], s[ii]), users, 300).tolist()
    return r


def honest_ops(fit_fn, Xtr_fn, Xte, y, users, folds, hasface):
    """Thresholds for precision >= p picked on inner grouped CV of each outer-train part, applied to held-out users.
    fit_fn(train_idx) -> model; Xtr_fn(model, idx) -> inner scores on the r448q80 rendering."""
    out = {}
    oof = np.zeros(len(y))
    ths = {0.9: [], 0.95: []}
    for tr, te in folds:
        model = fit_fn(tr)
        oof[te] = score(model, Xte[te])
        inner = np.zeros(len(tr))
        for itr, ite in GroupKFold(4).split(tr, y[tr], users[tr]):
            inner[ite] = score(fit_fn(tr[itr]), Xte[tr[ite]])
        for p in ths:
            ths[p].append((te, thr_at_precision(inner, y[tr], p)))
    for p, lst in ths.items():
        pred = np.zeros(len(y), bool)
        for te, t in lst:
            pred[te] = oof[te] >= t

        def st(ii, pred=pred):
            tp = int((pred[ii] & (y[ii] == 1)).sum()); fp = int((pred[ii] & (y[ii] == 0)).sum())
            P = int((y[ii] == 1).sum()); N = int((y[ii] == 0).sum())
            tpr, fpr = tp / max(P, 1), fp / max(N, 1)
            return dict(tp=tp, fp=fp, pos=P, neg=N, precision=tp / max(tp + fp, 1), recall=tpr, fpr=fpr,
                        precision_at_phone_prevalence=PHONE_PI * tpr / max(PHONE_PI * tpr + (1 - PHONE_PI) * fpr, 1e-12))
        o = st(np.arange(len(y)))
        o["precision_ci"] = boot_ci(lambda ii: st(ii)["precision"], users, 300).tolist()
        o["recall_ci"] = boot_ci(lambda ii: st(ii)["recall"], users, 300).tolist()
        o["face_subset"] = st(np.where(hasface)[0])
        o["noface_subset"] = st(np.where(~hasface)[0])
        o["fold_thresholds"] = [float(t) for _, t in lst]
        out[f"op_p{p}"] = o
    return oof, out


def main():
    labels_fn, feat_fn, sim_fn, out_dir = sys.argv[1:5]
    rows = {json.loads(l)["pageid"]: json.loads(l) for l in open(labels_fn)}
    F = np.load(feat_fn)
    faces = json.loads(str(F["faces_json"]))
    pid = F["pageid"]
    keep = [i for i, p in enumerate(pid) if int(p) in rows]
    pid = pid[keep]
    X = {"r1280": F["vec"][keep].astype(np.float32)}
    G = np.array([geometry(faces[i], F["W"][i], F["H"][i]) for i in keep])
    S = np.load(sim_fn)
    pos_of = {int(p): i for i, p in enumerate(S["pageid"])}
    sidx = np.array([pos_of[int(p)] for p in pid])
    for v in VARS:
        X[v] = S[v][sidx].astype(np.float32)
    y = np.array([rows[int(p)]["label"] for p in pid])
    users = np.array([rows[int(p)]["user"] for p in pid])
    hasface = G[:, 5] > 0
    folds = list(GroupKFold(5).split(X["r1280"], y, users))
    print(f"N={len(y)} pos={y.sum()} users={len(np.unique(users))}", flush=True)
    for v in VARS:
        cos = (X["r1280"] * X[v]).sum(1) / np.linalg.norm(X["r1280"], axis=1) / np.linalg.norm(X[v], axis=1)
        print(f"cosine r1280 vs {v}: mean {cos.mean():.4f} p5 {np.percentile(cos, 5):.4f} min {cos.min():.4f}", flush=True)
    R = {"cosine_vs_1280": {v: dict(mean=float(c.mean()), p5=float(np.percentile(c, 5)), min=float(c.min()))
                            for v in VARS for c in [(X["r1280"] * X[v]).sum(1) / np.linalg.norm(X["r1280"], axis=1)
                                                    / np.linalg.norm(X[v], axis=1)]}}
    allv = ["r1280"] + VARS

    # A: the shipped weights on every rendering
    W = json.load(open(f"{out_dir}/weights.json"))["vec_lr"]
    wf, bf = np.array(W["w_folded"]), W["b_folded"]
    R["A_old_weights"] = {v: summarize(X[v] @ wf + bf, y, hasface, users) for v in allv}
    for v in allv:
        a = R["A_old_weights"][v]
        print(f"A old weights on {v}: AUC {a['auc']:.4f} recall@0.5 {a['at05']['recall']:.3f} FPR@0.5 "
              f"{a['at05']['fpr']:.4f} mean logit neg {a['mean_logit_neg']:.2f} pos {a['mean_logit_pos']:.2f}", flush=True)

    # B: fit on 1,280 per fold, score held-out on every rendering
    C0 = 0.005
    oofB = {v: np.zeros(len(y)) for v in allv}
    for tr, te in folds:
        m = lr(X["r1280"][tr], y[tr], C0)
        for v in allv:
            oofB[v][te] = score(m, X[v][te])
    R["B_cross"] = {v: summarize(oofB[v], y, hasface, users) for v in allv}
    for v in allv:
        a = R["B_cross"][v]
        print(f"B fit 1280 -> test {v}: AUC {a['auc']:.4f} recall@0.5 {a['at05']['recall']:.3f} FPR@0.5 "
              f"{a['at05']['fpr']:.4f}", flush=True)

    # C: fit on r448q80; C by grouped CV
    best = None
    for C in [0.002, 0.005, 0.02, 0.05]:
        oof = np.zeros(len(y))
        for tr, te in folds:
            oof[te] = score(lr(X["r448q80"][tr], y[tr], C), X["r448q80"][te])
        a = roc_auc_score(y, oof)
        print("C phone", C, "oof AUC", round(a, 4), flush=True)
        if best is None or a > best[0]:
            best = (a, C)
    Cp = best[1]
    oofC = {v: np.zeros(len(y)) for v in allv}
    for tr, te in folds:
        m = lr(X["r448q80"][tr], y[tr], Cp)
        for v in allv:
            oofC[v][te] = score(m, X[v][te])
    R["C_phone"] = {v: summarize(oofC[v], y, hasface, users) for v in allv}
    _, R["C_phone_ops_r448q80"] = honest_ops(lambda idx: lr(X["r448q80"][idx], y[idx], Cp), None, X["r448q80"],
                                             y, users, folds, hasface)
    for v in allv:
        a = R["C_phone"][v]
        print(f"C fit r448q80 -> test {v}: AUC {a['auc']:.4f} recall@0.5 {a['at05']['recall']:.3f} FPR@0.5 "
              f"{a['at05']['fpr']:.4f}", flush=True)
    print("C ops", json.dumps({k: {kk: vv for kk, vv in o.items() if kk in ('precision', 'recall', 'tp', 'fp', 'fpr')}
                                for k, o in R["C_phone_ops_r448q80"].items()}), flush=True)

    # D: fit on all renderings stacked
    def fit_mix(idx, C=Cp):
        Xs = np.concatenate([X[v][idx] for v in allv]); ys = np.tile(y[idx], len(allv))
        return lr(Xs, ys, C)
    oofD = {v: np.zeros(len(y)) for v in allv}
    for tr, te in folds:
        m = fit_mix(tr)
        for v in allv:
            oofD[v][te] = score(m, X[v][te])
    R["D_mix"] = {v: summarize(oofD[v], y, hasface, users) for v in allv}
    _, R["D_mix_ops_r448q80"] = honest_ops(fit_mix, None, X["r448q80"], y, users, folds, hasface)
    for v in allv:
        a = R["D_mix"][v]
        print(f"D fit mix -> test {v}: AUC {a['auc']:.4f} recall@0.5 {a['at05']['recall']:.3f} FPR@0.5 "
              f"{a['at05']['fpr']:.4f}", flush=True)
    print("D ops", json.dumps({k: {kk: vv for kk, vv in o.items() if kk in ('precision', 'recall', 'tp', 'fp', 'fpr')}
                                for k, o in R["D_mix_ops_r448q80"].items()}), flush=True)

    # combo / geo on r448q80 (layout of weights.json)
    Gd = geo_design(G)
    oof_combo = np.zeros(len(y)); oof_geo = np.zeros(len(y))
    for tr, te in folds:
        oof_combo[te] = score(lr(np.hstack([X["r448q80"][tr], Gd[tr]]), y[tr], Cp), np.hstack([X["r448q80"][te], Gd[te]]))
        oof_geo[te] = score(lr(Gd[tr], y[tr], 1.0), Gd[te])
    R["combo_r448q80_auc"] = roc_auc_score(y, oof_combo)
    R["geo_lr_auc"] = roc_auc_score(y, oof_geo)
    R["C_phone_C"] = Cp
    print("combo AUC", R["combo_r448q80_auc"], "geo AUC", R["geo_lr_auc"], flush=True)

    # final weights: choose C or D by r448q80 AUC and FPR spread across renderings
    def spread(block):
        f = [R[block][v]["at05"]["fpr"] for v in allv]
        return max(f) - min(f)
    choice = "D_mix" if (R["D_mix"]["r448q80"]["auc"] >= R["C_phone"]["r448q80"]["auc"] - 0.001
                         and spread("D_mix") < spread("C_phone")) else "C_phone"
    R["choice"] = choice
    R["fpr_spread_at05"] = {b: spread(b) for b in ["B_cross", "C_phone", "D_mix"]}
    all_idx = np.arange(len(y))
    out = {"model": "PE-Core-B-16 image vector, PHONE-SIMULATED (448 px long side Lanczos + JPEG q80 4:2:0 round trip, "
                    "then open_clip squash-224, fp16) - " + ("trained on all renderings stacked (1,280 + r448q80 + "
                    "r480q80 + r448q95 + r448raw)" if choice == "D_mix" else "trained on r448q80 only"),
           "trained_on": f"Wikimedia Commons iPhone photos, EXIF front/back label, N={len(y)}, pos={int(y.sum())}",
           "geometry_features": GEO_NAMES,
           "geo_design": "sqrt(area_frac), centre_off, centre_y, log1p(n_faces), second_ratio, has_face",
           "score": "logit = dot(w_folded, v) + b_folded on the stored L2-normalized vector; selfie if logit >= threshold",
           "calibration_note": "thresholds below are Commons out-of-fold numbers; the phone keeps its per-library "
                               "calibration on its own tagged photos (PhotoIndex.selfieThreshold)"}

    def pack(model, C, oof, extra=None):
        m, sc = model
        w = m.coef_[0]; b = float(m.intercept_[0])
        d = {"C": C, "mean": sc.m.tolist(), "std": sc.s.tolist(), "w": w.tolist(), "b": b,
             "w_folded": (w / sc.s).tolist(), "b_folded": b - float((w * sc.m / sc.s).sum()),
             "threshold_p0.9": float(thr_at_precision(oof, y, 0.9)),
             "threshold_p0.95": float(thr_at_precision(oof, y, 0.95))}
        d.update(extra or {})
        return d
    vec_model = fit_mix(all_idx) if choice == "D_mix" else lr(X["r448q80"], y, Cp)
    oof_main = oofD["r448q80"] if choice == "D_mix" else oofC["r448q80"]
    tab = []
    for t in [-1, 0, 0.5, 1, 1.5, 2, 3]:
        r = rates(oof_main, y, t); rf = rates(oof_main[hasface], y[hasface], t)
        tab.append(dict(logit=t, recall=r["recall"], fpr_all=r["fpr"], recall_face=rf["recall"], fpr_face=rf["fpr"],
                        prec_commons=r["precision"]))
    out["vec_lr"] = pack(vec_model, Cp, oof_main, {"threshold_table_oof_r448q80": tab, "variant": choice})
    alt = lr(X["r448q80"], y, Cp) if choice == "D_mix" else fit_mix(all_idx)
    out["vec_lr_alt"] = pack(alt, Cp, oofC["r448q80"] if choice == "D_mix" else oofD["r448q80"],
                             {"variant": "C_phone" if choice == "D_mix" else "D_mix"})
    out["combo_lr"] = pack(lr(np.hstack([X["r448q80"], Gd]), y, Cp), Cp, oof_combo)
    out["geo_lr"] = pack(lr(Gd, y, 1.0), 1.0, oof_geo)
    out["geo_rule"] = {"feature": "largest face area / frame area (SCRFD box)",
                       "threshold_p0.9": float(thr_at_precision(G[:, 0], y, 0.9)),
                       "threshold_p0.95": float(thr_at_precision(G[:, 0], y, 0.95))}
    json.dump(out, open(f"{out_dir}/weights_phone.json", "w"))
    json.dump(R, open(f"{out_dir}/results_phone.json", "w"), default=float, indent=1)
    np.savez(f"{out_dir}/oof_scores_phone.npz", pageid=pid, y=y, users=users, hasface=hasface,
             **{f"B_{v}": oofB[v] for v in allv}, **{f"C_{v}": oofC[v] for v in allv},
             **{f"D_{v}": oofD[v] for v in allv})
    print("choice", choice, "spread", R["fpr_spread_at05"], "saved", flush=True)


if __name__ == "__main__":
    main()
