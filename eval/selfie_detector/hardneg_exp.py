"""M50 follow-up: can a refit separate CLOSE-FACE back-camera portraits from selfies better? (CPU, Slurm)

Finding that motivates it: on Commons, the photos whose largest face covers >= 5% of the frame
(2,521 front / 282 back) reproduce the phone's tagged-set numbers: AUC 0.96, FPR@0.5 about 0.2, recall at 95%
precision (at the phone's 26% prevalence) about 0.5. So the gap is the negative mix, not the rendition.
Experiments on the phone-simulated vectors (r448q80), same 5-fold GroupKFold by uploader, C = 0.005:
  base       : as fit_eval_phone C_phone
  w5 / w20   : back-camera photos with a face >= 2% of the frame weighted x5 / x20 in the loss
  face_only  : trained only on photos with a face >= 2% (a specialist for the hard case)
  combo      : vector + face-geometry features (SCRFD boxes)
Metrics: overall AUC; AUC inside face >= 5%; FPR@0.5 inside face >= 5%; at the phone's prevalence (581/2252) the
recall at 95% precision inside face >= 5% (few negatives above the cut: noisy, counts printed).
Usage: python eval/selfie_detector/hardneg_exp.py   (reads data/public/raw/commons_iphone + eval/selfie_detector)
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "eval/selfie_detector")
from fit_eval import Std, geo_design  # noqa: E402

PI = 581 / 2252


def t95(sp, sn):
    best = (np.inf, 0.0, 0.0, 0)
    for t in np.unique(np.concatenate([sp, sn])):
        tpr = (sp >= t).mean(); fpr = (sn >= t).mean()
        if tpr == 0:
            continue
        prec = PI * tpr / (PI * tpr + (1 - PI) * fpr)
        if prec >= 0.95 and tpr > best[1]:
            best = (float(t), float(tpr), float(fpr), int((sn >= t).sum()))
    return best


def main():
    O = np.load("eval/selfie_detector/oof_scores.npz")
    S = np.load("data/public/raw/commons_iphone/phone_sim.npz")
    pos_of = {int(p): i for i, p in enumerate(S["pageid"])}
    idx = np.array([pos_of[int(p)] for p in O["pageid"]])
    X = S["r448q80"][idx].astype(np.float32)
    y, users, G = O["y"], O["users"], O["G"]
    a = G[:, 0]
    big = a >= 0.05
    folds = list(GroupKFold(5).split(X, y, users))
    Gd = geo_design(G)
    res = {}
    for name in ["base", "w5", "w20", "face_only", "combo"]:
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            Xtr, Xte = X[tr], X[te]
            if name == "combo":
                Xtr, Xte = np.hstack([Xtr, Gd[tr]]), np.hstack([Xte, Gd[te]])
            sw = np.ones(len(tr))
            if name in ("w5", "w20"):
                sw[(y[tr] == 0) & (a[tr] >= 0.02)] = 5 if name == "w5" else 20
            keep = np.ones(len(tr), bool) if name != "face_only" else a[tr] >= 0.02
            sc = Std().fit(Xtr[keep])
            m = LogisticRegression(C=0.005, max_iter=5000).fit(sc(Xtr[keep]), y[tr][keep], sample_weight=sw[keep])
            oof[te] = m.decision_function(sc(Xte))
        sp, sn = oof[big & (y == 1)], oof[big & (y == 0)]
        r = dict(auc_all=roc_auc_score(y, oof), auc_big=roc_auc_score(y[big], oof[big]),
                 fpr05_big=float((sn >= 0.5).mean()), recall05_big=float((sp >= 0.5).mean()),
                 fpr05_all=float((oof[y == 0] >= 0.5).mean()), recall05_all=float((oof[y == 1] >= 0.5).mean()),
                 t95_big=t95(sp, sn))
        res[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(res, open("eval/selfie_detector/hardneg_exp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
