"""M46: text tables from results.json + oof_scores.npz, and id lists for the eye audit of the recommended method.

Precision is reported three ways, because the Commons class mix is not a phone's:
  at Commons prevalence (what the held-out folds contain),
  at phone prevalence pi=267/17000 (front-tagged / originals on Reza's phone) using the fold TPR/FPR,
  on the FACE SUBSET at pi_face = 0.9*267 / (0.425*17000) (faces in 42.5% of his photos: 79,576 / 187k; ~90% of
  front shots have a detected face on Commons: see report), using face-subset TPR/FPR.
Usage: python eval/selfie_detector/tables.py eval/selfie_detector METHOD_FOR_AUDIT
"""
from __future__ import annotations

import json
import sys

import numpy as np

PI = 267 / 17000
PI_FACE = 0.9 * 267 / (0.425 * 17000)


def at_pi(tpr, fpr, pi):
    d = pi * tpr + (1 - pi) * fpr
    return pi * tpr / d if d > 0 else float("nan")


def main():
    d, audit = sys.argv[1], sys.argv[2]
    R = json.load(open(f"{d}/results.json"))
    res = R["results"]
    print(f"N={R['n']} pos={R['pos']} users={R['users']}  C={R['C']}")
    print(f"{'method':9s} {'AUC [95% CI]':24s} {'AUC face':9s} {'AUC noface':10s}")
    for m, r in res.items():
        nf = r["auc_noface"]
        print(f"{m:9s} {r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]   {r['auc_face']:.3f}     "
              f"{nf if nf is None else round(nf, 3)}")
    for p in ["0.9", "0.95"]:
        print(f"\nOperating point picked on train folds for precision >= {p} (applied to held-out users):")
        print(f"{'method':9s} {'prec [CI]':22s} {'recall [CI]':22s} {'tp/pos':10s} {'fp/neg':10s} "
              f"{'P@phone':8s} | face: {'prec':5s} {'rec':5s} {'P@phone_face':12s} | noface: tp/pos fp/neg")
        for m, r in res.items():
            o = r[f"op_p{p}"]
            f, nf = o["face_subset"], o["noface_subset"]
            print(f"{m:9s} {o['precision']:.3f} [{o['precision_ci'][0]:.3f},{o['precision_ci'][1]:.3f}]  "
                  f"{o['recall']:.3f} [{o['recall_ci'][0]:.3f},{o['recall_ci'][1]:.3f}]  "
                  f"{o['tp']}/{o['pos']:<5d} {o['fp']}/{o['neg']:<6d} {o['precision_at_phone_prevalence']:.3f}    | "
                  f"{f['precision']:.3f} {f['recall']:.3f} {at_pi(f['recall'], f['fpr'], PI_FACE):.3f}        | "
                  f"{nf['tp']}/{nf['pos']} {nf['fp']}/{nf['neg']}")
    print("\nLearning curve (AUC on held-out users when training on n positives + 20n negatives; mean, sd, runs):")
    for n, v in R["learning_curve"].items():
        print(f"  n_pos={n:>5s}: " + "  ".join(f"{m} {a[0]:.3f}+-{a[1]:.3f} ({a[2]})" for m, a in v.items()))

    S = np.load(f"{d}/oof_scores.npz")
    s, y, pid = S[audit], S["y"], S["pageid"]
    t = json.load(open(f"{d}/weights.json"))[audit]["threshold_p0.9"]
    fp = np.where((s >= t) & (y == 0))[0]
    fp = fp[np.argsort(-s[fp])]
    fn = np.where((s < t) & (y == 1))[0]
    rng = np.random.default_rng(0)
    fn = rng.choice(fn, min(18, len(fn)), replace=False) if len(fn) else fn
    tp = np.where((s >= t) & (y == 1))[0]
    tp = rng.choice(tp, min(12, len(tp)), replace=False)
    for name, ii in [("audit_fp", fp[:24]), ("audit_fn", fn), ("audit_tp", tp)]:
        with open(f"{d}/../../.cache/tmp/selfie/{name}.txt", "w") as fh:
            for i in ii:
                fh.write(f"{pid[i]} {s[i]:.3f}\n")
    print(f"\naudit lists ({audit}, oof threshold {t:.3f}): fp {len(fp)} total, wrote top 24; fn sample {len(fn)}; tp sample {len(tp)}")


if __name__ == "__main__":
    main()
