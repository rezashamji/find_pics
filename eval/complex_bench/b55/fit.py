"""RESULTS 55: fit the pixel-veto thresholds on TRAIN libraries only (rule pre-registered in JOURNAL 10-13 ~05:15).
  T_bw    = 97.5th percentile of frac_chroma_gt30 over TRAIN b truth photos of black-and-white requests
  T_over  = 10th percentile of frac_clip_hi over TRAIN photos the facts tag quality "overexposed"
  T_blur  = 90th percentile of grad_mean over TRAIN b truth photos of blurry requests
Also reports how well each statistic separates (AUC: tagged/truth photos vs the rest of their libraries) on train.
Usage: python eval/complex_bench/b55/fit.py -> eval/complex_bench/b55/thresholds.json, fit.txt
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, load_facts  # noqa: E402

B55 = ROOT / "b55"
BW = re.compile(r"(?i)black[- ]and[- ]white|\bb ?& ?w\b|monochrome|gr[ae]y ?scale")
BLUR = re.compile(r"(?i)blurr|out of focus")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    train = json.loads((ROOT / "splits.json").read_text())["train"]
    pix = {u: pd.read_parquet(B55 / "pix" / f"pix_{u}.parquet").assign(item_id=lambda d: d.item_id.astype(str)).set_index("item_id")
           for u in train if (B55 / "pix" / f"pix_{u}.parquet").exists()}
    Q = [q for q in json.loads((ROOT / "queries" / "train.json").read_text())["queries"] if q["cat"] == "b"]
    out = [f"TRAIN libraries with pixel stats: {len(pix)} of {len(train)}"]
    T = {}

    def truth_stat(rx, col, exclude=None):
        pos, neg, reqs = [], [], []
        for q in Q:
            w = " ".join(q["turns"])
            if not rx.search(w) or (exclude and exclude.search(w)) or q["user"] not in pix:
                continue
            P = pix[q["user"]]
            gt = set().union(*map(set, q["answers"][-1]))
            pos += list(P.loc[P.index.isin(gt), col]); neg += list(P.loc[~P.index.isin(gt), col])
            reqs.append(f"{q['qid']} ({len(gt)}): {w[:60]}")
        return np.array(pos), np.array(neg), reqs

    pos, neg, reqs = truth_stat(BW, "frac_chroma_gt30", exclude=re.compile("(?i)sepia"))
    T["bw_max_frac_chroma_gt30"] = float(np.percentile(pos, 97.5))
    out += [f"\nBLACK AND WHITE: {len(reqs)} train b requests, {len(pos)} truth photos", *["  " + r for r in reqs],
            f"  frac_chroma_gt30 over truth photos: median {np.median(pos):.4f}, 90th {np.percentile(pos, 90):.4f}, "
            f"97.5th {T['bw_max_frac_chroma_gt30']:.4f}, max {pos.max():.4f}",
            f"  rest of those libraries ({len(neg)}): passing the veto {np.mean(neg <= T['bw_max_frac_chroma_gt30']):.3f}; "
            f"AUC (truth lower) {1 - auc(pos, neg):.3f}"]

    pos, neg, reqs = truth_stat(BLUR, "grad_mean")
    T["blur_max_grad_mean"] = float(np.percentile(pos, 90))
    out += [f"\nBLURRY: {len(reqs)} train b requests, {len(pos)} truth photos", *["  " + r for r in reqs],
            f"  grad_mean over truth photos: median {np.median(pos):.3f}, 90th {T['blur_max_grad_mean']:.3f}",
            f"  rest of those libraries ({len(neg)}): passing the veto {np.mean(neg <= T['blur_max_grad_mean']):.3f}; "
            f"AUC (truth lower) {1 - auc(pos, neg):.3f}"]

    tagged, rest = [], []
    for u, P in pix.items():
        F = load_facts(u)
        q = {str(i): [str(x).lower() for x in ((f or {}).get("quality") or [])] for i, f in F.items()}
        t = P.index.map(lambda i: "overexposed" in q.get(i, []))
        tagged += list(P.loc[t, "frac_clip_hi"]); rest += list(P.loc[~t, "frac_clip_hi"])
    tagged, rest = np.array(tagged), np.array(rest)
    T["over_min_frac_clip_hi"] = float(np.percentile(tagged, 10))
    out += [f"\nOVEREXPOSED (facts quality tag, train libraries): {len(tagged)} tagged of {len(tagged) + len(rest)} photos",
            f"  frac_clip_hi tagged: 10th {T['over_min_frac_clip_hi']:.4f}, median {np.median(tagged):.4f}; untagged median "
            f"{np.median(rest):.4f}; untagged passing the veto {np.mean(rest >= T['over_min_frac_clip_hi']):.3f}; AUC {auc(tagged, rest):.3f}"]
    (B55 / "thresholds.json").write_text(json.dumps(T, indent=1))
    out.append("\nthresholds: " + json.dumps(T))
    (B55 / "fit.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
