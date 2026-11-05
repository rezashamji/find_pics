"""M82: can a statistic computed ON THE PHONE (300 photos, food only: 28 judge-yes) tell a correct facts chain from a broken
one? Simulated on dev5k: 2,000 draws (numpy default_rng(0)) of 28 judge-yes + 272 judge-no photos (4B judge, P >= .7),
without replacement; per draw the AUC of each facts-P variant (score.py --ablate):
  full                  = the product facts P (f, q, v, t)
  shuffled              = a BROKEN chain: every photo carries another photo's facts vector
  facts-only            = the same head with the image vector set to 0 (ranking depends on f only)
  facts-only, shuffled  = the facts-only score of the broken chain
  python eval/device_anchor82/device_stat.py --tag=dev5k [--q=food] [--pos=28] [--n=300]
  -> eval/device_anchor82/device_stat_<tag>.json / .txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/device_anchor82")
from score import auc_w  # noqa: E402

D = Path("data/public/device_anchor82")


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def main():
    tag, q = arg("tag", "dev5k"), arg("q", "food")
    npos, n, B = int(arg("pos", "28")), int(arg("n", "300")), 2000
    pp = pd.read_parquet(D / f"per_photo_{tag}.parquet")
    out, lines = {}, []
    for w in ("trained_vllm61d", "base_vllm"):
        V = {"full": w, "shuffled": f"abl_shuffle_{w}", "facts-only": f"abl_zero_v_{w}",
             "facts-only, shuffled": f"abl_zero_v_shuffle_{w}"}
        T = {k: pp[pp.source == s].set_index("key")[f"pf_{q}"] for k, s in V.items() if (pp.source == s).any()}
        if len(T) < 4:
            continue
        y_all = pp[pp.source == w].set_index("key")[f"j4_{q}"] >= 0.7
        keys = y_all.index[y_all.index.isin(T["shuffled"].index)]
        y_all = y_all.loc[keys]
        pos, neg = keys[y_all.values], keys[~y_all.values]
        rng = np.random.default_rng(0)
        res = {k: [] for k in V}
        for _ in range(B):
            ks = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)])
            y = np.r_[np.ones(npos, bool), np.zeros(n - npos, bool)]
            for k in V:
                res[k].append(float(auc_w(y, T[k].loc[ks].to_numpy(np.float64), np.ones((1, n)))[0]))
        summ = {k: dict(median=float(np.median(v)), p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)),
                        p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95))) for k, v in res.items()}
        # best single cut separating correct from broken for the two pairs (equal error rate)
        sep = {}
        for good, bad in (("full", "shuffled"), ("facts-only", "facts-only, shuffled")):
            g, b = np.array(res[good]), np.array(res[bad])
            cuts = np.linspace(0.3, 1.0, 701)
            err = [max((g < c).mean(), (b >= c).mean()) for c in cuts]
            i = int(np.argmin(err))
            sep[f"{good} vs {bad}"] = dict(cut=float(cuts[i]), correct_below_cut=float((g < cuts[i]).mean()),
                                         broken_at_or_above_cut=float((b >= cuts[i]).mean()))
        out[w] = dict(pool_pos=len(pos), pool_neg=len(neg), summary=summ, separation=sep)
        lines.append(f"{w}: {q}, draws of {npos} judge-yes + {n - npos} judge-no from {len(pos)} / {len(neg)} (dev5k, 4B judge)")
        for k, sm in summ.items():
            lines.append(f"  {k:22s} AUC median {sm['median']:.3f}  95% of draws [{sm['p2_5']:.3f}, {sm['p97_5']:.3f}]")
        for k, sp in sep.items():
            lines.append(f"  cut {sp['cut']:.3f} for {k}: correct chain below it in {sp['correct_below_cut']:.1%} of draws, "
                         f"broken chain at/above it in {sp['broken_at_or_above_cut']:.1%}")
    (Path("eval/device_anchor82") / f"device_stat_{tag}.json").write_text(json.dumps(out, indent=1))
    (Path("eval/device_anchor82") / f"device_stat_{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
