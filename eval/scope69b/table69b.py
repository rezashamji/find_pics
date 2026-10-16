"""RESULTS 69b table + pre-registered decision (JOURNAL 10-16 "RESULTS 69b PRE-REGISTRATION").
  python eval/scope69b/table69b.py [--out=eval/scope69b/table.txt]
DRY fp32 at coverage 0/10/25/50% (eval/scope69b/replay/<split>/DRY_fp32_<cov>) + 100% = RESULTS 69 DRY_fp32; vs all-judge
(eval/scope69/base/<split>/alljudge) and vs the measured V at the same coverage (RESULTS 66 eval/partial66/replay/<split>/
<cov>_N_V, AJ bf16; 100% = RESULTS 65 100_N_A). Scored with score.py on queries/ (RESULTS 63 truth); paired bootstrap 2,000,
seed 0. Judge calls per search = pass A + pass B (DRY pass A asserted 0; measured: pass B only, as counted there).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scope69"))
from table69 import ROOT, SPLITS, arg, boot, f1s, fmt, stats  # noqa: E402

COVS = (0, 10, 25, 50, 100)


def main():
    F, C, ERR = {}, {}, {}
    for s in SPLITS:
        F[(s, "aj")] = f1s(f"eval/scope69/base/{s}/alljudge", s)
        for c in COVS:
            d = f"eval/scope69/replay/{s}/DRY_fp32" if c == 100 else f"eval/scope69b/replay/{s}/DRY_fp32_{c}"
            F[(s, c)] = f1s(d, s)
            st = stats(d)
            assert all(v.get("judgeA", 0) == 0 for v in st.values()), f"{d}: DRY pass A judged"
            C[(s, c)] = {q: v["judge"] + v.get("judgeA", 0) for q, v in st.items()}
            ERR[(s, c)] = sum(1 for f in (ROOT / d).glob("part*.json") for r in json.loads(f.read_text()) if r.get("error"))
            src = ROOT / (f"eval/partial65/replay/{s}/100_N_A" if c == 100 else f"eval/partial66/replay/{s}/{c}_N_V")
            dst = ROOT / f"eval/scope69b/base/{s}/meas_{c}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*part*.json"):
                shutil.copy(f, dst / f.name)
            F[(s, f"m{c}")] = f1s(dst.relative_to(ROOT), s)
            C[(s, f"m{c}")] = {q: v["judge"] for q, v in stats(src.relative_to(ROOT)).items()}
    qs = {s: sorted(F[(s, "aj")]) for s in SPLITS}
    for k, v in F.items():
        assert sorted(v) == qs[k[0]], f"{k}: {len(v)} vs {len(qs[k[0]])} (incomplete run)"

    def diff(s, a, b):
        return boot([F[(s, a)][q] - F[(s, b)][q] for q in qs[s]])

    def calls(s, c):
        v = np.array([C[(s, c)][q] for q in qs[s]], float)
        return f"{v.mean():.0f} / {np.median(v):.0f} / {np.percentile(v, 90):.0f}"
    L = ["RESULTS 69b: DRY fp32 (the phone) partial-coverage curve, newest-first, policy V; ALL F1 final round, RESULTS-63 truth",
         f"all-judge ALL F1: TRAIN {np.mean(list(F[('train', 'aj')].values())):.4f}, dev {np.mean(list(F[('dev', 'aj')].values())):.4f}",
         "", "coverage | split | ALL F1 | DRY fp32 vs all-judge | measured V (AJ bf16) vs all-judge | DRY fp32 - measured V | "
         "judge calls DRY mean / median / p90 | measured calls | errors"]
    passes = {}
    for c in COVS:
        lows = []
        for s in SPLITS:
            t = diff(s, c, "aj"); lows.append(t[1])
            L.append(f"{c}% | {s} | {np.mean([F[(s, c)][q] for q in qs[s]]):.4f} | {fmt(t)} | {fmt(diff(s, f'm{c}', 'aj'))} | "
                     f"{fmt(diff(s, c, f'm{c}'))} | {calls(s, c)} | {calls(s, f'm{c}')} | {ERR[(s, c)]}")
        passes[c] = min(lows) >= -.03
        L.append(f"   {c}%: lowers TRAIN {lows[0]:+.4f}, dev {lows[1]:+.4f} -> {'PASS' if passes[c] else 'no'}")
    first = next((c for c in COVS if passes[c]), None)
    L += ["", f"READINESS POINT (first passing coverage): {first}%",
          ("-> RESULTS 65 conclusion 'ready from 25%' STANDS for the phone" if first is not None and first <= 25 else
           f"-> readiness point MOVES to {first}%")]
    txt = "\n".join(L)
    print(txt)
    if arg("out"):
        Path(arg("out")).write_text(txt + "\n")


if __name__ == "__main__":
    main()
