"""RESULTS 71: choose NJ's ONE global facts-P threshold on TRAIN ONLY (pre-registered: grid .05 steps, maximise ALL F1; tie
rule fixed before scoring: the t closest to .70 = the identity). Reads no dev file.
  python eval/nj71/choose71.py  -> eval/nj71/threshold.txt ("<xx> <TRAIN ALL F1>") + eval/nj71/grid_train.txt"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common71 import ROOT, f1s, truth  # noqa: E402

GRID = [f"{x:02d}" for x in range(5, 100, 5)]
qs = sorted(truth("train"))
rows = []
for t in GRID:
    F = f1s(f"eval/nj71/replay/s1/train/NJ_fp32_T{t}", "train")
    assert sorted(F) == qs, f"T{t}: {len(F)}/{len(qs)} TRAIN requests (incomplete run)"
    rows.append((t, float(np.mean([F[q] for q in qs]))))
best = max(r[1] for r in rows)
tied = [r for r in rows if abs(r[1] - best) < 1e-12]
t_star = min(tied, key=lambda r: abs(int(r[0]) - 70))
L = ["RESULTS 71 NJ threshold grid, TRAIN (641 requests), ALL F1 final round, RESULTS-63 truth:"]
L += [f"  t = .{t} : {m:.4f}{'  <- chosen' if t == t_star[0] else ''}" for t, m in rows]
L.append(f"chosen t = .{t_star[0]} (max TRAIN ALL F1 {t_star[1]:.4f}; {len(tied)} tied at the max)")
(ROOT / "eval/nj71/grid_train.txt").write_text("\n".join(L) + "\n")
(ROOT / "eval/nj71/threshold.txt").write_text(f"{t_star[0]} {t_star[1]:.6f}\n")
print("\n".join(L))
