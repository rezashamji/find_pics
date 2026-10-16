"""RESULTS 65 table + pre-registered decisions D1-D3 (JOURNAL 10-16 "RESULTS 65 PRE-REGISTRATION").
  python eval/partial65/table.py [--out=eval/partial65/table.txt]
Every config dir is re-scored with eval/complex_bench/score.py; all-judge references copied and re-scored the same way.
Bootstrap = exact paired, 2,000 resamples over requests, seed 0 (as eval/dedup64/table.py); pooled = TRAIN + dev.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = {"train": "eval/facts_scorer/replay/train_alljudge", "dev": "eval/complex_bench/runs/cmp/r60v2_base_search"}
BASE = {"train": .6503, "dev": .6852}     # 61d per-photo baselines (reproduction check: 100_N_A within +-.002)
LEVELS = [0, 10, 25, 50, 75, 100]
W = {0: .05, 10: .125, 25: .2, 50: .25, 75: .25, 100: .125}
SPLITS = ("train", "dev")


def rescore(split, d):
    subprocess.run([sys.executable, "eval/complex_bench/score.py", split, str(d)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return {r["qid"]: r["final"]["f1"] for r in json.loads((ROOT / d / "scored.json").read_text())}


def boot(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi


def calls(d):
    v = []
    for f in sorted((ROOT / d).glob("stats*.json")):
        v += [r["judge"] for r in json.loads(f.read_text()).values()]
    return (np.mean(v), np.median(v)) if v else (np.nan, np.nan)


def cfg(c, o, p):
    """config dir name for a (coverage, order, policy) point; 0% and 100% are shared."""
    if c == 100:
        return "100_N_A"
    if c == 0:
        return "0_N_A" if p == "A" else "0_N_V"
    return f"{c}_{o}_{p}"


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    F, R, C, ERR = {}, {}, {}, {}
    for s in SPLITS:
        rc = ROOT / f"eval/partial65/replay/{s}_alljudge"
        rc.mkdir(parents=True, exist_ok=True)
        for f in (ROOT / REF[s]).glob("part*.json"):
            shutil.copy(f, rc / f.name)
        R[s] = rescore(s, rc.relative_to(ROOT))
        for d in sorted((ROOT / f"eval/partial65/replay/{s}").iterdir()):
            if d.is_dir() and (d / "part0.json").exists():
                F[(s, d.name)] = rescore(s, d.relative_to(ROOT))
                C[(s, d.name)] = calls(d.relative_to(ROOT))
                ERR[(s, d.name)] = sum(1 for f in d.glob("part*.json") for r in json.loads(f.read_text()) if r.get("error"))
    qs = {s: sorted(R[s]) for s in SPLITS}
    for (s, n), v in F.items():
        assert sorted(v) == qs[s], f"{s} {n}: {len(v)} requests vs {len(qs[s])} (incomplete run)"
    L = ["RESULTS 65: ALL F1 (final round) diff vs all-judge [95%], judge calls / search mean, median; errors",
         f"reproduction: 100_N_A TRAIN {np.mean(list(F[('train', '100_N_A')].values())):.4f} (61d .6503), "
         f"dev {np.mean(list(F[('dev', '100_N_A')].values())):.4f} (61d .6852)",
         "cov | order | policy | TRAIN ALL | TRAIN diff vs all-judge | TRAIN calls | dev ALL | dev diff | dev calls | errors T/d"]
    seen = set()
    for c in LEVELS:
        for o in "NRE":
            for p in ("A", "V", "VS"):
                n = cfg(c, o, p)
                if n in seen:
                    continue
                seen.add(n)
                row = [str(c), o if 0 < c < 100 else "-", p if c < 100 else "-"]
                for s in SPLITS:
                    v = F[(s, n)]
                    d = boot([v[q] - R[s][q] for q in qs[s]])
                    row += [f"{np.mean(list(v.values())):.4f}", fmt(d), f"{C[(s, n)][0]:.0f}, {C[(s, n)][1]:.0f}"]
                row += [f"{ERR[('train', n)]}/{ERR[('dev', n)]}"]
                L.append(" | ".join(row))

    def area(s, o, p):
        return np.array([sum(W[c] * F[(s, cfg(c, o, p))][q] for c in LEVELS) for q in qs[s]])

    def cmp(o1, p1, o2, p2):
        d = {s: area(s, o1, p1) - area(s, o2, p2) for s in SPLITS}
        pooled = boot(np.r_[d["train"], d["dev"]])
        return pooled, d["train"].mean(), d["dev"].mean()

    def passes(t):
        return t[0][1] > 0 and t[1] > 0 and t[2] > 0
    L.append("")
    L.append("CURVE AREA (trapezoid over coverage; per request), area vs all-judge:")
    for o in "NRE":
        for p in ("A", "V", "VS"):
            a = {s: area(s, o, p) - np.array([R[s][q] for q in qs[s]]) for s in SPLITS}
            L.append(f"  {o} {p}: TRAIN {fmt(boot(a['train']))}  dev {fmt(boot(a['dev']))}  pooled "
                     f"{fmt(boot(np.r_[a['train'], a['dev']]))}")
    L.append("D1 POLICY (order N): area(X) - area(A); pooled [95%], TRAIN point, dev point")
    ok = {}
    for X in ("V", "VS"):
        t = cmp("N", X, "N", "A"); ok[X] = passes(t)
        L.append(f"  {X} - A: pooled {fmt(t[0])}; TRAIN {t[1]:+.4f}; dev {t[2]:+.4f} -> {'PASS' if ok[X] else 'no'}")
    t = cmp("N", "VS", "N", "V")
    L.append(f"  VS - V: pooled {fmt(t[0])}; TRAIN {t[1]:+.4f}; dev {t[2]:+.4f}")
    pol = ("VS" if t[0][1] > 0 else "V") if ok["V"] and ok["VS"] else "V" if ok["V"] else "VS" if ok["VS"] else "A"
    L.append(f"  -> POLICY = {pol}")
    L.append(f"D2 ORDER (policy {pol}): area(X) - area(N)")
    best, bp = "N", -np.inf
    for X in ("R", "E"):
        t = cmp(X, pol, "N", pol)
        L.append(f"  {X} - N: pooled {fmt(t[0])}; TRAIN {t[1]:+.4f}; dev {t[2]:+.4f} -> {'PASS' if passes(t) else 'no'}")
        if passes(t) and t[0][0] > bp:
            best, bp = X, t[0][0]
    t = cmp("E", pol, "R", pol)
    L.append(f"  (E - R: pooled {fmt(t[0])}; TRAIN {t[1]:+.4f}; dev {t[2]:+.4f})")
    L.append(f"  -> ORDER = {best}")
    L.append(f"D3 READINESS ({best}, {pol}): lower 95% of ALL diff vs all-judge >= -.03 on both splits")
    ready = None
    for c in LEVELS:
        n = cfg(c, best, pol)
        lo = {s: boot([F[(s, n)][q] - R[s][q] for q in qs[s]])[1] for s in SPLITS}
        good = all(x >= -.03 for x in lo.values())
        L.append(f"  {c}%: TRAIN lower {lo['train']:+.4f}, dev lower {lo['dev']:+.4f} -> {'PASS' if good else 'no'}")
        if good and ready is None:
            ready = c
    L.append(f"  -> first passing coverage = {ready}%")
    txt = "\n".join(L); print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
