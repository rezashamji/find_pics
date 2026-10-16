"""RESULTS 66 table + pre-registered decisions (JOURNAL 10-16 "RESULTS 66 PRE-REGISTRATION"). New truth (queries/ = t63).
  python eval/partial66/table66.py [--out=eval/partial66/table.txt]
Config dirs eval/partial66/replay/<split>/<cov>_N_<policy> are re-scored with score.py; all-judge and the 100% point (65's
100_N_A, every photo has its line) come from eval/partial66/base63/<split>/ (rescore63.py, same truth).
Bootstrap = exact paired, 2,000 resamples over requests, seed 0; pooled = TRAIN + dev.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEVELS = [0, 10, 25, 50, 100]
W = {0: .05, 10: .125, 25: .2, 50: .375, 100: .25}     # trapezoid over 0..1 at these points (100 shared by all policies)
POL = ("V", "P", "PV")
SPLITS = ("train", "dev")


def f1s(d, split=None):
    """final F1 per request of a run dir; with split, (re-)score it first against queries/<split>.json."""
    if split:
        subprocess.run([sys.executable, "eval/complex_bench/score.py", split, d], cwd=ROOT, check=True,
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


def returned(d):
    out = {}
    for f in sorted((ROOT / d).glob("part*.json")):
        for r in json.loads(f.read_text()):
            out[r["qid"]] = json.dumps([t.get("final") and t["final"]["albums"] for t in r.get("turns") or []]) + str(r.get("error"))
    return out


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    F, R, C, ERR = {}, {}, {}, {}
    L = ["RESULTS 66: ALL F1 (final round, RESULTS-63 truth) diff vs all-judge [95%], judge calls / search mean, median"]
    for s in SPLITS:
        R[s] = f1s(f"eval/partial66/base63/{s}/alljudge")
        F[(s, 100)] = f1s(f"eval/partial66/base63/{s}/65_100_N_A")
        C[(s, 100)] = calls(f"eval/partial65/replay/{s}/100_N_A")
        for c in LEVELS[:-1]:
            for p in POL:
                d = f"eval/partial66/replay/{s}/{c}_N_{p}"
                F[(s, c, p)] = f1s(d, s)
                C[(s, c, p)] = calls(d)
                ERR[(s, c, p)] = sum(1 for f in (ROOT / d).glob("part*.json") for r in json.loads(f.read_text()) if r.get("error"))
        # reproduction: V here = RESULTS 65's V (same harness code path) per request
        for c in LEVELS[:-1]:
            a, b = returned(f"eval/partial66/replay/{s}/{c}_N_V"), returned(f"eval/partial65/replay/{s}/{c}_N_V")
            same = sum(a[q] == b.get(q) for q in a)
            old = f1s(f"eval/partial66/base63/{s}/65_{c}_N_V")
            L.append(f"reproduction {s} {c}_N_V: identical returned sets {same}/{len(a)}; ALL F1 {np.mean(list(F[(s, c, 'V')].values())):.4f}"
                     f" vs 65 {np.mean(list(old.values())):.4f}")
    qs = {s: sorted(R[s]) for s in SPLITS}
    for k, v in F.items():
        assert sorted(v) == qs[k[0]], f"{k}: {len(v)} vs {len(qs[k[0]])} (incomplete run)"
    L.append("cov | policy | TRAIN ALL | TRAIN diff vs all-judge | TRAIN calls | dev ALL | dev diff | dev calls | errors T/d")

    def get(s, c, p):
        return F[(s, 100)] if c == 100 else F[(s, c, p)]
    for c in LEVELS:
        for p in (POL if c < 100 else ("-",)):
            row = [str(c), p]
            for s in SPLITS:
                v = get(s, c, p)
                row += [f"{np.mean([v[q] for q in qs[s]]):.4f}", fmt(boot([v[q] - R[s][q] for q in qs[s]])),
                        "{:.0f}, {:.0f}".format(*C[(s, 100) if c == 100 else (s, c, p)])]
            row += [f"{ERR[('train', c, p)]}/{ERR[('dev', c, p)]}" if c < 100 else "-"]
            L.append(" | ".join(row))

    def area(s, p):
        return np.array([sum(W[c] * get(s, c, p)[q] for c in LEVELS) for q in qs[s]])

    def cmp(p1, p2):
        d = {s: area(s, p1) - area(s, p2) for s in SPLITS}
        return boot(np.r_[d["train"], d["dev"]]), d["train"].mean(), d["dev"].mean()

    def passes(t):
        return t[0][1] > 0 and t[1] > 0 and t[2] > 0
    L += ["", "CURVE AREA (weights 0:.05 10:.125 25:.2 50:.375 100:.25), area vs all-judge:"]
    for p in POL:
        a = {s: area(s, p) - np.array([R[s][q] for q in qs[s]]) for s in SPLITS}
        L.append(f"  {p}: TRAIN {fmt(boot(a['train']))}  dev {fmt(boot(a['dev']))}  pooled {fmt(boot(np.r_[a['train'], a['dev']]))}")
    L.append("D1 POLICY: area(X) - area(V); pooled [95%], TRAIN point, dev point")
    ok = {}
    for X in ("P", "PV"):
        t = cmp(X, "V"); ok[X] = passes(t)
        L.append(f"  {X} - V: pooled {fmt(t[0])}; TRAIN {t[1]:+.4f}; dev {t[2]:+.4f} -> {'PASS' if ok[X] else 'no'}")
    t = cmp("PV", "P")
    L.append(f"  PV - P: pooled {fmt(t[0])}; TRAIN {t[1]:+.4f}; dev {t[2]:+.4f}")
    pol = ("PV" if t[0][1] > 0 else "P") if ok["P"] and ok["PV"] else "P" if ok["P"] else "PV" if ok["PV"] else "V"
    L.append(f"  -> POLICY = {pol}")
    L.append("D3 READINESS: lower 95% of ALL diff vs all-judge >= -.03 on both splits (every policy; adopted one decides)")
    for p in POL:
        ready = None
        for c in LEVELS:
            lo = {s: boot([get(s, c, p)[q] - R[s][q] for q in qs[s]])[1] for s in SPLITS}
            good = all(x >= -.03 for x in lo.values())
            L.append(f"  {p} {c}%: TRAIN lower {lo['train']:+.4f}, dev lower {lo['dev']:+.4f} -> {'PASS' if good else 'no'}")
            if good and ready is None:
                ready = c
        L.append(f"  {p} -> first passing coverage = {ready}%{'   <- ADOPTED POLICY' if p == pol else ''}")
    txt = "\n".join(L)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
