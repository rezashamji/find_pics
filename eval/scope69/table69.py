"""RESULTS 69 table + pre-registered decisions (JOURNAL 10-16 "RESULTS 69 PRE-REGISTRATION").
  python eval/scope69/table69.py [--root=eval/scope69/replay] [--out=eval/scope69/table.txt]
Runs <root>/<split>/<cfg> are scored with score.py against queries/<split>.json (RESULTS 63 truth); all-judge and the
measured R59d point (65's 100_N_A) from eval/partial66/base63/<split>/ (re-scored here on the same truth).
Bootstrap = paired, 2,000 resamples over requests, seed 0. Judge calls per search = pass A + pass B distinct
(photo, leaf question, crop) judged (AJ: pass B only, as every measured number counted; its pass A is the all-judge cheat).
"""
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "dev")
CFGS = ("AJ_bf16", "DRY_bf16", "ON_bf16", "DRY_fp32", "ON_fp32")


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def f1s(d, split):
    subprocess.run([sys.executable, "eval/complex_bench/score.py", split, str(d)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return {r["qid"]: r["final"]["f1"] for r in json.loads((ROOT / d / "scored.json").read_text())}


def boot(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def stats(d):
    out = {}
    for f in sorted((ROOT / d).glob("stats*.json")):
        out.update(json.loads(f.read_text()))
    return out


def returned(d):
    out = {}
    for f in sorted((ROOT / d).glob("part*.json")):
        for r in json.loads(f.read_text()):
            out[r["qid"]] = json.dumps([t.get("final") and t["final"]["albums"] for t in r.get("turns") or []]) + str(bool(r.get("error")))
    return out


def judged_sets(rec_dir, cfg):
    out = {}
    for f in sorted((ROOT / rec_dir / cfg).glob("rec*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            out[r["qid"]] = {tuple(x) for x in r["judged"]}     # requeued shards re-append: the last line wins
    return out


def main():
    root = Path(arg("root", "eval/scope69/replay"))
    recroot = Path(arg("rec", "data/public/scope69/record"))
    out_f = arg("out")
    L = ["RESULTS 69: ALL F1 (final round, RESULTS-63 truth) diffs [95%]; judge calls per search mean / median / p90"]
    F, C, ERR = {}, {}, {}
    for s in SPLITS:
        for n in ("alljudge", "65_100_N_A"):
            src = ROOT / "eval/partial66/base63" / s / n
            dst = ROOT / "eval/scope69/base" / s / n
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*part*.json"):
                shutil.copy(f, dst / f.name)
            F[(s, n)] = f1s(dst.relative_to(ROOT), s)
        st65 = stats(f"eval/partial65/replay/{s}/100_N_A")
        C[(s, "65_100_N_A")] = {q: v["judge"] for q, v in st65.items()}
        for c in CFGS:
            d = root / s / c
            if not (ROOT / d).exists():
                continue
            F[(s, c)] = f1s(d, s)
            st = stats(d)
            C[(s, c)] = {q: v["judge"] + (v.get("judgeA", 0) if not c.startswith("AJ") else 0) for q, v in st.items()}
            ERR[(s, c)] = sum(1 for f in (ROOT / d).glob("part*.json") for r in json.loads(f.read_text()) if r.get("error"))
            assert not c.startswith("DRY") or all(v.get("judgeA", 0) == 0 for v in st.values()), "DRY pass A judged"
        if (s, "AJ_bf16") in F:
            a, b = returned(root / s / "AJ_bf16"), returned(f"eval/partial65/replay/{s}/100_N_A")
            same = sum(a[q] == b.get(q) for q in a)
            L.append(f"CONTROL {s}: AJ_bf16 returned sets identical to RESULTS 65 100_N_A {same}/{len(a)} "
                     f"({'OK' if same >= .98 * len(a) else 'FAIL: < 98%'}); ALL F1 {np.mean(list(F[(s, 'AJ_bf16')].values())):.4f} "
                     f"vs 65 {np.mean(list(F[(s, '65_100_N_A')].values())):.4f}")
    qs = {s: sorted(F[(s, "alljudge")]) for s in SPLITS}
    for k, v in F.items():
        assert sorted(v) == qs[k[0]], f"{k}: {len(v)} vs {len(qs[k[0]])} (incomplete run)"

    def diff(s, a, b):
        return boot([F[(s, a)][q] - F[(s, b)][q] for q in qs[s]])

    def calls(s, c):
        v = np.array([C[(s, c)][q] for q in qs[s] if q in C[(s, c)]], float)
        return f"{v.mean():.0f} / {np.median(v):.0f} / {np.percentile(v, 90):.0f}"
    L.append("")
    L.append("config | split | ALL F1 | vs all-judge | vs measured R59d (65 100_N_A) | vs AJ control | judge calls mean / median / p90 | errors")
    rows = ["65_100_N_A"] + [c for c in CFGS if ("train", c) in F]
    for c in rows:
        for s in SPLITS:
            m = np.mean([F[(s, c)][q] for q in qs[s]])
            L.append(f"{c} | {s} | {m:.4f} | {fmt(diff(s, c, 'alljudge'))} | "
                     f"{fmt(diff(s, c, '65_100_N_A')) if c != '65_100_N_A' else '-'} | "
                     f"{fmt(diff(s, c, 'AJ_bf16')) if ('train', 'AJ_bf16') in F and c not in ('65_100_N_A', 'AJ_bf16') else '-'} | "
                     f"{calls(s, c)} | {ERR.get((s, c), '-')}")
    L.append(f"all-judge ALL F1: TRAIN {np.mean(list(F[('train', 'alljudge')].values())):.4f}, dev {np.mean(list(F[('dev', 'alljudge')].values())):.4f}")
    # pre-registered decision
    L += ["", "DECISION (pre-registered): claim stands iff the adopted variant's lower 95% vs all-judge >= -.03 on both "
          "splits at bf16 AND fp32"]
    ok = {}
    for v in ("DRY", "ON"):
        if ("train", f"{v}_bf16") not in F:
            continue
        lows = {(s, p): diff(s, f"{v}_{p}", "alljudge")[1] for s in SPLITS for p in ("bf16", "fp32") if (s, f"{v}_{p}") in F}
        ok[v] = all(x >= -.03 for x in lows.values()) and len(lows) == 4
        p90 = np.percentile([C[(s, f"{v}_bf16")][q] for s in SPLITS for q in qs[s]], 90)
        L.append(f"  {v}: lowers " + ", ".join(f"{s} {p} {x:+.4f}" for (s, p), x in lows.items()) +
                 f" -> {'PASS' if ok[v] else 'no'}; pooled p90 judge calls (bf16) {p90:.0f}")
        ok[v + "_p90"] = p90
    if ok.get("DRY") and ok.get("ON"):
        adopt = "ON" if ok["ON_p90"] < .95 * ok["DRY_p90"] else "DRY"
    else:
        adopt = "DRY" if ok.get("DRY") else "ON" if ok.get("ON") else None
    L.append(f"  -> ADOPTED phone variant: {adopt or 'NONE passes'}")
    for v in ("DRY", "ON"):
        if ("train", f"{v}_bf16") not in F:
            continue
        for s in SPLITS:
            t = diff(s, f"{v}_bf16", "65_100_N_A")
            if t[1] < 0:
                L.append(f"  {v} {s}: (variant - measured) {fmt(t)} lower < 0 -> measured R59d optimistic by {-t[0]:.4f} on {s}")
    # fp32 vs bf16
    L += ["", "fp32 - bf16 (same scope source): ALL F1 [95%] | requests whose judged set differs | mean extra judge calls"]
    for v in ("DRY", "ON"):
        for s in SPLITS:
            if (s, f"{v}_fp32") not in F:
                continue
            J1, J2 = judged_sets(recroot / s, f"{v}_bf16"), judged_sets(recroot / s, f"{v}_fp32")
            nd = sum(J1.get(q) != J2.get(q) for q in qs[s])
            ex = np.mean([C[(s, f"{v}_fp32")][q] - C[(s, f"{v}_bf16")][q] for q in qs[s]])
            L.append(f"  {v} {s}: {fmt(diff(s, f'{v}_fp32', f'{v}_bf16'))} | {nd}/{len(qs[s])} | {ex:+.1f}")
    txt = "\n".join(L)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
