"""RESULTS 59b report: cascade variants vs the all-judge shipped stack (RESULTS 60 code) on dev - F1 per category
(compare.py), judge calls per search, estimated phone seconds per search.
  python eval/facts_scorer/cascade_report.py <cascade dir> ... [--out=file]
Phone estimate (ESTIMATE): judge call = 60/25 s (25 calls/min); towervec 1.6 MFLOP and facts reader 8.1 GFLOP per
photo-question at an assumed 1 TFLOP/s effective (RESULTS 48's basis: 0.88 TFLOP ~ a second).
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF = "eval/complex_bench/runs/cmp/r60v2_base_search"
PT = "eval/facts_scorer/replay/passthrough_r60"
SPLIT = "dev"
for _a in sys.argv:
    if _a.startswith("--ref="):
        REF = _a.split("=", 1)[1]
    if _a.startswith("--pt="):
        PT = _a.split("=", 1)[1]
    if _a.startswith("--split="):
        SPLIT = _a.split("=", 1)[1]
VARS = ["k0", "f500", "f2000", "f5000", "k500", "kall", "j100", "j300", "r59c", "r59d", "r59d_a", "r59d_b", "r59d_c",
        "aj0", "aj59c", "ajb", "aj59d"]
LABEL = {"k0": "ms scorer only (K=0)", "f500": "K=500 phone-scaled (~6)", "f2000": "K=2,000 phone-scaled (~21)",
         "f5000": "K=5,000 phone-scaled (~51)", "k500": "K=500", "kall": "K=all (= 2,000 = 5,000 here)",
         "j100": "POST HOC: K=all + judge only top 100 of pixel kinds", "j300": "POST HOC: K=all + judge only top 300 of pixel kinds",
         "r59c": "R59c: K=0 + trip route + judge budget max(50, 5%)",
         "r59d": "R59d: R59c + compound trip + OR/AND parts + budget extension + judged exclusions",
         "r59d_a": "R59c + (a) compound trip / tolerant places", "r59d_b": "R59c + (b) OR/AND parts",
         "r59d_c": "R59c + (c) budget extension + judged exclusions", "aj0": "all-judge through the harness",
         "aj59c": "all-judge + R59c trip route", "ajb": "all-judge + (b) parts", "aj59d": "all-judge + (a) + (b)"}


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    L = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{PT}/pairs*.parquet")])
    allj = L.drop_duplicates(["qid", "item_id", "question", "crop"]).groupby("qid").size()
    Q = [q["qid"] for q in json.loads(Path(f"eval/complex_bench/queries/{SPLIT}.json").read_text())["queries"]]
    allj = allj.reindex(Q, fill_value=0)
    lines = [f"All-judge reference ({REF}; calls from {PT}): judge calls per search mean {allj.mean():.0f}, "
             f"median {allj.median():.0f}, total {allj.sum():,} over {len(Q)} searches; est. phone minutes per search "
             f"mean {allj.mean() / 25:.1f}, median {allj.median() / 25:.1f}"]
    for d in dirs:
        lines.append(f"\n=== {d}")
        for v in VARS:
            if not Path(d, v).exists():
                continue
            subprocess.run([sys.executable, "eval/complex_bench/score.py", SPLIT, f"{d}/{v}"], capture_output=True)
        args = [f"judge9b={REF}"] + [f"{v}={d}/{v}" for v in VARS if Path(d, v).exists()]
        cmp = subprocess.run([sys.executable, "eval/complex_bench/compare.py"] + args + [f"--out={d}/compare.txt"],
                             capture_output=True, text=True).stdout
        lines.append(cmp)
        lines.append("variant | judge calls / search: mean, median (share of all-judge) | facts-reader pairs / search | "
                     "towervec pairs / search | est. phone s / search: mean, median (judge share) | judge cache misses")
        for v in VARS:
            S = {}
            for f in glob.glob(f"{d}/{v}/stats*.json"):
                S.update(json.loads(Path(f).read_text()))
            if not S:
                continue
            s = pd.DataFrame.from_dict(S, orient="index").reindex(Q)
            j, ce, tv = s.judge.fillna(0), s.ce.fillna(0), s.tv.fillna(0)
            sec = j * 60 / 25 + tv * 1.6e6 / 1e12 + ce * 8.1e9 / 1e12
            lines.append(f"{LABEL[v]} | {j.mean():.0f}, {j.median():.0f} ({j.sum() / allj.sum():.3f}) | {ce.mean():.0f} | "
                         f"{tv.mean():.0f} | {sec.mean():.0f}, {sec.median():.0f} ({(j * 60 / 25).sum() / max(sec.sum(), 1e-9):.3f}) | "
                         f"{int(s.misses.fillna(0).sum())}")
        S = {}
        for f in glob.glob(f"{d}/k0/stats*.json"):
            S.update(json.loads(Path(f).read_text()))
        R = pd.DataFrame([x["route"] for x in S.values()]).fillna(0).sum()
        lines.append("judge calls routed (pass A, all calls incl. repeats): " +
                     ", ".join(f"{k} {v / R.sum():.3f}" for k, v in R.sort_values(ascending=False).items()))
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
