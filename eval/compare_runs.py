"""Compare everyday-search runs per query: photos returned and overlap with a reference run (agreement, NOT truth;
truth comes from the blind eye audits). Usage: python eval/compare_runs.py <ref_dir> <run_dir> [<run_dir> ...]"""
import glob
import json
import sys

import pandas as pd


def load(d):
    return {(r["user"], r["query"]): r for f in sorted(glob.glob(f"{d}/part*.json")) for r in json.load(open(f))}


ref_dir, runs = sys.argv[1], sys.argv[2:]
R = load(ref_dir)
rows = []
for (u, q), r in R.items():
    row = dict(query=q, ref=len(r["exhaustive"]))
    for d in runs:
        o = load(d).get((u, q)) if False else None
        row[d] = 0
    rows.append(row)
out = {}
for d in runs:
    O = load(d)
    per = {}
    for (u, q), r in R.items():
        if (u, q) not in O:
            continue
        a, b = set(r["exhaustive"]), set(O[(u, q)]["exhaustive"])
        p = per.setdefault(q, dict(ref=0, n=0, both=0))
        p["ref"] += len(a); p["n"] += len(b); p["both"] += len(a & b)
    out[d] = per
qs = sorted(R and {q for (_, q) in R})
tab = pd.DataFrame(index=qs)
for d, per in out.items():
    tab[f"{d.split('/')[-1]} n"] = [per.get(q, {}).get("n", 0) for q in qs]
    tab[f"{d.split('/')[-1]} agree"] = [round(per[q]["both"] / max(per[q]["ref"], 1), 2) if q in per else None for q in qs]
tab.insert(0, "ref n", [sum(len(r["exhaustive"]) for (u, qq), r in R.items() if qq == q) for q in qs])
print(f"reference: {ref_dir} ({len({u for u, _ in R})} libraries)")
print(tab.to_string())
