"""RESULTS 57: aggregate plan shapes of dev plan files (final turn), per category: share of albums with a filter, an
anchor, an exclusion, a place, dates; and how many requests' final plans differ between the first file and the others.
Usage: python eval/complex_bench/b57/plan_shapes.py <name=plans.json> <name=plans.json> ...
"""
import json
import sys
from collections import defaultdict

runs = [a.split("=", 1) for a in sys.argv[1:]]
R = [(n, json.load(open(f))["plans"]) for n, f in runs]
print("final-turn albums: share with filter / anchor / exclude / place / dates; errors")
for n, P in R:
    by = defaultdict(list)
    err = 0
    for q, v in P.items():
        if v.get("error") or not v["turns"]:
            err += 1
            continue
        by[q.split(":")[1][0]] += v["turns"][-1]["plan"]["albums"]
    cells = []
    for c in sorted(by):
        A = by[c]
        f = lambda k: sum(bool(a.get(k)) for a in A) / len(A)
        dt = sum(bool(a.get("date_from") or a.get("date_to")) for a in A) / len(A)
        cells.append(f"{c} {f('filter_question'):.2f}/{f('anchor'):.2f}/{f('exclude_question'):.2f}/{f('place'):.2f}/{dt:.2f}")
    print(f"{n:6s} errors {err}  " + "  ".join(cells))
n0, P0 = R[0]
for n, P in R[1:]:
    diff = defaultdict(int)
    for q in P0:
        a = json.dumps(P0[q]["turns"][-1]["plan"] if P0[q]["turns"] else None, sort_keys=True)
        b = json.dumps(P.get(q, {}).get("turns", [{}])[-1].get("plan") if P.get(q, {}).get("turns") else None, sort_keys=True)
        diff[q.split(":")[1][0]] += a != b
    print(f"final plans differing {n} vs {n0}: " + " ".join(f"{c} {diff[c]}" for c in sorted(diff)) + f"  total {sum(diff.values())}/{len(P0)}")
