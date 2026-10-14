"""RESULTS 60: requests whose plan differs in ANY turn (notes ignored) between two plans.json files -> one qid per line.
Usage: python eval/complex_bench/r60/changed_qids.py <old plans.json> <new plans.json> > qids.txt"""
import json
import sys

o, n = (json.load(open(f))["plans"] for f in sys.argv[1:3])
strip = lambda t: json.dumps({**t["plan"], "notes": ""}, sort_keys=True)
for q in sorted(n):
    a, b = o[q]["turns"], n[q]["turns"]
    if len(a) != len(b) or any(strip(x) != strip(y) for x, y in zip(a, b)) or o[q]["error"] != n[q]["error"]:
        print(q)
