"""RESULTS 60: a full dev run = the reference run's records for requests whose plans did not change + the new search's
records for the changed ones (same plans + same judge cache -> same records: RESULTS 59's passthrough replay reproduced
r57_base_search 351/351). Usage: python eval/complex_bench/r60/merge.py <ref run dir> <changed run dir> <out dir>
[qids.txt: take only these requests from the changed run]"""
import glob
import json
import sys
from pathlib import Path

ref, new, out = sys.argv[1:4]
R = [r for f in sorted(glob.glob(f"{ref}/part*.json")) for r in json.load(open(f))]
N = {r["qid"]: r for f in sorted(glob.glob(f"{new}/part*.json")) for r in json.load(open(f))}
if len(sys.argv) > 4:
    keep = set(open(sys.argv[4]).read().split())
    N = {q: r for q, r in N.items() if q in keep}
M = [N.get(r["qid"], r) for r in R]
Path(out).mkdir(parents=True, exist_ok=True)
json.dump(M, open(f"{out}/part0.json", "w"))
print("MERGED", len(M), "records;", sum(r["qid"] in N for r in R), "from", new, "| changed run had", len(N))
