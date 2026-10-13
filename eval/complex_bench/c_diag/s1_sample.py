# RESULTS 49b step 1: FP/FN of 20 sampled dev c failures, split by the truth program's own day/place scope.
# Sample = c_diag/sample20.json (random.seed(0); random.sample(23 dev c requests with final F1 < 0.5 in A4old2, 20)).
# Usage (repo root, PYTHONPATH=src): cp eval/complex_bench/c_diag/sample20.json <work dir>/csamp.json;
#   python eval/complex_bench/c_diag/s1_sample.py <work dir>   -> <work dir>/cdiag.json. DEV ONLY.
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "eval/complex_bench")
from common import load_items
import build
SP = Path(sys.argv[1])
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {}
for f in glob.glob("eval/complex_bench/runs/cmp/A4old2_search/part*.json"):
    for r in json.load(open(f)): R[r["qid"]] = r
samp = json.load(open(SP / "csamp.json"))
it = load_items()
out = {}
libs = {}
for qid in samp:
    q, r = Q[qid], R[qid]
    u = q["user"]
    if u not in libs: libs[u] = build.Lib(u, it)
    L = libs[u]
    a = q["program"][0]
    gt = set().union(*map(set, q["answers"][-1]))
    got = set().union(*map(set, r["turns"][-1]["final"]["albums"]))
    meta = L.meta_mask(a); inscope = set(L.ids[meta])
    fp, fn = sorted(got - gt), sorted(gt - got)
    qs = (a.get("need") or []) + (a.get("need_any") or [])
    pkeys = [f"PERSON[{a['person']}]::{x.strip()}" for x in (a.get("person_need") or [])] if a.get("person") else []
    def orc(i):
        return {k[:40]: round(float(L.O.get(k.strip(), {}).get(i, -1)), 2) for k in qs + pkeys}
    pers = L.persons.get(a.get("person") or "", {})
    rows = []
    for kind, S in (("FP", fp), ("FN", fn), ("TP", sorted(got & gt))):
        for i in S:
            k = L.pos[i]
            rows.append(dict(kind=kind, id=i, in_scope=i in inscope, day=str(L.L.t.iloc[k])[:16], place=str(L.L.place.iloc[k])[:60],
                             path=L.L.path.iloc[k], oracle=orc(i), face=(round(pers[i], 2) if i in pers else None)))
    out[qid] = dict(words=" || ".join(q["turns"]), program=a, plan=r["turns"][-1]["plan"]["albums"],
                    n_got=len(got), n_gt=len(gt), tp=len(got & gt), fp_in=sum(x["in_scope"] for x in rows if x["kind"] == "FP"),
                    fp_out=sum(not x["in_scope"] for x in rows if x["kind"] == "FP"), n_fn=len(fn), n_scope=int(meta.sum()), rows=rows)
    print(qid, "|", out[qid]["words"][:70], "| got", len(got), "gt", len(gt), "tp", len(got & gt), "| FP in-scope", out[qid]["fp_in"],
          "out", out[qid]["fp_out"], "| FN", len(fn), "| truth meta scope", int(meta.sum()), "| person", a.get("person"), "persons found", len(pers))
json.dump(out, open(SP / "cdiag.json", "w"), indent=1, default=str)
