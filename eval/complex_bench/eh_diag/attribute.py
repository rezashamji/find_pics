# RESULTS 51: which rule touched each dev request whose final albums changed vs the baseline (49c), and the F1 change.
# Usage: python attribute.py <new run dir> <new scored dir> [<base scored dir>]
import json, glob, sys, re
from collections import defaultdict
new_dir, new_sc = sys.argv[1], sys.argv[2]
base_sc = sys.argv[3] if len(sys.argv) > 3 else "eval/complex_bench/runs/cmp/r49c_A4oldq4_new_truthnew"
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
B = {r["qid"]: r for f in glob.glob("eval/complex_bench/runs/cmp/A4oldq4_search/part*.json") for r in json.load(open(f))}
for f in (".cache/cb49c/replay_new_A4oldq4/part0.json", ".cache/cb49c/c10_A4oldq4/part0.json"):   # the 49c baseline
    B.update({r["qid"]: r for r in json.load(open(f))})
N = {r["qid"]: r for f in glob.glob(f"{new_dir}/part*.json") for r in json.load(open(f))}
SB = {s["qid"]: s for s in json.load(open(f"{base_sc}/scored.json"))}
SN = {s["qid"]: s for s in json.load(open(f"{new_sc}/scored.json"))}
def rules(r, rb):
    out = set()
    for ti, t in enumerate(r["turns"]):
        tb = rb["turns"][ti] if rb and ti < len(rb["turns"]) else None
        if tb and json.dumps({**t["plan"], "notes": ""}, sort_keys=True) != json.dumps({**tb["plan"], "notes": ""}, sort_keys=True):
            out.add("plan(from-moment)")
        for a, tr in zip(t["plan"]["albums"], (t["final"] or {}).get("traces", [])):
            if tr.get("dates_moved_back_years"): out.add("year-back")
            if tr.get("anchor_closest_p") is not None: out.add("closest-moment")
            if tr.get("anchor_found", 0) > tr.get("anchor_used", 0): out.add("one-day")
            if a.get("place") and re.search(r"(?i)\s+(and|or)\s+|&", a["place"]): out.add("place-list")
    return out
rows = []
for qid in sorted(SN):
    if qid not in SB or qid not in N: continue
    r, rb = N[qid], B.get(qid)
    same = rb and not r.get("error") and not rb.get("error") and len(r["turns"]) == len(rb["turns"]) and all(
        (t["final"] or {}).get("albums") == (tb["final"] or {}).get("albums") for t, tb in zip(r["turns"], rb["turns"]))
    d = SN[qid]["final"]["f1"] - SB[qid]["final"]["f1"]
    ru = rules(r, rb)
    if not same or abs(d) > 1e-9 or ru:
        rows.append((qid, Q[qid]["cat"], round(SB[qid]["final"]["f1"], 3), round(SN[qid]["final"]["f1"], 3), round(d, 3),
                     ",".join(sorted(ru)) or "NONE", " || ".join(Q[qid]["turns"])[:110]))
by = defaultdict(list)
for x in rows:
    by[x[5]].append(x)
for k, xs in sorted(by.items()):
    print(f"== {k}: {len(xs)} requests, mean dF1 {sum(x[4] for x in xs) / len(xs):+.3f}")
    for x in sorted(xs, key=lambda x: x[4]):
        print("  ", *x[:5], "|", x[6])
