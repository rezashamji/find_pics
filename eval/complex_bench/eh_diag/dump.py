# RESULTS 51 step 1: compact dump of dev e/h failures (F1 < 0.5): words, truth program, plans, traces, counts.
import json, glob, sys
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
S = {s["qid"]: s for s in json.load(open("eval/complex_bench/runs/cmp/r49c_A4oldq4_new_truthnew/scored.json"))}
R = {r["qid"]: r for f in glob.glob("eval/complex_bench/runs/cmp/A4oldq4_search/part*.json") for r in json.load(open(f))}
cats = sys.argv[1] if len(sys.argv) > 1 else "eh"
def compact(a):
    keep = {k: v for k, v in a.items() if v not in (None, [], "any", "all") and k != "name"}
    return json.dumps(keep)
for qid, s in sorted(S.items(), key=lambda x: x[1]["final"]["f1"]):
    if s["cat"] not in cats or s["final"]["f1"] >= 0.5:
        continue
    q, r = Q[qid], R[qid]
    print("=" * 100)
    print(qid, s["cat"], "F1 %.2f P %.2f R %.2f got %d gt %d today %s" % (s["final"]["f1"], s["final"]["p"], s["final"]["r"], s["final"]["n_got"], s["final"]["n_gt"], q["today"]))
    print("  why:", q.get("why"), "| stripped:", q.get("stripped"), "| verified", q.get("verified"), "cut", q.get("cut"))
    progs = q["program"] if q["cat"] == "h" else [q["program"]]
    for ti, msg in enumerate(q["turns"]):
        print(f"  T{ti+1}: {msg}")
        print(f"     truth sizes {[len(a) for a in q['answers'][ti]]}")
        print(f"     program: {json.dumps(progs[ti] if ti < len(progs) else None)[:600]}")
        if ti < len(r["turns"]):
            t = r["turns"][ti]
            for a in t["plan"]["albums"]:
                print(f"     plan: {compact(a)[:700]}")
            fin = t["final"]
            if fin:
                gt = set().union(*map(set, q["answers"][ti])) if q["answers"][ti] else set()
                got = [set(x) for x in fin["albums"]]
                print(f"     got {[len(x) for x in got]} tp {[len(x & gt) for x in got]} traces {json.dumps(fin['traces'])[:400]}")
    if r.get("error"):
        print("  ERROR", r["error"])
