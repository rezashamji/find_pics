"""RESULTS 57: requests whose final F1 moved by > 0.05 between two scored runs, with both final plans (compact).
Usage: python eval/complex_bench/b57/per_request.py <ref run dir> <ref plans.json> <new run dir> <new plans.json> [cats]"""
import json
import sys

ra, pa, rb, pb = sys.argv[1:5]
cats = sys.argv[5] if len(sys.argv) > 5 else "abcdefgh"
A = {r["qid"]: r for r in json.load(open(f"{ra}/scored.json"))}
B = {r["qid"]: r for r in json.load(open(f"{rb}/scored.json"))}
PA, PB = json.load(open(pa))["plans"], json.load(open(pb))["plans"]
keep = ("person", "judge_question", "filter_question", "exclude_question", "anchor", "window", "place", "date_from", "date_to")


def short(P):
    out = []
    for a in (P["turns"][-1]["plan"]["albums"] if P["turns"] else []):
        g = {k: a[k] for k in keep if a.get(k)}
        if isinstance(g.get("anchor"), dict):
            g["anchor"] = g["anchor"].get("judge_question")
        out.append(g)
    return json.dumps(out, ensure_ascii=False)


rows = sorted(((B[q]["final"]["f1"] - A[q]["final"]["f1"], q) for q in A if q in B and q.split(":")[1][0] in cats))
for d, q in rows:
    if abs(d) <= 0.05:
        continue
    print(f"{d:+.2f} {q} {A[q]['final']['f1']:.2f} -> {B[q]['final']['f1']:.2f} | {A[q]['words'][:150]}")
    print(f"     ref: {short(PA[q])[:400]}")
    print(f"     new: {short(PB[q])[:400]}")
