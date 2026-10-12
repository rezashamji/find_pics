"""Count how many requests' FINAL plans differ between two plan sets (plans.json from plan_system.py, or a run_system
part*.json folder), ignoring the notes text. Usage: python eval/complex_bench/plan_diff.py <A> <B> [--show=N]"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path


def load(p):
    p = Path(p)
    if p.is_file():
        return {k: [t["plan"] for t in v["turns"]] for k, v in json.loads(p.read_text())["plans"].items() if not v["error"]}
    return {r["qid"]: [t["plan"] for t in r["turns"]] for f in sorted(glob.glob(str(p / "*part*.json")))
            for r in json.loads(open(f).read()) if not r.get("error")}


def key(pl):
    return json.dumps({**pl, "notes": ""}, sort_keys=True)


if __name__ == "__main__":
    A, B = load(sys.argv[1]), load(sys.argv[2])
    show = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--show=")), 0))
    common = sorted(set(A) & set(B))
    diff = [q for q in common if key(A[q][-1]) != key(B[q][-1])]
    cats = Counter(q.split(":")[1][0] for q in diff)
    print(f"{len(diff)}/{len(common)} final plans differ; by category {dict(sorted(cats.items()))}")
    for q in diff[:show]:
        print("==", q)
        for k in sorted(set(A[q][-1]["albums"][0]) if A[q][-1]["albums"] else []):
            pass
        print("  A:", json.dumps([{k: v for k, v in a.items() if v not in (None, [], "any", "all")} for a in A[q][-1]["albums"]])[:600])
        print("  B:", json.dumps([{k: v for k, v in a.items() if v not in (None, [], "any", "all")} for a in B[q][-1]["albums"]])[:600])
