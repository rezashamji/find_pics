# RESULTS 60: dev plan stage replayed on CPU from a run's raw planner outputs (no model) with the code on PYTHONPATH:
# which plans change, and which prompts are new (they need a model answer on the GPU).
# Usage: PYTHONPATH=<snap>/src python eval/complex_bench/r60/plan_replay.py <run plan dir> [split] [out plans.json]
import json, sys
from datetime import date
sys.path.insert(0, "eval/complex_bench")
from plan_system import people_of
from findpics.converse import plan_turn
d = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else "dev"
Q = json.load(open(f"eval/complex_bench/queries/{split}.json"))["queries"]
raw = json.load(open(f"{d}/raw.json"))
base = json.load(open(f"{d}/plans.json"))["plans"]
class Miss(Exception): pass
def rec(prompt, max_tokens=1024):
    if prompt not in raw: raise Miss()
    return raw[prompt]
changed, missing = [], []
res = {}
for q in Q:
    hist, P, people, today = [], None, people_of(q["user"]), date.fromisoformat(q["today"])
    res[q["qid"]] = dict(turns=[], error=None)
    for t, msg in enumerate(q["turns"]):
        try:
            P = plan_turn(msg, rec, history=hist, current=P, people=people, today=today)
        except Miss:
            missing.append((q["qid"], t)); res[q["qid"]]["error"] = "MissingRaw"; break
        hist.append(msg)
        for a in P.albums:
            if a.person and a.person not in people: a.person = None
            a.with_people = [w for w in a.with_people if w in people]
        new = json.loads(P.model_dump_json())
        res[q["qid"]]["turns"].append(dict(message=msg, plan=new))
        if t >= len(base[q["qid"]]["turns"]):
            print("NEWTURN", q["qid"], t); continue
        old = base[q["qid"]]["turns"][t]["plan"]
        strip = lambda p: json.dumps({**p, "notes": ""}, sort_keys=True)
        if strip(new) != strip(old):
            changed.append((q["qid"], t))
            for ao, an in zip(old["albums"], new["albums"]):
                diff = {k: (ao.get(k), an.get(k)) for k in set(ao) | set(an) if ao.get(k) != an.get(k)}
                print("CHANGED", q["qid"], q["cat"], "t", t, "|", msg[:70], "|", json.dumps(diff)[:600])
print("plans changed:", len(changed), "requests:", len({c[0] for c in changed}), "| new model answers needed:", missing)
if len(sys.argv) > 3:
    json.dump(dict(model="replayed from " + d, split=split, plans=res), open(sys.argv[3], "w"))
    print("WROTE", sys.argv[3], len(res))
