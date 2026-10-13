# RESULTS 51: plan stage replayed on CPU from the baseline's raw planner outputs (no model): which dev plans change with
# the new grounding, and which follow-up prompts changed (they need a new model answer on the GPU).
import json, sys
from datetime import date
sys.path.insert(0, "eval/complex_bench")
from plan_system import people_of
from findpics.converse import build_prompt, plan_turn
Q = json.load(open("eval/complex_bench/queries/dev.json"))["queries"]
raw = json.load(open(".cache/cb49c/plan_A4oldq4/raw.json"))
base = json.load(open(".cache/cb49c/plan_A4oldq4/plans.json"))["plans"]
class Miss(Exception): pass
def rec(prompt, max_tokens=1024):
    if prompt not in raw: raise Miss()
    return raw[prompt]
changed, missing = [], []
for q in Q:
    hist, P, people, today = [], None, people_of(q["user"]), date.fromisoformat(q["today"])
    for t, msg in enumerate(q["turns"]):
        try:
            P = plan_turn(msg, rec, history=hist, current=P, people=people, today=today)
        except Miss:
            missing.append((q["qid"], t)); break
        hist.append(msg)
        for a in P.albums:
            if a.person and a.person not in people: a.person = None
            a.with_people = [w for w in a.with_people if w in people]
        new = json.loads(P.model_dump_json()); old = base[q["qid"]]["turns"][t]["plan"]
        strip = lambda p: json.dumps({**p, "notes": ""}, sort_keys=True)
        if strip(new) != strip(old):
            changed.append((q["qid"], t))
            for ao, an in zip(old["albums"], new["albums"]):
                diff = {k: (ao.get(k), an.get(k)) for k in set(ao) | set(an) if ao.get(k) != an.get(k)}
                print("CHANGED", q["qid"], q["cat"], "t", t, "|", msg[:70], "|", json.dumps(diff)[:400])
print("plans changed:", len(changed), "requests:", len({c[0] for c in changed}), "| new model answers needed:", missing)
