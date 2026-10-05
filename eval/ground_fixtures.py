"""Golden cases for the Swift port of converse.ground (+ the unanswerable fallback): raw 9B planner outputs on the
distillation requests (synthetic) -> the grounded plan the Python produces. Owner / people / today are re-derived exactly
as eval/distill_planner_data.py chose them (random.Random(first message))."""
import json, glob, random, re, sys
from datetime import date, timedelta
sys.path.insert(0, "eval")
from distill_planner_data import OWNERS, PEOPLE
from findpics import converse as C

rows = [json.loads(l) for f in sorted(glob.glob("data/public/distill/planner_part*.jsonl")) for l in open(f)]
out = []
for r in rows:
    first = (r.get("history") or [r["request"]])[0]
    rnd = random.Random(first)
    owner = rnd.choice(OWNERS); people = rnd.sample(PEOPLE, rnd.randint(0, 4)) + [owner]
    today = date(2026, 10, 4) - timedelta(days=rnd.randint(0, 900))
    hist = r.get("history") or []
    try:
        m = re.search(r"\{.*\}", r["output"], re.S); raw = json.loads(m.group(0))
        for al in raw.get("albums", []):
            an = al.get("anchor") if isinstance(al, dict) else None
            if isinstance(an, dict) and not an.get("judge_question"):
                lk = [x for x in an.get("looks") or [] if x]
                al["anchor"] = dict(an, judge_question=f"Does this photo show {lk[0]}?") if lk else None
        P = C.Plan.model_validate(raw)
        said = " \n ".join(hist + [r["request"]])
        if C._unanswerable(P, list(people) + [owner], said):
            P = C._drop_unanswerable(P, said, list(people) + [owner])
        G = C.ground(P, r["request"], hist, today, owner, people)
    except Exception as e:
        continue
    out.append(dict(message=r["request"], history=hist, owner=owner, people=people, today=today.isoformat(), raw=raw,
                    grounded=json.loads(G.model_dump_json())))
json.dump(out, open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/ground.json", "w"))
print("GROUND_FIXTURES", len(out))
