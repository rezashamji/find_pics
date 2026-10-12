"""Golden cases for the Swift port of converse.ground (+ the unanswerable fallback): raw 9B planner outputs on the
distillation requests (synthetic) -> the grounded plan the Python produces. Owner / people / today are re-derived exactly
as eval/distill_planner_data.py chose them (random.Random(first message))."""
import json, glob, random, re, sys
from datetime import date, timedelta
sys.path.insert(0, "eval")
from distill_planner_data import OWNERS, PEOPLE
from findpics import converse as C

rows = [json.loads(l) for f in sorted(glob.glob("data/public/distill/planner_part*.jsonl")) for l in open(f)]
out, fup = [], []
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
        C.strip_identity_conditions(P, owner)   # as plan_turn does, before the check (M42)
        if C._unanswerable(P, list(people) + [owner], said):
            P = C._drop_unanswerable(P, said, list(people) + [owner])
        G = C.ground(P, r["request"], hist, today, owner, people)
    except Exception as e:
        continue
    out.append(dict(message=r["request"], history=hist, owner=owner, people=people, today=today.isoformat(), raw=raw,
                    grounded=json.loads(G.model_dump_json())))
    # follow-up edits (converse.follow_up_edits, after grounding) against the current plan the prompt showed (10-12)
    cm = re.search(r"Current plan:\n(\{.*?\})\nReturn the WHOLE", r["prompt"], re.S)
    if hist and cm:
        cur = C.Plan.model_validate(json.loads(cm.group(1)))
        F = C.follow_up_edits(C.Plan.model_validate(json.loads(G.model_dump_json())), cur, r["request"])
        fup.append(dict(message=r["request"], current=json.loads(cur.model_dump_json()), grounded=json.loads(G.model_dump_json()),
                        edited=json.loads(F.model_dump_json())))
json.dump(out, open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/ground.json", "w"))
print("GROUND_FIXTURES", len(out))
# + hand-written cases from the complex-bench dev failure kinds (Python tests test_complex_bench_dev_*), so the rules
# that rarely fire on the distillation requests are still compared
fq = "Is there a real parade float anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
narrowed = ("Is there a real sea lion on a dock anywhere in this photo (not a drawing, painting, statue, toy, model or picture "
            "of one) and are there multiple sea lions crowded together (not just one or two)?")
for cur, new, msg in [
        ({"name": "p", "looks": ["a parade float"], "judge_question": fq},
         {"name": "p", "looks": ["a parade float", "a singer"], "judge_question": fq[:-1] + " AND is there a singer anywhere in "
          "this photo (not a drawing, painting, statue, toy, model or picture of one)?"}, "Add the ones with the singer too."),
        ({"name": "h", "judge_question": "Is there a hockey game in this photo?", "filter_question": "Is there a maroon jersey in this photo?"},
         {"name": "h", "judge_question": "Is there a hockey game in this photo?", "filter_question": "Is there a white jersey in this photo?"},
         "actually show me the white ones too"),
        ({"name": "s", "judge_question": "Is there a statue in this photo?"},
         {"name": "s", "judge_question": "Is there a statue in this photo?", "filter_question": "Is there a church in this photo?"},
         "Actually, include the ones with the church too."),
        ({"name": "j", "judge_question": "Is there a jaguar in this photo?"},
         {"name": "j", "judge_question": "Is there a tiger in this photo?", "exclude_question": "Is there a leopard in this photo?"},
         "Actually, add the tigers back, but exclude the ones with leopards."),
        ({"name": "s", "looks": ["a sea lion", "a dock"], "judge_question": narrowed},
         {"name": "s", "looks": ["a sea lion"], "judge_question": narrowed},
         "Actually, show me all the sea lion photos from that day, even the ones in the water."),
        ({"name": "t", "judge_question": "Is there a monster truck in this photo?", "filter_question": "Is the truck in the air?"},
         {"name": "t", "judge_question": "Is there a monster truck in this photo?", "filter_question": "Is the truck in the air?",
          "exclude_question": "Is there a crushed car in this photo?"},
         "Actually, show all of them, but remove the ones with the crushed cars."),
        ({"name": "b", "looks": ["a boat"], "judge_question": "Is there a boat with lights?", "filter_question": "Is there a Santa float?"},
         {"name": "b", "looks": ["a boat"], "judge_question": "Is there a boat with lights?"},
         "Actually, show all the photos from that night."),
        ({"name": "f", "judge_question": "Is there a pink flower in this photo?"},
         {"name": "f", "judge_question": "Is there a pink flower or a yellow flower in this photo?"}, "wait, include the yellow ones too")]:
    c, n = C.Plan.model_validate({"albums": [cur]}), C.Plan.model_validate({"albums": [new]})
    F = C.follow_up_edits(C.Plan.model_validate(json.loads(n.model_dump_json())), c, msg)
    fup.append(dict(message=msg, current=json.loads(c.model_dump_json()), grounded=json.loads(n.model_dump_json()),
                    edited=json.loads(F.model_dump_json())))
json.dump(fup, open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/followup.json", "w"))
print("FOLLOWUP_FIXTURES", len(fup), "changed by the edit rules",
      sum(json.dumps({**f["grounded"], "notes": ""}) != json.dumps({**f["edited"], "notes": ""}) for f in fup))
