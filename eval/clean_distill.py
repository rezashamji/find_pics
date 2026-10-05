"""Clean the planner distillation set: drop targets that fail the planner's own checks (the 9B's last failed retry was
recorded for 145 requests); a retry prompt ("Previous output was invalid ...") is restored to the original prompt so
the corrected answer is learned for the first try. Prints 20 random training targets for an eye check."""
import json, glob, random, re, sys
sys.path.insert(0, "eval")
from distill_planner_data import OWNERS, PEOPLE
from findpics import converse as C
rows = [json.loads(l) for f in sorted(glob.glob("data/public/distill/planner_part*.jsonl")) for l in open(f)]
keep, dropped, fixed = [], 0, 0
for r in rows:
    first = (r.get("history") or [r["request"]])[0]; rnd = random.Random(first)
    owner = rnd.choice(OWNERS); people = rnd.sample(PEOPLE, rnd.randint(0, 4)) + [owner]
    said = " \n ".join((r.get("history") or []) + [r["request"]])
    try:
        P = C.Plan.model_validate(json.loads(re.search(r"\{.*\}", r["output"], re.S).group(0)))
        if not P.albums or C._unanswerable(P, people + [owner], said):
            dropped += 1; continue
    except Exception:
        dropped += 1; continue
    i = r["prompt"].find("\n(Previous output was invalid:")
    if i >= 0:
        r = dict(r, prompt=r["prompt"][:i]); fixed += 1
    keep.append(r)
with open("data/public/distill/planner_clean.jsonl", "w") as fh:
    for r in keep: fh.write(json.dumps(r) + "\n")
tr = [r for r in keep if r["split"] == "train"]
print(f"kept {len(keep)} (train {len(tr)}, test {len(keep) - len(tr)}); dropped {dropped}; retry prompts restored {fixed}")
rnd = random.Random(7)
for r in rnd.sample(tr, 20):
    P = json.loads(re.search(r"\{.*\}", r["output"], re.S).group(0))
    al = [(a.get("name"), a.get("person"), a.get("judge_question"), a.get("time_phrase"), a.get("media"), a.get("filter_question"), a.get("exclude_question")) for a in P.get("albums", [])]
    print("\n>>", (" / ".join(r.get("history", [])) + " => " if r.get("history") else "") + r["request"][:150]); print("  ", al)
