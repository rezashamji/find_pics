"""Which grounded plans changed between two ground.json fixtures (notes ignored). Usage: fixture_diff.py OLD NEW"""
import json, sys
o, n = (json.load(open(f)) for f in sys.argv[1:3])
key = lambda c: (c["message"], tuple(c["history"]), json.dumps(c["raw"], sort_keys=True))
strip = lambda g: json.dumps({k: v for k, v in g.items() if k != "notes"}, sort_keys=True)
ko = {key(c): c for c in o}
ch = 0
for c in n:
    oc = ko.get(key(c))
    if oc is None:
        print("NEWCASE", c["message"][:80]); continue
    if strip(oc["grounded"]) != strip(c["grounded"]):
        ch += 1
        for ao, an in zip(oc["grounded"]["albums"], c["grounded"]["albums"]):
            if ao != an:
                print("CHANGED|", c["message"][:90], "| person", an.get("person"), "| OLD", ao.get("judge_question"),
                      ao.get("looks"), "| NEW", an.get("judge_question"), an.get("looks"))
print("FIXTURE_CHANGED", ch, "of", len(n), "old", len(o))
