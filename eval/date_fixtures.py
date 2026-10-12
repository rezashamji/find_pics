"""Refresh the Swift golden cases for converse.resolve_relative / resolve_time_of_day (Fixtures/dates.json) after a
Python change: every stored (phrase, today) is recomputed with the current Python, NEW phrases are added for every
stored "today", and the script prints how many stored answers changed (each change must be intended).
Usage: python eval/date_fixtures.py"""
import json
import sys
from datetime import date

sys.path.insert(0, "src")
from findpics.converse import resolve_relative, resolve_time_of_day

F = "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/dates.json"
NEW = ["November 27th", "nov 27", "on the 27th of november", "the 27th of November 2012", "november 27 2012", "july 4th",
       "february 29", "feb 30", "march 1st", "december 31st", "the 1st of january", "may 2", "october 6th", "august 20th"]
d = json.load(open(F))
changed = []
for c in d["relative"]:
    w = resolve_relative(c["phrase"], date.fromisoformat(c["today"]))
    w = list(w) if w else None
    if w != c["want"]:
        changed.append((c["phrase"], c["today"], c["want"], w))
    c["want"] = w
tod_changed = sum((resolve_time_of_day(c["text"]) != c["want"]) for c in d["time_of_day"])
todays = sorted({c["today"] for c in d["relative"]})
have = {(c["phrase"], c["today"]) for c in d["relative"]}
added = 0
for p in NEW:
    for t in todays:
        if (p, t) not in have:
            w = resolve_relative(p, date.fromisoformat(t))
            d["relative"].append(dict(phrase=p, today=t, want=list(w) if w else None)); added += 1
json.dump(d, open(F, "w"), indent=1)
print(f"DATE_FIXTURES relative {len(d['relative'])} (added {added}); stored answers changed: {len(changed)}; "
      f"time-of-day changed: {tod_changed}")
for x in changed:
    print("  CHANGED", x)
