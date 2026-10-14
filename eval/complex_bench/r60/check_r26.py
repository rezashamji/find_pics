"""RESULTS 60 check (2): the RESULTS 26 searches where the real-X rewrite was measured must get byte-identical judge
questions under the new code (then the judge's answers, and so the eye-labelled drops, are identical).
Sources: eval/everyday/all.json (96 plans: 8 DISBench users x 12 everyday queries, the eye-labelled audit set of
eval/everyday/eye_labels.json), eval_real_subject.py's WITH queries and Open Images concepts, eval_real_misses.py's
concepts. all.json holds the plans of the 10-05 run BEFORE fix 1 (plain "Is there a dog anywhere in this photo?"); each
question (any real-X rewrite undone) is grounded by the code on PYTHONPATH. Usage: PYTHONPATH=<snap>/src python eval/complex_bench/r60/check_r26.py > out.txt"""
import json
import re
from datetime import date

from findpics import converse as C

T = date(2026, 10, 5)
TAIL = re.compile(r"^Is there (a real |real )(.+?) anywhere in this photo \(not a drawing, painting, statue, toy, model or "
                  r"picture of one\)\?$")


def raw_of(q):
    m = TAIL.match(q or "")
    if not m:
        return q
    return f"Is there {'a ' if m.group(1) == 'a real ' else ''}{m.group(2)} in this photo?"


cases = []
for r in json.load(open("eval/everyday/all.json")):
    for a in r["plan"]["albums"]:
        cases.append((r["query"], raw_of(a.get("judge_question")), a.get("judge_question")))
OI = ["Dog", "Cat", "Bicycle", "Horse", "Guitar", "Cake", "Pizza", "Coffee cup", "Wine glass", "Sunglasses"]
for c in OI + ["dog", "car", "bicycle", "boat"]:
    cases.append((f"photos with a {c.lower()}", f"Is there a {c.lower()} in this photo?", None))
seen, n_real = set(), 0
for msg, raw, _ in cases:
    if (msg, raw) in seen:
        continue
    seen.add((msg, raw))
    P = C.Plan.model_validate({"albums": [{"name": msg, "judge_question": raw}]})
    G = C.ground(P, msg, [], T, "me", [])
    q = G.albums[0].judge_question
    real = bool(TAIL.match(q or ""))
    n_real += real
    print("real" if real else "plain", "|", msg, "|", raw, "->", q)
print(f"R26_CHECK {len(seen)} distinct (request, question) pairs; grounded with the real-X tail {n_real} "
      f"(compare the output of the old and the new code: it must be identical)")
