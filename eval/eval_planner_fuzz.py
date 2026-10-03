"""Planner robustness on 100 varied, realistic requests the LLM writes itself (not hand-picked): every request is planned
with the product planner (converse.plan_turn); automatic checks flag: exception, zero albums, dates without a grounded
phrase, a surviving relational question (one photo cannot answer it), an album with no condition at all, a 'red box'
in an album without a person. Flagged plans are saved for reading. Usage (vLLM env, GPU): python eval/eval_planner_fuzz.py"""
import json
import os
import re
from datetime import date
from pathlib import Path

PEOPLE = ["Reza", "Dad", "Mom", "Sara", "Ali"]
# v2: v1's generator wrote web searches / photo edits for most kinds ("how to train a beagle"); now it is told the box
# searches the person's OWN library, and off-target requests are one explicit kind (the plan must still not crash).
KINDS = ["off-target things people type anyway (web questions, photo edits, half-typed messages)", "a person", "a pet", "food", "a place or trip", "a date range", "an event", "an object", "a mood or activity",
         "screenshots or documents", "videos", "a follow-up edit", "something excluded"]


# FUZZ_SET=1.. -> a different writer (held-out requests: the fixes of 10-03 were made reading set 0)
PERSONAS = ["a person", "a busy parent in their 40s who types fast with typos", "a retired grandparent who writes long, polite requests",
            "a college student who uses slang and abbreviations", "a travel photographer with a huge library"]
PERSONA = PERSONAS[int(os.environ.get("FUZZ_SET", "0"))]


def main():
    from findpics.converse import _RELATIONAL, plan_turn
    from findpics.vlm import VLLMJudge
    J = VLLMJudge(gpu_mem=0.8); T = date(2026, 10, 3)
    reqs = []
    for k in KINDS:
        out = J.text(f"Write 9 different realistic requests {PERSONA} might type into the search box of the photo app on their own phone, "
                     f"which searches only THEIR OWN photos and videos, about {k}. "
                     f"Vary phrasing, length and detail; some casual, some precise. One per line, no numbering.", max_tokens=400)
        reqs += [l.strip(" -*0123456789.").strip() for l in out.splitlines() if len(l.strip()) > 8][:9]
    from findpics.converse import _METADATA, _NAMED, _IS_NAME, _personal
    plans, flagged, stats = [], [], dict(n=len(reqs), exception=0, zero_albums=0, dates_without_phrase=0, relational=0,
                                     unseeable_q=0, known_name_in_q=0, date_span_over_400d=0,
                              no_condition=0, red_box_without_person=0)
    for r in reqs:
        try:
            P = plan_turn(r, J.text, owner="Reza", people=PEOPLE, today=T)
        except Exception as e:
            stats["exception"] += 1; flagged.append(dict(request=r, issue="exception", error=str(e)[:200])); continue
        issues = []
        if not P.albums:
            issues.append("zero_albums")
        for a in P.albums:
            if (a.date_from or a.date_to) and not a.time_phrase:
                issues.append("dates_without_phrase")
            for q in (a.judge_question, a.filter_question, a.exclude_question, a.anchor.judge_question if a.anchor else None):
                if q and _RELATIONAL.search(q):
                    issues.append("relational")
            for q in (a.judge_question, a.filter_question, a.exclude_question):
                if q and (_METADATA.search(q) or _NAMED.search(q) or _IS_NAME.search(q) or _personal(q)):
                    issues.append("unseeable_q")
                if q and any(re.search(r"\b" + n + r"\b", q) for n in PEOPLE if n != a.person):
                    issues.append("known_name_in_q")
            if a.date_from and a.date_to and a.time_phrase and not re.search(r"\d{4}|year|decade|s\b", a.time_phrase) and \
                    (date.fromisoformat(a.date_to) - date.fromisoformat(a.date_from)).days > 400:
                issues.append("date_span_over_400d")
            if not (a.person or a.judge_question or a.looks or a.date_from or a.date_to or a.place or a.anchor or a.media != "any"):
                issues.append("no_condition")
            for q in (a.judge_question, a.filter_question, a.exclude_question):
                if q and not a.person and "red box" in q.lower():
                    issues.append("red_box_without_person")
        plans.append(dict(request=r, issues=sorted(set(issues)), plan=P.model_dump(exclude_none=True)))
        for i in set(issues):
            stats[i] += 1
        if issues:
            flagged.append(dict(request=r, issues=sorted(set(issues)), plan=P.model_dump()))
    print(json.dumps(stats, indent=1))
    Path("eval/planners").mkdir(exist_ok=True)
    json.dump(dict(stats=stats, requests=reqs, flagged=flagged, plans=plans), open(f"eval/planners/fuzz{os.environ.get('FUZZ_SET', '') or ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
