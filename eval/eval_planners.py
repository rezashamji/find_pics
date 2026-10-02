"""Does ONE conversational planner (findpics.converse) replace the one-step planner, the multi-step planner and the
refine-op menu without getting worse? Text only (no photos), Qwen3.5-9B via vLLM.
  A. 12 scripted conversations (first request + follow-ups) with checks on the final plan.
  B. The 72 generalization queries (single photos described in words): how often does the new planner invent a
     two-step plan where none is needed? Old vs new judge questions saved side by side for reading.
  C. DISBench queries: how often does each planner make a two-step plan (old multi-step planner vs new)?
Output: eval/planners/{conversations,general,disbench}.json + printed pass counts.
Usage (vLLM env, GPU): python eval/eval_planners.py
"""
import json
from datetime import date
from pathlib import Path

T = date(2026, 10, 2)
PEOPLE = ["Reza", "Dad", "Mom"]


def _q(a):
    return " ".join([a.judge_question or ""] + a.looks).lower()


def _has(words, s):
    return any(w in s for w in words)


CONVS = [
    (["me looking heavier vs me looking fit in the past 6 months"],
     lambda P: len(P.albums) == 2 and all(a.person for a in P.albums) and sum(bool(a.date_from) for a in P.albums) == 1
     and all(a.anchor is None for a in P.albums)),
    (["photos of Dad at the beach", "only the ones from 2019"],
     lambda P: P.albums[0].person and "dad" in P.albums[0].person.lower() and _has(["beach"], _q(P.albums[0]))
     and (P.albums[0].date_from or "").startswith("2019")),
    (["all my photos with bread", "drop the sandwiches"],
     lambda P: _has(["bread"], _q(P.albums[0])) and _has(["sandwich"], (P.albums[0].exclude_question or "").lower())),
    (["bicycles", "look harder"],
     lambda P: len(P.albums) == 1 and _has(["bicycle", "bike"], _q(P.albums[0]))),   # no-op: the search just keeps going
    (["photos from the day of my graduation"],
     lambda P: P.albums[0].anchor is not None and P.albums[0].window in ("same_day", "same_event")),
    (["pictures of dogs", "actually just videos"],
     lambda P: P.albums[0].media == "video" and _has(["dog"], _q(P.albums[0]))),
    (["the best photos of sunsets"],
     lambda P: P.albums[0].want == "best" and P.albums[0].anchor is None),
    (["food photos from the week I went to the Grand Canyon, no burgers"],
     lambda P: P.albums[0].anchor is not None and P.albums[0].window == "same_week"
     and _has(["burger"], (P.albums[0].exclude_question or "").lower())),
    (["photos of me with Dad", "remove the ones where we're at a restaurant"],
     lambda P: any(a.person for a in P.albums) and _has(["restaurant"], " ".join((a.exclude_question or "") for a in P.albums).lower())),
    (["screenshots of text messages"],
     lambda P: P.albums[0].anchor is None and P.albums[0].person is None and _has(["screenshot", "text", "message"], _q(P.albums[0]))),
    (["me looking heavier vs me looking fit in the past 6 months", "make the fit album only photos where I'm at the gym"],
     lambda P: len(P.albums) == 2 and any(_has(["gym"], _q(a)) and a.date_from for a in P.albums)
     and any(not _has(["gym"], _q(a)) and not a.date_from for a in P.albums)),
    (["photos of my cat", "also videos of her", "only from Paris"],
     lambda P: _has(["cat"], _q(P.albums[0])) and P.albums[0].media in ("any", "video") and all(a.place == "Paris" for a in P.albums)),
]


def main():
    from findpics import agent, planner
    from findpics.converse import plan_turn
    from findpics.vlm import VLLMJudge
    J = VLLMJudge(gpu_mem=0.8)
    out = Path("eval/planners"); out.mkdir(parents=True, exist_ok=True)

    conv = []
    for msgs, check in CONVS:
        P, hist, err = None, [], None
        try:
            for m in msgs:
                P = plan_turn(m, J.text, history=hist, current=P, owner="Reza", people=PEOPLE, today=T); hist.append(m)
            ok = bool(P.albums) and bool(check(P))
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"[:300]
        conv.append(dict(messages=msgs, ok=ok, error=err, plan=P.model_dump() if P else None))
        print("PASS" if ok else "FAIL", msgs, err or "", flush=True)
    json.dump(conv, open(out / "conversations.json", "w"), indent=1)
    print(f"A. conversations: {sum(c['ok'] for c in conv)}/{len(conv)} pass")

    Qs = json.load(open("eval/general/queries.json"))
    gen = []
    for q in Qs:
        try:
            new = plan_turn(q["query"], J.text, today=T)
            old = planner.plan(q["query"], J.text, today=T)
            gen.append(dict(k=q["k"], query=q["query"], two_step=any(a.anchor for a in new.albums), n_new=len(new.albums),
                            n_old=len(old.albums), new_q=[a.judge_question for a in new.albums],
                            old_q=[a.judge_question for a in old.albums], new=new.model_dump()))
        except Exception as e:
            gen.append(dict(k=q["k"], query=q["query"], error=str(e)[:300]))
    json.dump(gen, open(out / "general.json", "w"), indent=1)
    ok = [g for g in gen if "error" not in g]
    print(f"B. generalization queries: {len(ok)}/{len(gen)} planned; two-step (should be ~0): "
          f"{sum(g['two_step'] for g in ok)}/{len(ok)}; album count same as old planner: {sum(g['n_new'] == g['n_old'] for g in ok)}/{len(ok)}")

    D = [json.loads(l) for l in open("data/public/raw/disbench/queries.jsonl")]
    dis = []
    for q in D:
        try:
            new = plan_turn(q["query"], J.text, today=T)
            old = agent.make_plan(q["query"], J.text, today=T)
            dis.append(dict(query_id=q["query_id"], event_type=q.get("event_type"), query=q["query"],
                            new_two_step=any(a.anchor for a in new.albums), old_two_step=old.anchor is not None,
                            new_exclude=any(a.exclude_question for a in new.albums), old_exclude=bool(old.exclude_question),
                            new=new.model_dump(), old=old.model_dump()))
        except Exception as e:
            dis.append(dict(query_id=q["query_id"], query=q["query"], error=str(e)[:300]))
    json.dump(dis, open(out / "disbench.json", "w"), indent=1)
    ok = [d for d in dis if "error" not in d]
    agree = sum(d["new_two_step"] == d["old_two_step"] for d in ok)
    print(f"C. DISBench: {len(ok)}/{len(dis)} planned; two-step new {sum(d['new_two_step'] for d in ok)} vs old "
          f"{sum(d['old_two_step'] for d in ok)}; agree {agree}/{len(ok)}; exclusion new {sum(d['new_exclude'] for d in ok)} "
          f"vs old {sum(d['old_exclude'] for d in ok)}")


if __name__ == "__main__":
    main()
