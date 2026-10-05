"""Held-out check of the distilled phone planner: the 313 test-split prompts of the 27B distillation set (never trained
on). Base Qwen3.5-4B vs the distilled 4B (models/planner_4b27b_merged) vs the 27B teacher's answer, all GROUNDED by the
same code (converse.ground). Structured-field agreement with the teacher (album count, person, media, dates, time of
day, place, has filter/exclude) + 20 random cases printed for an eye check. The teacher is not ground truth.
Usage (vLLM env, GPU): python eval/eval_distill_test.py
"""
import json
import random
import re
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, "eval")
from distill_planner_data import OWNERS, PEOPLE  # noqa: E402

FIELDS = ["person", "media", "date_from", "date_to", "time_of_day", "place"]


def ctx(r):
    first = (r.get("history") or [r["request"]])[0]
    rnd = random.Random(first)
    owner = rnd.choice(OWNERS); people = rnd.sample(PEOPLE, rnd.randint(0, 4)) + [owner]
    today = date(2026, 10, 4) - timedelta(days=rnd.randint(0, 900))
    return owner, people, today


def grounded(out, r):
    from findpics import converse as C
    owner, people, today = ctx(r)
    try:
        raw = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        for al in raw.get("albums", []):
            an = al.get("anchor") if isinstance(al, dict) else None
            if isinstance(an, dict) and not an.get("judge_question"):
                lk = [x for x in an.get("looks") or [] if x]
                al["anchor"] = dict(an, judge_question=f"Does this photo show {lk[0]}?") if lk else None
        P = C.Plan.model_validate(raw)
        hist = r.get("history") or []
        said = " \n ".join(hist + [r["request"]])
        if C._unanswerable(P, people + [owner], said):
            P = C._drop_unanswerable(P, said, people + [owner])
        return C.ground(P, r["request"], hist, today, owner, people)
    except Exception:
        return None


def summary(P):
    if P is None:
        return None
    return dict(n=len(P.albums), albums=[{f: getattr(a, f) for f in FIELDS} | dict(
        filt=bool(a.filter_question), excl=bool(a.exclude_question), anchor=bool(a.anchor), q=a.judge_question) for a in P.albums])


def agree(a, b):
    if a is None or b is None:
        return dict(valid=a is not None, n=False, fields=0.0)
    n = a["n"] == b["n"]
    if not n:
        return dict(valid=True, n=False, fields=0.0)
    keys = FIELDS + ["filt", "excl", "anchor"]
    m = sum(x[k] == y[k] for x, y in zip(a["albums"], b["albums"]) for k in keys) / (len(keys) * a["n"])
    return dict(valid=True, n=True, fields=m)


def main():
    # one model per process (vLLM does not reliably free the GPU in-process): `gen <name>` twice, then `report`
    rows = [json.loads(l) for l in open("data/public/distill/planner27b_clean.jsonl")]
    test = [r for r in rows if r["split"] == "test"]
    paths = {"base_4b": "Qwen/Qwen3.5-4B", "distilled_4b": "models/planner_4b27b_merged",
             "distilled_4b_all": "models/planner_4b27ball_merged"}   # + 27B chained multi-turn edits
    if sys.argv[1] == "gen":
        from vllm import LLM, SamplingParams
        name = sys.argv[2]
        llm = LLM(model=paths[name], max_model_len=8192, gpu_memory_utilization=0.85, trust_remote_code=True, limit_mm_per_prompt={"image": 1})
        outs = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in test], SamplingParams(temperature=0, max_tokens=1024),
                        use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        json.dump([o.outputs[0].text for o in outs], open(f"eval/distill_test_{name}.json", "w"))
        return
    teacher = [summary(grounded(r["output"], r)) for r in test]
    res, plans = {}, {}
    for name in [n for n in paths if Path(f"eval/distill_test_{n}.json").exists()]:
        S = [summary(grounded(o, r)) for o, r in zip(json.load(open(f"eval/distill_test_{name}.json")), test)]
        A = [agree(x, t) for x, t in zip(S, teacher)]
        res[name] = dict(valid=sum(a["valid"] for a in A), same_album_count=sum(a["n"] for a in A),
                         field_agreement=round(sum(a["fields"] for a in A) / len(A), 3), n=len(A))
        plans[name] = S
        print(name, res[name], flush=True)
    rnd = random.Random(3)
    for i in rnd.sample(range(len(test)), 20):
        r = test[i]
        print("\n>>", (" / ".join(r.get("history", [])) + " => " if r.get("history") else "") + r["request"][:140])
        for name in plans:
            print(f"  {name:13s}", json.dumps(plans[name][i])[:400])
        print("  27b teacher  ", json.dumps(teacher[i])[:400])
    json.dump(res, open("eval/results_distill_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
