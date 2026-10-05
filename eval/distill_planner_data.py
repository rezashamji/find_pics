"""Training data for a phone-size planner: the 9B planner's exact answers (prompt -> JSON plan) on the fuzz-corpus
requests (LLM-written, synthetic: no one's real data), plus one LLM-written follow-up edit per request for multi-turn.
Why: Qwen3.5-4B as planner passed 20/30 scripted conversations vs 9B 29-30/30 (10-05), failing mostly on follow-up
edits ("drop the sandwiches", "only red ones"). The 30 eval conversations are held out (never in training), and 15% of
requests (by hash) form a test split.
Usage (vLLM env, GPU, 9B): python eval/distill_planner_data.py --shard=k/K -> data/public/distill/planner_part{k}.jsonl
"""
import glob
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import os
OUT = Path("data/public/distill")
TAG = os.environ.get("FP_DISTILL_TAG", "")   # e.g. "27b": a different teacher writes planner27b_part*.jsonl
FOLLOW = """Here is a message someone typed into a photo-search app about their own photo library:
"{msg}"
Write ONE short follow-up message the same person might type next to change the results (narrow them, exclude
something, add something, change the dates, split into two albums, or ask for videos too). Write it the way this
person writes (same style, typos welcome). Reply with only the message."""
OWNERS = ["Reza", "Maya", "Sam", "Priya", "Jordan", "Lena", "Omar", "Kai"]
PEOPLE = ["Mom", "Dad", "Jay", "Nic", "Patricia", "Grandma", "Leo", "Ana", "Chris", "Zoe"]


def held_out():
    sys.path.insert(0, "eval")
    import eval_planners as E
    msgs = set()
    for msgs_, _check in E.CONVS:
        for m in msgs_:
            msgs.add(m.strip().lower())
    assert len(msgs) > 30, len(msgs)
    return msgs


def main():
    from findpics.converse import build_prompt, plan_turn
    from findpics.vlm import VLLMJudge
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    reqs = set()
    for f in glob.glob("eval/planners/fuzz*.json"):
        d = json.load(open(f))
        if isinstance(d, dict):
            reqs |= {r.strip() for r in d.get("requests", []) if r and r.strip()}
    hold = held_out()
    reqs = sorted(r for r in reqs if r.lower() not in hold)[k::K]
    J = VLLMJudge(gpu_mem=0.85)
    cache, calls = {}, []

    def batch(prompts, max_tokens=1024):      # one vLLM batch instead of one request at a time (10 s each)
        todo = [p for p in dict.fromkeys(prompts) if p not in cache]
        outs = J._chat([[{"role": "user", "content": p}] for p in todo], J.SP(temperature=0.0, max_tokens=max_tokens))
        cache.update({p: o.outputs[0].text for p, o in zip(todo, outs)})

    def rec(prompt, max_tokens=1024):          # what plan_turn calls: cached batch answer, else a single call (retries)
        out = cache.get(prompt)
        if out is None:
            out = J.text(prompt, max_tokens=max_tokens)
        calls.append((prompt, out))
        return out

    ctx = {}
    for msg in reqs:
        rnd = random.Random(msg)
        owner = rnd.choice(OWNERS); people = rnd.sample(PEOPLE, rnd.randint(0, 4)) + [owner]
        today = date(2026, 10, 4) - timedelta(days=rnd.randint(0, 900))
        split = "test" if int(hashlib.md5(msg.encode()).hexdigest(), 16) % 100 < 15 else "train"
        ctx[msg] = dict(owner=owner, people=people, today=today, split=split, follow=rnd.random() < 0.6)
    batch([build_prompt(m, [], None, c["owner"], c["people"], c["today"]) for m, c in ctx.items()])
    print("turn-1 batch done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / f"planner{TAG}_part{k}.jsonl", "w")
    plans = {}
    for msg, c in ctx.items():
        try:
            calls.clear()
            plans[msg] = plan_turn(msg, rec, owner=c["owner"], people=c["people"], today=c["today"])
            fh.write(json.dumps(dict(split=c["split"], turn=1, request=msg, prompt=calls[-1][0], output=calls[-1][1])) + "\n")
        except Exception as e:
            print("skip", msg[:60], type(e).__name__, str(e)[:100], flush=True)
    fols = [m for m in plans if ctx[m]["follow"]]
    batch([FOLLOW.format(msg=m) for m in fols], max_tokens=60)
    follow = {m: cache[FOLLOW.format(msg=m)].strip().strip('"').splitlines()[0][:200] if cache[FOLLOW.format(msg=m)].strip()
              else None for m in fols}
    follow = {m: f for m, f in follow.items() if f}
    batch([build_prompt(f, [m], plans[m], ctx[m]["owner"], ctx[m]["people"], ctx[m]["today"]) for m, f in follow.items()])
    print("turn-2 batch done", flush=True)
    for m, f in follow.items():
        c = ctx[m]
        try:
            calls.clear()
            plan_turn(f, rec, history=[m], current=plans[m], owner=c["owner"], people=c["people"], today=c["today"])
            fh.write(json.dumps(dict(split=c["split"], turn=2, request=f, history=[m], prompt=calls[-1][0],
                                     output=calls[-1][1])) + "\n")
        except Exception as e:
            print("skip2", f[:60], type(e).__name__, str(e)[:100], flush=True)
    fh.close()
    print("DONE", len(reqs), "requests", len(follow), "follow-ups", flush=True)


if __name__ == "__main__":
    main()
