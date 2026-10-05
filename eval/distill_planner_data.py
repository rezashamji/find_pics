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

OUT = Path("data/public/distill")
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
    from findpics.converse import plan_turn
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
    calls = []

    def rec(prompt, max_tokens=1024):
        out = J.text(prompt, max_tokens=max_tokens)
        calls.append((prompt, out))
        return out

    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / f"planner_part{k}.jsonl", "w")
    for i, msg in enumerate(reqs):
        rnd = random.Random(msg)
        owner = rnd.choice(OWNERS); people = rnd.sample(PEOPLE, rnd.randint(0, 4)) + [owner]
        today = date(2026, 10, 4) - timedelta(days=rnd.randint(0, 900))
        split = "test" if int(hashlib.md5(msg.encode()).hexdigest(), 16) % 100 < 15 else "train"
        try:
            calls.clear()
            P = plan_turn(msg, rec, owner=owner, people=people, today=today)
            p1, o1 = calls[-1]
            fh.write(json.dumps(dict(split=split, turn=1, request=msg, prompt=p1, output=o1)) + "\n")
            if rnd.random() < 0.6:            # a follow-up edit for most requests
                follow = J.text(FOLLOW.format(msg=msg), max_tokens=60).strip().strip('"').splitlines()[0][:200]
                calls.clear()
                plan_turn(follow, rec, history=[msg], current=P, owner=owner, people=people, today=today)
                p2, o2 = calls[-1]
                fh.write(json.dumps(dict(split=split, turn=2, request=follow, history=[msg], prompt=p2, output=o2)) + "\n")
        except Exception as e:
            print("skip", msg[:60], type(e).__name__, str(e)[:100], flush=True)
        if i % 50 == 0:
            print(f"{i}/{len(reqs)}", flush=True); fh.flush()
    fh.close()
    print("DONE", len(reqs), flush=True)


if __name__ == "__main__":
    main()
