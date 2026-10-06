"""Training data for a phone-size JUDGE: the 9B's P(yes) on (photo, question) pairs, to distill into the 4B (whose own
answers miss ~3x more real photos than the 9B on real libraries, 10-05 eye audit) - the same idea as the planner
distillation. Public data only (DISBench Flickr libraries).
Questions: the judge/filter/exclude questions the planners wrote for the synthetic fuzz requests (3,242 distinct),
minus anything about the everyday eval queries (held out: dog, cat, car, bicycle, boat, beach, food, flower, sunset,
church, selfie, night), so the everyday eye-labeled sets stay a clean test.
Photos per question: the 24 best image-vector matches for the question text (likely yes) + 24 random (likely no).
Usage (vLLM env, GPU): python eval/judge_distill_data.py --shard=k/K -> data/public/judge_distill/part{k}.jsonl
"""
import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path("data/public/judge_distill")
HELD_OUT = re.compile(r"(?i)\b(dogs?|puppy|puppies|cats?|kittens?|cars?|bicycles?|bikes?|boats?|beach(es)?|food|meals?|"
                      r"flowers?|sunsets?|church(es)?|selfies?|night)\b")
N_TOP, N_RAND = 24, 24


def questions():
    qs = set()
    for f in glob.glob("data/public/distill/planner27ball_clean.jsonl"):
        for line in open(f):
            r = json.loads(line)
            m = re.search(r"\{.*\}", r["output"], re.S)
            try:
                P = json.loads(m.group(0))
            except Exception:
                continue
            for a in P.get("albums", []):
                for k in ("judge_question", "filter_question", "exclude_question"):
                    q = (a.get(k) or "").strip()
                    named = re.match(r"(Is|Are) [A-Z][a-z]+( and [A-Z][a-z]+)? (visible|in|present|there)\b", q)
                    if q and not HELD_OUT.search(q) and "red box" not in q.lower() and len(q) < 200 and not named:
                        qs.add(q)
    return sorted(qs)


def main():
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    k, K = map(int, next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1").split("/"))
    qs = questions()[k::K]
    idx = store.load("data/public/index_disbench")
    enc = ImageTextEncoder(idx.clip_model)
    units = np.asarray(idx.clip, np.float32)
    unit_item = idx.units.item_row.to_numpy()
    photo = (idx.items.media == "photo").to_numpy()
    J = VLLMJudge(gpu_mem=0.75)
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / f"part{k}.jsonl", "w")
    rng = random.Random(k)
    T = enc.texts(qs)
    for qi, q in enumerate(qs):
        s = units @ T[qi]
        best = np.full(len(idx.items), -9.0, np.float32)
        np.maximum.at(best, unit_item, s)
        best[~photo] = -9.0
        top = list(np.argsort(-best)[:N_TOP])
        rest = [i for i in rng.sample(range(len(idx.items)), N_RAND * 3) if photo[i] and i not in top][:N_RAND]
        rows = np.array(top + rest, int)
        p = _judge_rows(idx, J, rows, np.full(len(rows), -1), q, batch=96)
        for r, pr, kind in zip(rows, p, ["top"] * len(top) + ["rand"] * len(rest)):
            fh.write(json.dumps(dict(question=q, item_id=str(idx.items.item_id.iloc[r]), p9=round(float(pr), 4),
                                     kind=kind)) + "\n")
        if qi % 50 == 0:
            fh.flush(); print(qi, "/", len(qs), flush=True)
    fh.close(); print("DONE", len(qs), flush=True)


if __name__ == "__main__":
    main()
