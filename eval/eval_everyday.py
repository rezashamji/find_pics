"""Everyday searches on REAL people's libraries (DISBench: 57 Flickr users' own photo collections, ~1,900 photos each),
through the product's chat planner + streaming engine. The stand-in for Reza's full-library final exam while the Apple
copy is pending: messy real libraries instead of a hand-picked sample.

Per (user, query): round 1 (fast answer) and the last round (exhaustive: the judge has looked at every in-scope photo).
Reports the fast-vs-exhaustive gap; the exhaustive set itself is audited BY EYE afterwards (eval/everyday/audit_*.jpg),
because the judge is not ground truth.
Usage (vLLM env, GPU): python eval/eval_everyday.py --shard=k/K   (one user library per shard step; merge at the end)
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

QUERIES = ["food photos", "photos with a dog", "photos of a cat", "beach photos", "photos taken at night",
           "photos with a car", "photos of flowers", "photos with a bicycle", "sunset photos", "selfies",
           "photos of a church", "photos with a boat"]
if __import__("os").environ.get("FP_EVERYDAY_QUERIES"):     # e.g. "selfies": re-measure one query after a change
    QUERIES = __import__("os").environ["FP_EVERYDAY_QUERIES"].split("|")
N_USERS = int(__import__("os").environ.get("FP_EVERYDAY_N", "8"))
EXCLUDE = __import__("os").environ.get("FP_EVERYDAY_EXCLUDE")   # a run dir: skip its users (fresh libraries)
OUT = Path(__import__("os").environ.get("FP_EVERYDAY_OUT", "eval/everyday"))   # v2: rerun after the 10-05 fixes


def users(idx):
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0]
    vc = user_of.value_counts()
    rng = np.random.default_rng(0)                       # 8 random users with a normal-size library (1,000-4,000)
    ok = sorted(vc[(vc >= 1000) & (vc <= 4000)].index)
    if EXCLUDE:
        seen = {r["user"] for f in Path(EXCLUDE).glob("part*.json") for r in json.load(open(f))}
        ok = [u for u in ok if u not in seen]
    return user_of.to_numpy(), list(rng.choice(ok, size=min(N_USERS, len(ok)), replace=False))


def main():
    from findpics import store
    from findpics.converse import plan_turn, stream_plan
    from findpics.engine import Thresholds
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    idx = store.load("data/public/index_disbench")
    user_of, us = users(idx)
    enc = ImageTextEncoder(idx.clip_model)
    J = VLLMJudge(gpu_mem=0.7)
    import os
    from findpics.converse import Plan
    if os.environ.get("FP_EVERYDAY_PLANS"):   # reuse another run's plans: only the JUDGE differs (judge comparisons)
        ref = [r for f in sorted(Path(os.environ["FP_EVERYDAY_PLANS"]).glob("part*.json")) for r in json.load(open(f))]
        plans = {q: Plan.model_validate(next(r["plan"] for r in ref if r["query"] == q)) for q in QUERIES}
    else:
        plans = {q: plan_turn(q, J.text, today=date(2026, 10, 4)) for q in QUERIES}   # same plan for every user
    out = []
    for u in us[k::K]:
        sub = store.subset(idx, np.where(user_of == u)[0])
        for q in QUERIES:
            P = plans[q].model_copy(deep=True)
            for a in P.albums:
                a.person = None                       # no identity tags on Flickr; "selfies" stays a look
            t0 = time.time(); rounds = []
            try:
                for rs in stream_plan(sub, P, enc, J, th=Thresholds(tail_budget=300, stream=True)):
                    got = set().union(*[set(r.returned.item_id.astype(str)) for r in rs]) if rs else set()
                    rounds.append(dict(t=round(time.time() - t0, 1), ids=sorted(got)))
                err = None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
            first, last = (rounds[0]["ids"], rounds[-1]["ids"]) if rounds else ([], [])
            ov = len(set(first) & set(last))
            out.append(dict(user=u, query=q, library=len(sub.items), plan=P.model_dump(), rounds=len(rounds),
                            fast=first, exhaustive=last, fast_found=ov, fast_extra=len(set(first) - set(last)),
                            t_fast=rounds[0]["t"] if rounds else None, t_all=rounds[-1]["t"] if rounds else None,
                            error=err))
            print(f"{u} lib={len(sub.items)} {q!r}: fast {len(first)} exhaustive {len(last)} "
                  f"(fast found {ov}/{len(last)}) {out[-1]['t_fast']}s/{out[-1]['t_all']}s {err or ''}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"part{k}.json", "w"), indent=1, default=str)


def merge():
    rows = [r for f in sorted(OUT.glob("part*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df["n_ex"] = df.exhaustive.map(len); df["n_fast"] = df.fast.map(len)
    g = df.groupby("query").agg(users=("user", "nunique"), exhaustive=("n_ex", "sum"), fast=("n_fast", "sum"),
                                fast_found=("fast_found", "sum"), t_fast=("t_fast", "median"), t_all=("t_all", "median"))
    g["fast_recall_vs_exhaustive"] = (g.fast_found / g.exhaustive.clip(lower=1)).round(3)
    print(g.to_string()); print("errors", int(df.error.notna().sum()), "of", len(df))
    df.drop(columns=["n_ex", "n_fast"]).to_json(OUT / "all.json", orient="records", indent=1, default_handler=str)
    (OUT / "summary.txt").write_text(g.to_string())


if __name__ == "__main__":
    merge() if "merge" in sys.argv else main()
