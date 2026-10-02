"""DISBench (real users' libraries; hard, often multi-step queries) through the CURRENT product pipeline.
Per query: library = that user's photos only (store.subset); planner on the raw query; fast mode; score returned set
vs ground-truth photo ids: precision, recall, F1, exact match. This is the BASELINE before the multi-step planner.
Usage (vLLM env, GPU): python eval/eval_disbench.py [max_queries]
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    from findpics import store
    from findpics.engine import Thresholds, run_album
    from findpics.models import ImageTextEncoder
    from findpics.planner import plan
    from findpics.vlm import VLLMJudge
    mode = "agent" if "agent" in sys.argv else "unified" if "unified" in sys.argv else "baseline"
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    nmax = int(nums[0]) if nums else 10_000
    from findpics.agent import execute, make_plan
    Q = [json.loads(l) for l in open("data/public/raw/disbench/queries.jsonl")][:nmax]
    # sharding across GPUs: "--shard k/K" (Slurm array: k = SLURM_ARRAY_TASK_ID) -> eval/disbench/<mode>_part<k>.json
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), None)
    if shard:
        k, K = map(int, shard.split("/")); Q = Q[k::K]
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    J = VLLMJudge(gpu_mem=0.7)
    out = []
    for q in Q:
        rows = np.where(user_of == q["user_id"])[0]
        sub = store.subset(idx, rows)
        try:
            if mode == "unified":     # the product path: one planner for everything (findpics.converse)
                from findpics.converse import plan_turn, run_plan
                spec = plan_turn(q["query"], J.text, today=date(2026, 10, 2))
                for a in spec.albums:
                    a.person = None    # DISBench has no identity tags
                rs = run_plan(sub, spec, enc, J, th=Thresholds(tail_budget=300))
                got = set().union(*[set(r.returned.item_id.astype(str)) for r in rs]) if rs else set()
                trace = [getattr(r, "trace", {}) for r in rs]
            elif mode == "agent":
                spec = make_plan(q["query"], J.text, today=date(2026, 10, 2))
                res = execute(sub, spec, enc, J, th=Thresholds(tail_budget=300))
                got = set(res["returned"].item_id.astype(str)); trace = res["trace"]
            else:
                P = plan(q["query"], J.text, today=date(2026, 10, 2))
                spec = P.albums[0]
                if not spec.judge_question:
                    spec.judge_question = q["query"]
                spec.person = None  # DISBench has no identity tags; the planner's "I/we" must not trigger face mode
                r = run_album(sub, spec, enc, J, None, th=Thresholds(tail_budget=300))
                got = set(r.returned.item_id.astype(str)); trace = None
            err = None
        except Exception as e:
            got, spec, trace, err = set(), None, None, f"{type(e).__name__}: {e}"[:300]
        gt = set(map(str, q["answer"]))
        tp = len(got & gt); prec = tp / len(got) if got else 0.0; rec = tp / len(gt)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append(dict(query_id=q["query_id"], user=q["user_id"], event_type=q.get("event_type"), library=len(rows),
                        query=q["query"], gt=len(gt), returned=len(got), tp=tp, precision=prec, recall=rec, f1=f1,
                        exact=got == gt, plan=spec.model_dump() if spec is not None else None, trace=trace, error=err))
        print(f"q{q['query_id']} lib={len(rows)} gt={len(gt)} got={len(got)} tp={tp} F1={f1:.2f} | {q['query'][:90]}", flush=True)
    df = pd.DataFrame(out)
    print(df.groupby("event_type")[["precision", "recall", "f1", "exact"]].mean().round(3).to_string())
    print("ALL:", df[["precision", "recall", "f1", "exact"]].mean().round(3).to_dict(), "errors", int(df.error.notna().sum()))
    Path("eval/disbench").mkdir(exist_ok=True)
    name = f"{mode}_part{shard.split('/')[0]}" if shard else mode
    df.to_json(f"eval/disbench/{name}.json", orient="records", indent=1, default_handler=str)


def merge(mode):
    """python eval/eval_disbench.py merge <mode>: join the shard files into eval/disbench/<mode>.json and summarize."""
    fs = sorted(Path("eval/disbench").glob(f"{mode}_part*.json"))
    df = pd.concat([pd.read_json(f) for f in fs], ignore_index=True).sort_values("query_id")
    print(f"{len(fs)} shards, {len(df)} queries")
    print(df.groupby("event_type")[["precision", "recall", "f1", "exact"]].mean().round(3).to_string())
    print("ALL:", df[["precision", "recall", "f1", "exact"]].mean().round(3).to_dict(), "errors", int(df.error.notna().sum()),
          "| returned nothing", int((df.returned == 0).sum()), "| >=1 correct", int((df.tp > 0).sum()))
    df.to_json(f"eval/disbench/{mode}.json", orient="records", indent=1, default_handler=str)


if __name__ == "__main__":
    merge(sys.argv[2]) if sys.argv[1] == "merge" else main()
