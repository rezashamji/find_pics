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
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    Q = [json.loads(l) for l in open("data/public/raw/disbench/queries.jsonl")][:nmax]
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    J = VLLMJudge(gpu_mem=0.7)
    out = []
    for q in Q:
        rows = np.where(user_of == q["user_id"])[0]
        sub = store.subset(idx, rows)
        try:
            P = plan(q["query"], J.text, today=date(2026, 10, 2))
            spec = P.albums[0]
            if not spec.judge_question:
                spec.judge_question = q["query"]
            spec.person = None  # DISBench has no identity tags; the planner's "I/we" must not trigger face mode
            r = run_album(sub, spec, enc, J, None, th=Thresholds(tail_budget=300))
            got = set(r.returned.item_id.astype(str))
            err = None
        except Exception as e:
            got, spec, err = set(), None, f"{type(e).__name__}: {e}"[:300]
        gt = set(map(str, q["answer"]))
        tp = len(got & gt); prec = tp / len(got) if got else 0.0; rec = tp / len(gt)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append(dict(query_id=q["query_id"], user=q["user_id"], event_type=q.get("event_type"), library=len(rows),
                        query=q["query"], gt=len(gt), returned=len(got), tp=tp, precision=prec, recall=rec, f1=f1,
                        exact=got == gt, plan=spec.model_dump() if spec is not None else None, error=err))
        print(f"q{q['query_id']} lib={len(rows)} gt={len(gt)} got={len(got)} tp={tp} F1={f1:.2f} | {q['query'][:90]}", flush=True)
    df = pd.DataFrame(out)
    print(df.groupby("event_type")[["precision", "recall", "f1", "exact"]].mean().round(3).to_string())
    print("ALL:", df[["precision", "recall", "f1", "exact"]].mean().round(3).to_dict(), "errors", int(df.error.notna().sum()))
    Path("eval/disbench").mkdir(exist_ok=True)
    df.to_json("eval/disbench/baseline.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
