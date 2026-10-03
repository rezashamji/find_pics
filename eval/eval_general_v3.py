"""Generalization, re-measured with the CURRENT product planner (findpics.converse) and streaming.
Same 72 queries (eval/general/queries.json, written from real library photos) and same 5,000-photo subset as
eval_general.py; the plan now comes from converse.plan_turn (person dropped: this public library has no reference faces
for "I/we"). The judge answers the planned main question on EVERY subset photo (oracle). Analysis replays streaming
round 1 on those stored answers: round-1 recall of the oracle set, stated bound, seed photo found.
Usage (vLLM env, GPU): python eval/eval_general_v3.py oracle <shard> <n_shards>   |   (CPU) python eval/eval_general_v3.py analyze
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
from eval_general import IDX, OUT as V1, subset_rows  # noqa: E402

OUT = Path("eval/general_v3"); OUT.mkdir(parents=True, exist_ok=True)


def oracle(shard, n):
    from findpics import store
    from findpics.converse import plan_turn
    from findpics.engine import _judge_rows, look_scores
    from findpics.models import ImageTextEncoder
    from findpics.planner import fix_red_box
    from findpics.vlm import VLLMJudge
    Q = json.load(open(V1 / "queries.json"))[shard::n]
    idx = store.load(IDX); rows = subset_rows(idx.n_items)
    enc = ImageTextEncoder(idx.clip_model); J = VLLMJudge(gpu_mem=0.70)
    for q in Q:
        f = OUT / f"oracle_{q['k']}.parquet"
        if f.exists():
            continue
        try:
            P = plan_turn(q["query"], J.text, today=date(2026, 10, 2))
            for a in P.albums:
                a.person = None
            fix_red_box(P); a = P.albums[0]
        except Exception as e:
            json.dump(dict(k=q["k"], error=str(e)[:300]), open(OUT / f"plan_fail_{q['k']}.json", "w")); continue
        jq = a.judge_question
        json.dump(dict(k=q["k"], plan=a.model_dump(), judge_question=jq), open(OUT / f"plan_{q['k']}.json", "w"))
        if not jq:          # no visual condition: the album is "everything in scope"; nothing to judge
            continue
        look = look_scores(idx, enc, a.looks or [q["query"]], a.avoid)
        p = _judge_rows(idx, J, rows, np.full(len(rows), -1), jq, batch=96)
        pd.DataFrame(dict(item_row=rows, p=p, look=look[rows])).to_parquet(f)
        print(f"query {q['k']} [{q['dimension']}] -> '{jq}': yes@0.7={int((p >= .7).sum())}", flush=True)


def analyze():
    import findpics.engine as E
    from findpics.planner import AlbumSpec
    from findpics.store import Index
    Qs = {q["k"]: q for q in json.load(open(V1 / "queries.json"))}
    res = []
    for f in sorted(OUT.glob("oracle_*.parquet")):
        k = int(f.stem.split("_")[1]); q = Qs[k]; d = pd.read_parquet(f); pl = json.load(open(OUT / f"plan_{k}.json"))
        P = d.p.to_numpy(); L = d.look.to_numpy(); n = len(d)
        items = pd.DataFrame(dict(item_id=d.item_row.astype(str), path="", media="photo", taken="2020-01-01T00:00:00+00:00"))
        idx = Index(None, items, pd.DataFrame(dict(item_row=np.arange(n), frame_t=-1.0)), np.zeros((n, 1), np.float16),
                    pd.DataFrame(), np.zeros((0, 512), np.float16), pd.DataFrame())
        E._frame_for = lambda idx, r, fr: r; E._boxed = lambda idx, im, fr: im
        E.look_scores = lambda idx, enc, looks, avoid, _L=L: _L

        class SJ:
            def p_yes(self, ims, qq):
                return [float(P[int(i)]) for i in ims]
        r = next(E.stream_album(idx, AlbumSpec(name="g", looks=["x"], judge_question=pl["judge_question"]), None, SJ(), None,
                                th=E.Thresholds(stream=True)))
        oracle_set = set(np.where(P >= 0.7)[0]); fast = set(r.returned.item_row)
        seed = int(np.where(d.item_row.to_numpy() == q["seed_row"])[0][0])
        res.append(dict(k=k, dimension=q["dimension"], query=q["query"], judge_question=pl["judge_question"],
                        oracle=len(oracle_set), round1_recall=len(fast & oracle_set) / max(len(oracle_set), 1),
                        bound=r.cert["recall_lower"] if r.cert else None, holds=(r.cert["recall_lower"] <= len(fast & oracle_set) /
                        max(len(oracle_set), 1) + 1e-9) if r.cert and oracle_set else None,
                        seed_yes=bool(P[seed] >= 0.7), seed_round1=seed in fast))
    df = pd.DataFrame(res)
    print(df.groupby("dimension").agg(queries=("k", "count"), round1_recall=("round1_recall", "mean"),
                                      seed_yes=("seed_yes", "mean"), oracle_median=("oracle", "median")).round(2).to_string())
    print("ALL: round-1 recall", round(df.round1_recall.mean(), 3), "| seed judged yes", round(df.seed_yes.mean(), 3),
          "| bound held", int(df.holds.sum()), "of", int(df.holds.notna().sum()),
          "| no-condition plans", len(list(OUT.glob("plan_*.json"))) - len(df))
    df.to_json("eval/results_general_v3.json", orient="records", indent=1)


if __name__ == "__main__":
    oracle(int(sys.argv[2]), int(sys.argv[3])) if sys.argv[1] == "oracle" else analyze()
