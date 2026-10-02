"""Generalization test that does not depend on anyone's imagination.

  gen    (vLLM env, 1 GPU): sample photos from the library -> the VLM describes each -> the LLM writes ONE realistic
         search query that photo answers, for an assigned query dimension (rotating through the list below).
         Every query is therefore grounded in real content and has >= 1 true answer. -> eval/general/queries.json
  oracle (vLLM env, array): each query goes through the REAL planner (tests the planner too); the judge answers the
         planned yes/no question on EVERY item of a fixed 5,000-item subset -> eval/general/oracle_<k>.parquet
  analyze (CPU): replay fast mode on the same stored judgments; per query and per dimension: oracle size, fast recall
         of the oracle, seed photo found?, judge calls. -> eval/results_general.json
Dimensions (initial; to be extended from research/05_query_space.md):
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

IDX = "data/public/index_testlib"
OUT = Path("eval/general"); OUT.mkdir(parents=True, exist_ok=True)
DIMS = ["object", "activity", "scene or place type", "text visible in the image", "count of things",
        "spatial relation between two things", "negation (something present but something else absent)",
        "attribute such as color or clothing", "mood, emotion or facial expression", "event or occasion",
        "time of day, weather or lighting", "photo quality or style (blurry, close-up, black and white)"]
N_SUBSET = 5000


def subset_rows(n_items):
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n_items, min(N_SUBSET, n_items), replace=False))


def gen(n_queries=72):
    from findpics import store
    from findpics.engine import _frame_for
    from findpics.vlm import VLLMJudge, _data_url
    idx = store.load(IDX)
    rows = subset_rows(idx.n_items)
    rng = np.random.default_rng(7)
    seeds = rng.choice(rows, n_queries, replace=False)
    J = VLLMJudge(max_model_len=8192)
    ims = [_frame_for(idx, int(r), -1) for r in seeds]
    msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(im)}},
             {"type": "text", "text": "Describe this photo in detail: people (no names), animals, objects, text visible, "
                                      "setting, activity, lighting, mood. 4-6 sentences."}]}] for im in ims]
    caps = [o.outputs[0].text.strip() for o in J._chat(msgs, J.SP(temperature=0.0, max_tokens=300))]
    out = []
    for k, (r, cap) in enumerate(zip(seeds, caps)):
        dim = DIMS[k % len(DIMS)]
        prompt = (f"A photo in someone's camera roll looks like this:\n{cap}\n\nWrite ONE realistic search request that "
                  f"person might type into their photo app to find this photo and others like it, focusing on: {dim}. "
                  f"Write it the way people actually talk (e.g. 'show me pics where...'). Do not use proper names. "
                  f"Return JSON: {{\"query\": str}}")
        txt = J.text(prompt, max_tokens=120)
        try:
            q = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["query"]
        except Exception:
            q = txt.strip().splitlines()[0][:200]
        out.append(dict(k=k, seed_row=int(r), seed_id=idx.items.item_id.iloc[int(r)], dimension=dim, caption=cap, query=q))
        print(k, dim, "|", q, flush=True)
    json.dump(out, open(OUT / "queries.json", "w"), indent=1)


def oracle(task, per_task=6):
    from datetime import date
    from findpics import store
    from findpics.engine import _judge_rows, look_scores
    from findpics.models import ImageTextEncoder
    from findpics.planner import plan
    from findpics.vlm import VLLMJudge
    Q = json.load(open(OUT / "queries.json"))
    only = [int(x) for x in os.environ.get("FP_ONLY_K", "").split(",") if x]
    Q = [q for q in Q if q["k"] in only] if only else Q[task * per_task:(task + 1) * per_task]
    idx = store.load(IDX)
    rows = subset_rows(idx.n_items)
    enc = ImageTextEncoder(idx.clip_model)
    J = VLLMJudge(gpu_mem=0.70)
    for q in Q:
        f = OUT / f"oracle_{q['k']}.parquet"
        if f.exists():
            continue
        try:
            P = plan(q["query"], J.text, today=date(2026, 10, 2))
            # this library has no reference faces for "I/we/me": the product would refuse such an album, so the test
            # drops the person and re-applies the red-box rule (else the judge is asked about a box never drawn:
            # 6/72 plans in v2, source photo accepted 3/6 vs 54/66)
            from findpics.planner import fix_red_box
            for al in P.albums:
                al.person = None
            fix_red_box(P)
            a = P.albums[0]
        except Exception as e:
            json.dump(dict(k=q["k"], error=str(e)[:300]), open(OUT / f"plan_fail_{q['k']}.json", "w")); continue
        jq = a.judge_question or q["query"]
        look = look_scores(idx, enc, a.looks or [q["query"]], a.avoid)
        t = time.time()
        p = _judge_rows(idx, J, rows, np.full(len(rows), -1), jq, batch=96)
        pd.DataFrame(dict(item_row=rows, p=p, look=look[rows])).to_parquet(f)
        json.dump(dict(k=q["k"], plan=a.model_dump(), judge_question=jq), open(OUT / f"plan_{q['k']}.json", "w"))
        print(f"query {q['k']} [{q['dimension']}] '{q['query']}' -> judge '{jq}': yes@0.7={int((p >= .7).sum())} "
              f"({time.time()-t:.0f}s)", flush=True)
    n_all = len(json.load(open(OUT / "queries.json")))
    if len(list(OUT.glob("oracle_*.parquet"))) == n_all:   # the last job to finish writes the analysis
        import contextlib
        with open(OUT / "analyze_latest.txt", "w") as fh, contextlib.redirect_stdout(fh):
            analyze()


def analyze():
    import findpics.engine as E
    from findpics.planner import AlbumSpec
    from findpics.store import Index
    Qs = {q["k"]: q for q in json.load(open(OUT / "queries.json"))}
    res = []
    for f in sorted(OUT.glob("oracle_*.parquet")):
        k = int(f.stem.split("_")[1]); q = Qs[k]; d = pd.read_parquet(f)
        pl = json.load(open(OUT / f"plan_{k}.json"))
        P = d.p.to_numpy(); L = d.look.to_numpy(); n = len(d)
        items = pd.DataFrame(dict(item_id=d.item_row.astype(str), path="", media="photo", taken="2020-01-01T00:00:00+00:00"))
        idx = Index(None, items, pd.DataFrame(dict(item_row=np.arange(n), frame_t=-1.0)), np.zeros((n, 1), np.float16),
                    pd.DataFrame(), np.zeros((0, 512), np.float16), pd.DataFrame())
        E._frame_for = lambda idx, r, fr: r; E._boxed = lambda idx, im, fr: im
        E.look_scores = lambda idx, enc, looks, avoid, _L=L: _L

        class SJ:
            calls = 0
            def p_yes(self, ims, qq):
                SJ.calls += len(ims); return [float(P[int(i)]) for i in ims]
        r = E.run_album(idx, AlbumSpec(name="g", looks=["x"], judge_question=pl["judge_question"]), None, SJ(), None)
        oracle = set(np.where(P >= 0.7)[0]); fast = set(r.returned.item_row)
        seed_pos = int(np.where(d.item_row.to_numpy() == q["seed_row"])[0][0])
        res.append(dict(k=k, dimension=q["dimension"], query=q["query"], judge_question=pl["judge_question"],
                        oracle=len(oracle), fast=len(fast), fast_recall_of_oracle=len(fast & oracle) / max(len(oracle), 1),
                        seed_judged_yes=bool(P[seed_pos] >= 0.7), seed_in_fast=seed_pos in fast, judge_calls=SJ.calls,
                        bound=r.cert["recall_lower"] if r.cert else None))
    df = pd.DataFrame(res)
    print(df.groupby("dimension").agg(queries=("k", "count"), fast_recall=("fast_recall_of_oracle", "mean"),
                                      seed_yes=("seed_judged_yes", "mean"), seed_found=("seed_in_fast", "mean"),
                                      oracle_median=("oracle", "median")).round(2).to_string())
    print("ALL:", df.fast_recall_of_oracle.mean().round(3), "seed judged yes", df.seed_judged_yes.mean().round(3))
    df.to_json("eval/results_general.json", orient="records", indent=1)


if __name__ == "__main__":
    {"gen": lambda: gen(), "oracle": lambda: oracle(int(sys.argv[2])), "analyze": analyze}[sys.argv[1]]()
