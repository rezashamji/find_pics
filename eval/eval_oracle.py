"""Oracle = the judge answers the query on EVERY in-scope item. Fast mode = what the product does by default.
  judge   (vLLM env, one query per Slurm array task): score all 19,218 items -> eval/oracle/<qid>.parquet
  analyze (CPU): replay fast mode from those SAME stored scores (same judge, same answers) -> how much of the oracle's
           set does fast mode recover, at what judge cost, and does its stated bound hold vs the oracle?
Usage: python eval/eval_oracle.py judge <task_id> | python eval/eval_oracle.py analyze
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

IDX = "data/public/index_testlib"
OUT = Path("eval/oracle"); OUT.mkdir(parents=True, exist_ok=True)
CONCEPTS = ['Bread', 'Baked goods', 'Pizza', 'Cake', 'Sandwich', 'Dog', 'Bicycle', 'Guitar', 'Christmas tree',
            'Swimming pool', 'Cat', 'Horse', 'Sunglasses', 'Wine glass', 'Coffee cup']
QUERIES = [dict(qid=c.lower().replace(" ", "_"), gt=f"concept:{c}", looks=[f"a photo of {c.lower()}"],
                q=f"Is there {('a ' if not c.endswith('s') and c not in ('Bread', 'Baked goods') else '')}{c.lower()} visible in this image?")
           for c in CONCEPTS]
QUERIES += [
    dict(qid="dog_on_beach", gt=None, looks=["a dog on a beach"], q="Is there a dog on a beach or at the sea shore in this image?"),
    dict(qid="person_sunglasses", gt=None, looks=["a person wearing sunglasses"], q="Is a person in this image wearing sunglasses?"),
    dict(qid="birthday_candles", gt=None, looks=["a birthday cake with lit candles"], q="Is there a cake with candles on it in this image?"),
    dict(qid="eating_pizza", gt=None, looks=["people eating pizza"], q="Is someone eating pizza in this image?"),
    dict(qid="bike_street", gt=None, looks=["a bicycle on a city street"], q="Is there a bicycle on a city street in this image?"),
]


def judge(task):
    from findpics import store
    from findpics.engine import _judge_rows, look_scores
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    Q = QUERIES[task]
    idx = store.load(IDX)
    enc = ImageTextEncoder(idx.clip_model)
    look = look_scores(idx, enc, Q["looks"], [])     # also sets idx.best_frame_t for videos
    del enc
    import torch; torch.cuda.empty_cache()
    J = VLLMJudge(gpu_mem=0.75)
    rows = np.arange(idx.n_items)
    t = time.time()
    p = _judge_rows(idx, J, rows, np.full(len(rows), -1), Q["q"], batch=96)
    pd.DataFrame(dict(item_row=rows, p=p, look=look)).to_parquet(OUT / f"{Q['qid']}.parquet")
    print(f"{Q['qid']}: judged {len(rows)} in {time.time()-t:.0f}s, yes@0.7={int((p >= 0.7).sum())}")


def analyze():
    import findpics.engine as E
    from findpics import store
    from findpics.planner import AlbumSpec
    idx = store.load(IDX)
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    row = {u: i for i, u in enumerate(idx.items.item_id)}
    E._frame_for = lambda idx, r, fr: r
    E._boxed = lambda idx, im, fr: im
    out = {}
    print(f"{'query':18s} {'oracle':>6s} {'fast∩oracle':>11s} {'gap':>5s} {'judge calls fast/oracle':>24s} {'bound':>6s} {'oracle-recall true':>6s}")
    for Q in QUERIES:
        f = OUT / f"{Q['qid']}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f); P = d.p.to_numpy(); L = d.look.to_numpy()
        oracle = set(np.where(P >= 0.7)[0])

        class StoredJudge:
            calls = 0
            def p_yes(self, ims, q):
                StoredJudge.calls += len(ims); return [float(P[int(i)]) for i in ims]
        E.look_scores = lambda idx, enc, looks, avoid, _L=L: _L
        res = E.run_album(idx, AlbumSpec(name=Q["qid"], looks=Q["looks"], judge_question=Q["q"]), None, StoredJudge(), None)
        fast = set(res.returned.item_row)
        rec = len(fast & oracle) / max(len(oracle), 1)
        holds = res.cert["recall_lower"] <= rec + 1e-9 if res.cert else None
        r = dict(oracle=len(oracle), fast=len(fast), fast_and_oracle=len(fast & oracle), fast_recall_of_oracle=rec,
                 judge_calls_fast=StoredJudge.calls, judge_calls_oracle=int(len(P)), stated_lower=res.cert["recall_lower"] if res.cert else None,
                 bound_holds_vs_oracle=holds)
        if Q["gt"]:
            pos = set(row[u] for u in gt[Q["gt"]]["pos"] if u in row)
            r.update(oracle_recall_vs_labels=len(oracle & pos) / len(pos), fast_recall_vs_labels=len(fast & pos) / len(pos))
        out[Q["qid"]] = r
        print(f"{Q['qid']:18s} {len(oracle):6d} {len(fast & oracle):11d} {1-rec:5.2f} {StoredJudge.calls:>12d}/{len(P):<11d} "
              f"{(r['stated_lower'] or 0):6.2f} {r.get('oracle_recall_vs_labels', float('nan')):6.2f}")
    json.dump(out, open("eval/results_oracle_gap.json", "w"), indent=1)


if __name__ == "__main__":
    judge(int(sys.argv[2])) if sys.argv[1] == "judge" else analyze()
