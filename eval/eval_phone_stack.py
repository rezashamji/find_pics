"""The whole phone stack vs today's cluster stack, on a real first round of search (4 concepts, 19,218 test photos).
  cluster stack : PE-Core-L image/text vectors + Qwen3.5-9B judge (16-bit)          -> replayed from eval/oracle (no GPU)
  phone stack   : PE-Core-B-16 vectors (data/public/index_testlib_B) + 9B judge with 4-bit weights (Intel AutoRound)
Both run streaming ROUND 1 (adaptive head + random tail check). Truth for both = the 16-bit 9B judge on EVERY photo
(eval/oracle/<qid>.parquet, yes at 0.7). Reports per stack: found, recall of truth, precision vs truth, judge calls,
stated lower bound. Usage (vLLM env, GPU): python eval/eval_phone_stack.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QIDS = ["bread", "christmas_tree", "sunglasses", "dog"]


def main():
    sys.path.insert(0, "eval")
    import findpics.engine as E
    from findpics import store
    from findpics.models import ImageTextEncoder
    from findpics.planner import AlbumSpec
    from findpics.vlm import VLLMJudge
    from eval_oracle import QUERIES
    Qs = {q["qid"]: q for q in QUERIES}
    idxL = store.load("data/public/index_testlib")
    idxB = store.load("data/public/index_testlib_B", clip_model="hf-hub:timm/PE-Core-B-16")
    encB = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    J4 = VLLMJudge(model="Intel/Qwen3.5-9B-int4-AutoRound", gpu_mem=0.8)
    res = {}
    for qid in QIDS:
        Q = Qs[qid]; d = pd.read_parquet(f"eval/oracle/{qid}.parquet").set_index("item_row")
        P = d.p.reindex(range(idxL.n_items)).fillna(0).to_numpy(); truth = set(np.where(P >= 0.7)[0])
        spec = AlbumSpec(name=qid, looks=Q["looks"], judge_question=Q["q"])
        out = {}
        # cluster stack: replay stored 16-bit answers with the L ranking (no GPU)
        L = d.look.reindex(range(idxL.n_items)).fillna(-9).to_numpy()

        class Stored:
            calls = 0
            def p_yes(self, ims, q):
                Stored.calls += len(ims); return [float(P[int(i)]) for i in ims]
        saved = (E._frame_for, E._boxed, E.look_scores)
        E._frame_for = lambda idx, r, fr: r; E._boxed = lambda idx, im, fr: im
        E.look_scores = lambda idx, enc, looks, avoid, _L=L: _L
        r = E.run_album(idxL, spec, None, Stored(), None)
        E._frame_for, E._boxed, E.look_scores = saved
        out["cluster"] = (set(r.returned.item_row), Stored.calls, r.cert["recall_lower"])

        # phone stack: B-16 ranking + real 4-bit judge calls
        class Count:
            calls = 0
            def p_yes(self, ims, q):
                Count.calls += len(ims); return J4.p_yes(ims, q)
        r = E.run_album(idxB, spec, encB, Count(), None)
        out["phone"] = (set(r.returned.item_row), Count.calls, r.cert["recall_lower"])
        res[qid] = {}
        for k, (found, calls, lo) in out.items():
            tp = len(found & truth)
            res[qid][k] = dict(found=len(found), truth=len(truth), recall=round(tp / max(len(truth), 1), 3),
                               precision=round(tp / max(len(found), 1), 3), judge_calls=calls, stated_lower=round(lo, 3))
        print(qid, json.dumps(res[qid]), flush=True)
    json.dump(res, open("eval/results_phone_stack.json", "w"), indent=1)


if __name__ == "__main__":
    main()
