"""Selfie question (eye audit 10-05: "Is this a selfie?" accepted any close face; 9B 4/10 right, 4B-only 0/6).
Candidates ask WHO took the photo. Scored on every photo the 9B (everyday_v2) or 4B (everyday_4b) run returned for
"selfies"; the dropped/kept sets are then audited by eye (eval/selfie_q/).
Usage (vLLM env, GPU): [FP_VLM_MODEL=Qwen/Qwen3.5-4B] python eval/eval_selfie_q.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

QS = {"old": "Is this a selfie?",
      "who": "Is this a selfie, taken by a person in the photo (camera held at arm's length or in a mirror), not a photo "
             "someone else took of them?",
      "took": "Was this photo taken by one of the people in it (a selfie: arm's length or a mirror), rather than by "
              "someone else?"}
OUT = Path("eval/selfie_q")


def main():
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted({i for t in ("everyday_v2", "everyday_4b") for f in glob.glob(f"eval/{t}/part*.json")
                  for r in json.load(open(f)) if r["query"] == "selfies" for i in r["exhaustive"]})
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    rows = np.array([drow[i] for i in ids], int)
    J = VLLMJudge(gpu_mem=0.85)
    df = pd.DataFrame(dict(item_id=ids))
    for k, q in QS.items():
        df[k] = _judge_rows(di, J, rows, np.full(len(rows), -1), q, batch=96)
        print(k, "kept", int((df[k] >= 0.7).sum()), "of", len(df), flush=True)
    m = os.environ.get("FP_VLM_MODEL", "Qwen/Qwen3.5-9B").split("/")[-1]
    df.to_parquet(OUT / f"scores_{m}.parquet")


if __name__ == "__main__":
    main()
