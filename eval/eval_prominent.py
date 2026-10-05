"""'X photos' that only share the setting (eye audit v2 10-05: flowers 4/10 right, the subject was a fountain / car /
bandstand with a flower bed behind). "Is X the main subject?" fixed the wrong ones but dropped half the right flower
photos (RESULTS 26). Middle form tested here:
  PROMINENT: "Is {x} clearly visible as a prominent part of this photo (not only a small part of the background)?"
Re-judged on the 9B v2 exhaustive results of the scene/subject queries; scored on every eye-labeled photo
(v1: eval/everyday/eye_labels.json, v2: eval/everyday_v2/eye_labels.json): how many right / wrong ones it drops.
Usage (vLLM env, GPU): python eval/eval_prominent.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROM = "Is {x} clearly visible as a prominent part of this photo (not only a small part of the background)?"
SUBJ = {"food photos": "food", "beach photos": "a beach", "photos of flowers": "flowers", "sunset photos": "a sunset",
        "photos of a church": "a church", "photos of a cat": "a cat"}
OUT = Path("eval/prominent")


def eye():
    lab = {}
    for q, d in json.load(open("eval/everyday/eye_labels.json")).items():
        for i, v in d.items():
            lab[(q, i)] = {"R": "right", "W": "wrong", "U": "unsure"}[v]
    for r in json.load(open("eval/everyday_v2/eye_labels.json")):
        lab[(r["query"], r["item_id"])] = r["label"]
    return lab


def main():
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for f in sorted(glob.glob("eval/everyday_v2/part*.json")) for r in json.load(open(f))]
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    lab = eye()
    J = VLLMJudge(gpu_mem=0.85)
    res = {}
    for q, x in SUBJ.items():
        ids = sorted({i for r in rows if r["query"] == q for i in r["exhaustive"]} | {i for (qq, i) in lab if qq == q})
        p = np.array(_judge_rows(di, J, np.array([drow[i] for i in ids], int), np.full(len(ids), -1), PROM.format(x=x)))
        keep = dict(zip(ids, (p >= 0.7).tolist()))
        lab_q = {i: v for (qq, i), v in lab.items() if qq == q}
        res[q] = dict(returned=len(ids), kept=int(sum(keep.values())),
                      dropped={k: sum(1 for i, v in lab_q.items() if v == k and not keep[i]) for k in ("right", "wrong", "unsure")},
                      labeled={k: sum(1 for v in lab_q.values() if v == k) for k in ("right", "wrong", "unsure")})
        print(q, res[q], flush=True)
        pd.DataFrame(dict(item_id=ids, p=p)).to_parquet(OUT / f"{q.replace(' ', '_')}.parquet")
    json.dump(res, open(OUT / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
