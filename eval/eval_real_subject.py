"""Real-library failure (eval_everyday eye audit 10-05: 52/80 right): (a) depictions accepted as the real thing (mural
bike, cartoon van, toy cars, drawn boat: 5 of 19 wrong); (b) "X photos" accepting photos that only share the setting
(coastal town as beach, distant dome as church, golden light as sunset).
Two fixed, generic question forms (no per-concept LLM lists: those cut recall, section 12):
  real    : "Is there a real X anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
  subject : "Is X the main subject of this photo?"
Measured on
  1. human-labeled Open Images photos (testlib), "with X" concepts: precision/recall plain vs real (recall cost);
  2. the everyday exhaustive results on real Flickr libraries: how many returned photos each form drops, and on the 80
     eye-labeled ones (eval/everyday/eye_labels.json) how many wrong / right ones it drops.
Usage (vLLM env, GPU): python eval/eval_real_subject.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REAL = "Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
SUBJECT = "Is {x} the main subject of this photo?"
WITH = {"photos with a dog": "dog", "photos with a car": "car", "photos with a bicycle": "bicycle",
        "photos with a boat": "boat"}
SUBJ = {"food photos": "food", "photos of a cat": "a cat", "beach photos": "a beach", "photos of flowers": "flowers",
        "sunset photos": "a sunset", "photos of a church": "a church"}
OI = ["Dog", "Cat", "Bicycle", "Horse", "Guitar", "Cake", "Pizza", "Coffee cup", "Wine glass", "Sunglasses"]
OUT = Path("eval/real_subject")


def pr(p, y):
    j = p >= 0.7; tp = int((j & y).sum())
    return dict(precision=round(tp / max(int(j.sum()), 1), 3), recall=round(tp / max(int(y.sum()), 1), 3),
                false_yes=int((j & ~y).sum()), missed=int((~j & y).sum()), labeled=int(len(y)))


def main():
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    OUT.mkdir(parents=True, exist_ok=True)
    J = VLLMJudge(gpu_mem=0.85)
    res = {"open_images": {}, "everyday": {}}
    # 1. labeled photos: plain (stored oracle answers) vs real
    idx = store.load("data/public/index_testlib")
    row = {u: i for i, u in enumerate(idx.items.item_id)}
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    for c in OI:
        g = gt.get(f"concept:{c}")
        if not g:
            continue
        lab = {row[u]: 1 for u in g["pos"] if u in row} | {row[u]: 0 for u in g["neg"] if u in row}
        rows = np.array(sorted(lab), int); y = np.array([lab[r] for r in rows], bool)
        qid = c.lower().replace(" ", "_")
        plain = pd.read_parquet(f"eval/oracle/{qid}.parquet").set_index("item_row").p.reindex(rows).to_numpy()
        real = np.array(_judge_rows(idx, J, rows, np.full(len(rows), -1), REAL.format(x=c.lower()), batch=96))
        res["open_images"][c] = dict(plain=pr(plain, y), real=pr(real, y))
        print(c, res["open_images"][c], flush=True)
    # 2. everyday real libraries: re-judge the exhaustive (plain-question) results
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    every = json.load(open("eval/everyday/all.json"))
    eye = json.load(open("eval/everyday/eye_labels.json"))
    for q, x in list(WITH.items()) + list(SUBJ.items()):
        ids = sorted({i for r in every if r["query"] == q for i in r["exhaustive"]})
        rows = np.array([drow[i] for i in ids], int)
        form = REAL if q in WITH else SUBJECT
        p = np.array(_judge_rows(di, J, rows, np.full(len(rows), -1), form.format(x=x), batch=96))
        keep = dict(zip(ids, (p >= 0.7).tolist()))
        e = eye.get(q, {})
        drop = {k: sum(1 for i, v in e.items() if v == k and not keep.get(i, True)) for k in ("R", "W", "U")}
        tot = {k: sum(1 for v in e.values() if v == k) for k in ("R", "W", "U")}
        res["everyday"][q] = dict(question=form.format(x=x), returned=len(ids), kept=int(sum(keep.values())),
                                  eye_dropped=drop, eye_total=tot)
        print(q, res["everyday"][q], flush=True)
        pd.DataFrame(dict(item_id=ids, p=p)).to_parquet(OUT / f"{q.replace(' ', '_')}.parquet")
    json.dump(res, open(OUT / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
