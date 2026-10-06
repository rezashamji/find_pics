"""Question variants for the weakest everyday queries (eye audits 10-05/06: sunset 5/12, food 5/12, dog 7/12, cat
10/12; errors = golden light as sunset, "person holding a pastry" as food, tiger / teddy bear / deer as cat / dog).
Each variant re-judges every photo the 9B returned for the query (8 + 16 libraries) and is scored on EVERY eye label we
have (four audits): right photos it drops (recall cost) vs wrong photos it drops (precision gain), plus how many
unlabeled photos it drops (which must then be looked at before adopting anything).
Usage (vLLM env, GPU): python eval/eval_question_variants.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = {
    "sunset photos": ["Is this a photo of a sunset?",
                      "Is the sun setting or just set in this photo, with a sunset sky (not just warm or golden light)?"],
    "food photos": ["Is this a photo of food?",
                    "Is food what this photo is mainly about (a dish, meal or snack), not just present in the scene?"],
    "photos with a dog": ["Is there a real dog anywhere in this photo (not a drawing, painting, statue, toy, model or "
                          "picture of one)?",
                          "Is there a real, living dog anywhere in this photo (not another animal, a toy, a stuffed "
                          "animal, a drawing or a statue)?"],
    "photos of a cat": ["Is this a photo of a cat?",
                        "Is there a real house cat in this photo (not a wild cat like a tiger or lion, a toy, a "
                        "drawing or a statue)?"],
}
import os
if os.environ.get("FP_QV_SET") == "generic":   # generic forms (no per-word lists), every query: what would ship
    X = {"photos with a dog": "dog", "photos with a car": "car", "photos with a bicycle": "bicycle",
         "photos with a boat": "boat"}
    S = {"food photos": "food", "photos of a cat": "a cat", "beach photos": "a beach", "photos of flowers": "flowers",
         "sunset photos": "a sunset", "photos of a church": "a church"}
    REAL = ("Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?")
    REAL2 = ("Is there a real {x} anywhere in this photo (not something that only looks like one, a toy, a stuffed "
             "animal, a drawing, a painting, a statue, a model or a picture of one)?")
    SUBJ0 = "Is this a photo of {x}?"
    SUBJ1 = "Is {x} what this photo is mainly about, not just something present in the scene?"
    VARIANTS = {q: [REAL.format(x="a " + x), REAL2.format(x="a " + x)] for q, x in X.items()}
    VARIANTS |= {q: [SUBJ0.format(x=x), SUBJ1.format(x=x), REAL2.format(x=x)] for q, x in S.items()}
RUNS = ["eval/everyday_v2", "eval/everyday16_9b"]
OUT = Path("eval/question_variants" + ("_generic" if os.environ.get("FP_QV_SET") == "generic" else ""))
NORM = {"R": "right", "W": "wrong", "U": "unsure", "right": "right", "wrong": "wrong", "unsure": "unsure",
        "match": "right", "no match": "wrong"}


def labels():
    lab = {}
    for q, d in json.load(open("eval/everyday/eye_labels.json")).items():
        for i, v in d.items():
            lab[(q, str(i))] = NORM[v]
    for f in ["eval/everyday_v2/eye_labels.json", "eval/everyday16_audit/eye_labels.json",
              "eval/everyday_q_audit/eye_labels.json", "eval/everyday_4b/eye_labels.json",
              "eval/everyday16_q3vl_audit/eye_labels.json"]:
        if Path(f).exists():
            d = json.load(open(f))
            for r in (d["labels"] if isinstance(d, dict) else d):   # newer audits: {"protocol", "labels"}
                if r.get("label") in NORM:
                    lab.setdefault((r["query"], str(r["item_id"])), NORM[r["label"]])
    return lab


def main():
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for d in RUNS for f in sorted(glob.glob(f"{d}/part*.json")) for r in json.load(open(f))]
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    lab = labels()
    J = VLLMJudge(gpu_mem=0.85)
    res = {}
    for q, qs in VARIANTS.items():
        ids = sorted({i for r in rows if r["query"] == q for i in r["exhaustive"]} | {i for (qq, i) in lab if qq == q})
        ids = [i for i in ids if i in drow]
        rr = np.array([drow[i] for i in ids], int)
        df = pd.DataFrame(dict(item_id=ids))
        for k, question in enumerate(qs):
            df[f"v{k}"] = _judge_rows(di, J, rr, np.full(len(rr), -1), question, batch=96)
        df.to_parquet(OUT / f"{q.replace(' ', '_')}.parquet")
        L = {i: v for (qq, i), v in lab.items() if qq == q}
        res[q] = {}
        for k, question in enumerate(qs):
            keep = dict(zip(ids, (df[f"v{k}"] >= 0.7).tolist()))
            res[q][f"v{k}"] = dict(question=question, kept=int(sum(keep.values())), of=len(ids),
                                   labeled_kept={c: sum(1 for i, v in L.items() if v == c and keep.get(i)) for c in ("right", "wrong", "unsure")},
                                   labeled={c: sum(1 for v in L.values() if v == c) for c in ("right", "wrong", "unsure")})
            print(q, f"v{k}", res[q][f"v{k}"], flush=True)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
