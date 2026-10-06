"""Test the distilled phone judge against the base 4B and the 9B teacher.
(a) held-out QUESTIONS (10%, eval/train_judge_lora.split_of): agreement with the 9B's yes (P >= 0.7) and mean |P - P9|;
(b) the everyday eye labels (all audits, eval/eval_question_variants.labels): with each query's v2 question, how many
    eye-RIGHT photos each judge keeps (recall) and eye-WRONG photos it keeps (false positives). Eye labels are the truth
    here, not the 9B.
Usage (vLLM env, GPU): python eval/eval_judge_distill.py gen <name> <model>   (one model per process)
                       python eval/eval_judge_distill.py report
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "eval")
OUT = Path("eval/judge_distill_test")
QUESTION = {"food photos": "Is this a photo of food?", "photos of a cat": "Is this a photo of a cat?",
            "beach photos": "Is this a photo of a beach?", "photos of flowers": "Is this a photo of flowers?",
            "sunset photos": "Is this a photo of a sunset?", "photos of a church": "Is this a photo of a church?"}
for q, x in {"photos with a dog": "dog", "photos with a car": "car", "photos with a bicycle": "bicycle",
             "photos with a boat": "boat"}.items():
    QUESTION[q] = f"Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"


def cases():
    from train_judge_lora import split_of
    from eval_question_variants import labels
    held = [json.loads(l) for f in sorted(glob.glob("data/public/judge_distill/part*.jsonl")) for l in open(f)]
    held = [r for r in held if split_of(r["question"]) == "test"]
    lab = [(q, i, v) for (q, i), v in labels().items() if q in QUESTION]
    return held, lab


def gen(name, model):
    import os
    os.environ["FP_VLM_MODEL"] = model
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    held, lab = cases()
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    J = VLLMJudge(model=model, gpu_mem=0.8)
    res = {"held": {}, "eye": {}}
    for q in sorted({r["question"] for r in held}):
        rs = [r for r in held if r["question"] == q and r["item_id"] in drow]
        p = _judge_rows(di, J, np.array([drow[r["item_id"]] for r in rs], int), np.full(len(rs), -1), q, batch=96)
        res["held"].update({f"{q}\t{r['item_id']}": float(x) for r, x in zip(rs, p)})
    for q in QUESTION:
        rs = [(i, v) for (qq, i, v) in lab if qq == q and i in drow]
        p = _judge_rows(di, J, np.array([drow[i] for i, _ in rs], int), np.full(len(rs), -1), QUESTION[q], batch=96)
        res["eye"].update({f"{q}\t{i}": float(x) for (i, _), x in zip(rs, p)})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{name}.json", "w"))


def report():
    held, lab = cases()
    p9 = {f"{r['question']}\t{r['item_id']}": r["p9"] for r in held}
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        ks = [k for k in d["held"] if k in p9]
        agree = sum((d["held"][k] >= 0.7) == (p9[k] >= 0.7) for k in ks)
        mae = np.mean([abs(d["held"][k] - p9[k]) for k in ks]) if ks else float("nan")
        kept = {c: 0 for c in ("right", "wrong", "unsure")}; tot = dict(kept)
        for q, i, v in lab:
            k = f"{q}\t{i}"
            if k in d["eye"]:
                tot[v] += 1; kept[v] += d["eye"][k] >= 0.7
        print(f"{f.stem:14s} held-out: agree with 9B {agree}/{len(ks)}  mean|P-P9| {mae:.3f} | eye labels kept: "
              f"right {kept['right']}/{tot['right']}  wrong {kept['wrong']}/{tot['wrong']}  unsure {kept['unsure']}/{tot['unsure']}")


if __name__ == "__main__":
    gen(sys.argv[2], sys.argv[3]) if sys.argv[1] == "gen" else report()
