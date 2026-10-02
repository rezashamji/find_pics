"""Can a phone-sized judge replace the 9B? For each finished oracle concept (eval/oracle/*.parquet, judged by
Qwen3.5-9B on all 19,218 items), sample every 9B-yes item (cap 300) + an equal number of random 9B-no items, and judge
them with Qwen3.5-4B and Qwen3.5-2B. Report agreement with the 9B (yes-recall, false-yes) and, where Open Images
labels exist, accuracy vs human labels. Usage (vLLM env, GPU): python eval/eval_judge_size.py <model_id>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    model = sys.argv[1]
    idx = store.load("data/public/index_testlib")
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    row = {u: i for i, u in enumerate(idx.items.item_id)}
    J = VLLMJudge(model=model, gpu_mem=float(os.environ.get('FP_GPU_MEM', '0.7')))
    rng = np.random.default_rng(0); res = {}
    for f in sorted(Path("eval/oracle").glob("*.parquet")):
        qid = f.stem; d = pd.read_parquet(f); P9 = d.p.to_numpy()
        yes = np.where(P9 >= 0.7)[0]; no = np.where(P9 < 0.7)[0]
        yes = rng.choice(yes, min(300, len(yes)), replace=False); no = rng.choice(no, len(yes), replace=False)
        rows = np.r_[yes, no]
        c = qid.replace("_", " ")
        q = f"Is there {'a ' if c not in ('bread', 'baked goods') else ''}{c} visible in this image?"
        p = _judge_rows(idx, J, rows, np.full(len(rows), -1), q, batch=96)
        small_yes = p >= 0.7; big_yes = P9[rows] >= 0.7
        Path("eval/judge_size").mkdir(exist_ok=True)   # per-item answers, for looking at what the small model rejects
        pd.DataFrame(dict(item_row=rows, p_small=p, p_9b=P9[rows])).to_parquet(f"eval/judge_size/{model.split('/')[-1]}_{qid}.parquet")
        r = dict(n=int(len(rows)), agree=float((small_yes == big_yes).mean()),
                 recall_of_9b_yes=float(small_yes[big_yes].mean()), false_yes_on_9b_no=float(small_yes[~big_yes].mean()))
        key = f"concept:{c.capitalize() if c != 'baked goods' else 'Baked goods'}"
        if key in gt:
            pos = set(row[u] for u in gt[key]["pos"] if u in row); neg = set(row[u] for u in gt[key]["neg"] if u in row)
            lab = np.array([1 if x in pos else 0 if x in neg else -1 for x in rows])
            m = lab >= 0
            if m.sum():
                r["acc_vs_labels_small"] = float((small_yes[m] == (lab[m] == 1)).mean())
                r["acc_vs_labels_9b"] = float((big_yes[m] == (lab[m] == 1)).mean()); r["n_labeled"] = int(m.sum())
        res[qid] = r; print(model, qid, r, flush=True)
    Path("eval/judge_size").mkdir(exist_ok=True)
    json.dump(res, open(f"eval/judge_size/{model.split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
