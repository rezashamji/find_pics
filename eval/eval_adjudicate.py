"""Third opinion where the 9B judge and the human Open Images label DISAGREE (neither is truth: 10-03 raw look found the
judge wrong on a crepe but the labels wrong on omelettes/mushrooms labeled "baked goods").
For every labeled photo where (9B p>=0.7) != label, ask a larger open VLM the same question. Output
eval/adjudicate/<model>_<qid>.parquet (item_row, label, p_9b, p_big). Then: adjusted truth = label, overridden where the
big model AND the 9B agree against the label; Claude raw-looks a sample of each outcome before any number is used.
Usage (vLLM env, one 80 GB GPU for 27B bf16): python eval/eval_adjudicate.py Qwen/Qwen3.5-27B
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    sys.path.insert(0, "eval")
    from eval_oracle import IDX, QUERIES
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    model = sys.argv[1]
    idx = store.load(IDX)
    row = {u: i for i, u in enumerate(idx.items.item_id)}
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    out = Path("eval/adjudicate"); out.mkdir(exist_ok=True)
    J = VLLMJudge(model=model, gpu_mem=0.9)
    for Q in QUERIES:
        if not Q["gt"] or Q["gt"] not in gt:
            continue
        f = out / f"{model.split('/')[-1]}_{Q['qid']}.parquet"
        if f.exists():
            continue
        p9 = pd.read_parquet(f"eval/oracle/{Q['qid']}.parquet").set_index("item_row").p
        lab = {row[u]: 1 for u in gt[Q["gt"]]["pos"] if u in row} | {row[u]: 0 for u in gt[Q["gt"]]["neg"] if u in row}
        rows = np.array([r for r, y in lab.items() if (p9[r] >= 0.7) != bool(y)], int)
        if not len(rows):
            continue
        pb = _judge_rows(idx, J, rows, np.full(len(rows), -1), Q["q"], batch=96)
        pd.DataFrame(dict(item_row=rows, label=[lab[r] for r in rows], p_9b=p9[rows].to_numpy(), p_big=pb)).to_parquet(f)
        big_sides_label = ((pb >= 0.7) == np.array([lab[r] for r in rows], bool)).mean()
        print(f"{Q['qid']}: {len(rows)} disagreements; big model sides with the human label on {big_sides_label:.2f}", flush=True)


if __name__ == "__main__":
    main()
