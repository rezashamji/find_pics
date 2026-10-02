"""Phone-size judge on the WHOLE test library (real prevalence), to compare with the 9B oracle (eval/oracle/<qid>.parquet).
The balanced-sample test (eval_judge_size.py) cannot show how many false yeses a small judge adds when ~95% of photos are
non-matches. Same questions as eval_oracle.py. Output: eval/judge_size/full_<model>_<qid>.parquet (item_row, p).
Usage (vLLM env, fits a 20 GB GPU): python eval/eval_small_oracle.py Qwen/Qwen3.5-2B bread christmas_tree sunglasses dog
Analysis (CPU): compare p>=cut with the 9B's p>=0.7 per item; look at disagreements at full resolution.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    sys.path.insert(0, "eval")
    from eval_oracle import IDX, QUERIES
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    model, qids = sys.argv[1], sys.argv[2:]
    idx = store.load(IDX)
    J = VLLMJudge(model=model, gpu_mem=0.85)
    rows = np.arange(idx.n_items)
    for Q in QUERIES:
        if Q["qid"] not in qids:
            continue
        tag = model.split('/')[-1] + (f"-{os.environ['FP_QUANT']}" if os.environ.get("FP_QUANT") else "")
        out = Path(f"eval/judge_size/full_{tag}_{Q['qid']}.parquet")
        if out.exists():
            continue
        t = time.time()
        p = _judge_rows(idx, J, rows, np.full(len(rows), -1), Q["q"], batch=96)
        pd.DataFrame(dict(item_row=rows, p=p)).to_parquet(out)
        print(f"{model} {Q['qid']}: yes@0.7={int((p >= .7).sum())} yes@0.3={int((p >= .3).sum())} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
