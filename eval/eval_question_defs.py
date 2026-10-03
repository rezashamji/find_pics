"""Failure #1 (judge over-calls: conifers -> "christmas tree", eyeglasses -> "sunglasses", crepes -> "bread").
General mechanism, not per-word tuning: the planner LLM writes the yes/no question WITH a one-line definition: what
counts, and the common look-alikes that do not. Compare on every human-labeled photo (Open Images) per concept:
  plain   : the oracle question ("Is there a christmas tree visible in this image?")
  defined : LLM-written question with what counts / what does not (same LLM as the product planner, no hand edits)
Precision/recall vs labels; labels are noisy (10-03 look), so disagreements get the 27B adjudication + a raw look.
Output: eval/question_defs/<qid>.parquet (item_row, label, p_plain, p_defined) + questions.json.
Usage (vLLM env, GPU): python eval/eval_question_defs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEF_PROMPT = """Write ONE yes/no question for a vision model that looks at a single photo, to decide whether the photo
shows: {concept}.
Include a short definition inside the question: what counts, and the most common look-alikes that do NOT count.
Return only the question, one line, ending with a question mark."""


def main():
    sys.path.insert(0, "eval")
    from eval_oracle import IDX, QUERIES
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    idx = store.load(IDX)
    row = {u: i for i, u in enumerate(idx.items.item_id)}
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    out = Path("eval/question_defs"); out.mkdir(exist_ok=True)
    J = VLLMJudge(gpu_mem=0.85)
    qs = {}
    for Q in QUERIES:
        if not Q["gt"] or Q["gt"] not in gt:
            continue
        concept = Q["qid"].replace("_", " ")
        defined = J.text(DEF_PROMPT.format(concept=concept), max_tokens=120).strip().splitlines()[0].strip()
        qs[Q["qid"]] = dict(plain=Q["q"], defined=defined)
        lab = {row[u]: 1 for u in gt[Q["gt"]]["pos"] if u in row} | {row[u]: 0 for u in gt[Q["gt"]]["neg"] if u in row}
        rows = np.array(sorted(lab), int); y = np.array([lab[r] for r in rows], bool)
        p9 = pd.read_parquet(f"eval/oracle/{Q['qid']}.parquet").set_index("item_row").p.reindex(rows).to_numpy()
        pdf = _judge_rows(idx, J, rows, np.full(len(rows), -1), defined, batch=96)
        pd.DataFrame(dict(item_row=rows, label=y, p_plain=p9, p_defined=pdf)).to_parquet(out / f"{Q['qid']}.parquet")
        for name, p in (("plain", p9), ("defined", pdf)):
            j = p >= 0.7; tp = (j & y).sum()
            print(f"{Q['qid']:15s} {name:8s} precision {tp / max(j.sum(), 1):.3f} recall {tp / max(y.sum(), 1):.3f} "
                  f"(false yes {int((j & ~y).sum())}, missed {int((~j & y).sum())}, labeled {len(y)})", flush=True)
        print(f"   defined question: {defined}", flush=True)
    json.dump(qs, open(out / "questions.json", "w"), indent=1)


if __name__ == "__main__":
    main()
