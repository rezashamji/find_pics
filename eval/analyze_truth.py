"""Adjudicated truth = human label, overridden where the 27B AND the 9B both disagree with it (two models against one
noisy label). Then: judge precision/recall vs adjudicated truth, for the plain 9B question and (if present) the
planner-defined question (eval/question_defs/<qid>.parquet). Also writes the per-photo outcomes so a sample of each
outcome can be looked at before any number is used.
Usage (CPU): PYTHONPATH=src python eval/analyze_truth.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rows = []
    for f in sorted(Path("eval/adjudicate").glob("Qwen3.5-27B_*.parquet")):
        qid = f.stem.split("_", 1)[1]
        adj = pd.read_parquet(f).set_index("item_row")
        qd = Path(f"eval/question_defs/{qid}.parquet")
        base = pd.read_parquet(qd).set_index("item_row") if qd.exists() else None
        if base is None:
            continue
        truth = base.label.astype(bool).copy()
        flip = adj[(adj.p_big >= 0.7) == (adj.p_9b >= 0.7)]          # both models agree with each other, against the label
        truth.loc[flip.index] = flip.p_big >= 0.7
        for name, col in (("plain", "p_plain"), ("defined", "p_defined")):
            j = base[col] >= 0.7; y = truth
            tp = int((j & y).sum())
            rows.append(dict(concept=qid, question=name, labeled=len(y), labels_overridden=len(flip),
                             precision=tp / max(int(j.sum()), 1), recall=tp / max(int(y.sum()), 1),
                             false_yes=int((j & ~y).sum()), missed=int((~j & y).sum())))
        base.assign(truth=truth, overridden=base.index.isin(flip.index)).to_parquet(f"eval/adjudicate/truth_{qid}.parquet")
    df = pd.DataFrame(rows)
    if len(df):
        print(df.round(3).to_string(index=False))
        print(df.groupby("question")[["precision", "recall"]].median().round(3).to_string())
        df.to_json("eval/results_truth.json", orient="records", indent=1)
    else:
        print("nothing to analyze yet (needs eval/adjudicate/Qwen3.5-27B_*.parquet and eval/question_defs/*.parquet)")


if __name__ == "__main__":
    main()
