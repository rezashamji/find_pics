"""RESULTS 59d design diagnostic, TRAIN ONLY: the millisecond scorer's P (towervec12B on 4B phone facts, FOLD models: a
library is scored only by the model that never trained on it) for EVERY (photo, question) pair the all-judge TRAIN stack
asked (train_passthrough pairs, with the 9B's P), so the R59c losses in h can be split into (i) scorer-routed questions
the scorer answers differently from the 9B, by the question's role in the plan (album question / filter / exclude /
anchor), and (ii) judge-routed questions whose 5%-of-scope budget is too small (the 9B's yes rate inside the budget).
  python eval/facts_scorer/r59d/diag_ms.py  -> eval/facts_scorer/r59d/train_ms_pairs.parquet  (GPU job; public data)
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "eval/facts_scorer")
from data import scorer_question  # noqa: E402

M = "eval/facts_scorer/models"
FACTS = "eval/facts_scorer/data/facts_q3vl4bq4rm.parquet"


def main():
    from scorers import Scorer
    from train import b16_image_vectors, b16_text, train_folds
    L = pd.concat([pd.read_parquet(f) for f in glob.glob("eval/facts_scorer/replay/train_passthrough/pairs*.parquet")])
    L = L.drop_duplicates(["qid", "item_id", "question", "crop"]).reset_index(drop=True)
    L["user"] = L.qid.str.split(":").str[0]
    FT = pd.read_parquet(FACTS)
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    IMG = b16_image_vectors(set(L.item_id.astype(str)))
    folds = train_folds()
    out = []
    for fold in sorted(set(folds.values())):
        S = Scorer(f"{M}/towervec12B_q3vl4bq4r_h{fold}", facts=facts, img=IMG, qvec_fn=b16_text)
        sub = L[L.user.map(folds) == fold].copy()
        sub["p_ms"] = S.p(sub.item_id.astype(str).tolist(), [scorer_question(q) for q in sub.question])
        out.append(sub)
        print("fold", fold, len(sub), flush=True)
    R = pd.concat(out)
    R.to_parquet("eval/facts_scorer/r59d/train_ms_pairs.parquet")
    print("DIAG_DONE", len(R), flush=True)


if __name__ == "__main__":
    main()
