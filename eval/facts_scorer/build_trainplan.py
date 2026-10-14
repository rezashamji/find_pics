"""RESULTS 59 fix B: eval/facts_scorer/data/oracle_plan/*.parquet (27B on the shipped planner's TRAIN questions) ->
data/pairs_trainplan.parquet in pairs_train's columns (scorer reads the question as the product asks it; no QNORM)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import scorer_question  # noqa: E402

D = Path("eval/facts_scorer/data")
P = []
for f in sorted((D / "oracle_plan").glob("*.parquet")):
    x = pd.read_parquet(f); x["user"] = f.stem; P.append(x)
P = pd.concat(P, ignore_index=True)
P["item_id"] = P.item_id.astype(str)
P["person"] = False
P["qs"] = P.q.map({q: scorer_question(q) for q in P.q.unique()})
P = P.drop_duplicates(["user", "item_id", "q"])
P[["item_id", "q", "p", "user", "person", "qs"]].to_parquet(D / "pairs_trainplan.parquet")
print(f"pairs_trainplan: {len(P)} pairs, {P.user.nunique()} libraries, {P.q.nunique()} questions, {(P.p >= .8).sum()} >= .8")
