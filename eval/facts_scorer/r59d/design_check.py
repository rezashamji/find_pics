"""RESULTS 59d design check, TRAIN ONLY, no scores: how the R59d rule rewrites every judge question the TRAIN all-judge
stack asked - (a) trip questions: R59c metadata name (engine.place_mask) vs R59d tolerant name, photos matched, and the
remainder; (b) the parts of every OR / AND composition. -> eval/facts_scorer/r59d/design_train.txt"""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/facts_scorer")
from route import TRIP, _TAIL, PlaceIndex, place_candidates, remainder, route, split_parts  # noqa: E402
from findpics.engine import place_mask  # noqa: E402

I = pd.read_parquet("data/public/index_disbench/items.parquet")
I["user"] = I.path.str.extract(r"/images/images/([^/]+)/")[0]
L = pd.concat([pd.read_parquet(f) for f in glob.glob("eval/facts_scorer/replay/train_passthrough/pairs*.parquet")])
L["user"] = L.qid.str.split(":").str[0]
L["cat"] = L.qid.str.split(":").str[1].str[0]
G = L.groupby(["user", "question", "crop"]).agg(n=("item_id", "size"), cats=("cat", lambda s: "".join(sorted(set(s))))).reset_index()
out = []
PI = {}
for (u, q, cr), r in G.set_index(["user", "question", "crop"]).iterrows():
    qb = _TAIL.sub("", q)
    if TRIP.search(qb):
        items = I[I.user == u]
        if u not in PI:
            PI[u] = PlaceIndex(items.place)
        old = next(((c, int(np.asarray(place_mask(items.place, c)).sum())) for c in place_candidates(q)
                    if np.asarray(place_mask(items.place, c)).any()), None)
        new = next(((c, int(PI[u].mask(c).sum())) for c in place_candidates(q) if PI[u].mask(c).any()), None)
        rem = remainder(q, new[0]) if new else None
        out.append(f"TRIP [{r.cats}] {qb}\n    R59c meta: {old}   R59d meta: {new}   remainder: {rem!r}"
                   + (f" -> route {route(rem, cr) or ('trip-no-place' if TRIP.search(_TAIL.sub('', rem)) else 'scorer')}" if rem else ""))
    sp = split_parts(q)
    if sp or " or " in qb or " and also " in qb:
        out.append(f"SPLIT [{r.cats}] {qb}\n    -> {sp[0] if sp else None}: " +
                   (" | ".join(f"{p} [{route(p, cr) or 'scorer'}]" for p in sp[1]) if sp else "NOT SPLIT"))
open("eval/facts_scorer/r59d/design_train.txt", "w").write("\n".join(out) + "\n")
print(len(out), "lines")
