"""RESULTS 66: the place + taken PREFIX of a facts line, built exactly as eval/facts_scorer/data.py (--meta=1) builds it
for the writer lines (place = reverse-geocoded address parts without digits, de-duplicated; taken = "%B %Y" of the capture
time); the phone has both from day 1. prefixes() -> {item_id: "place: .... taken: Month YYYY." | ""} (trailing space
stripped; "" = no place and no date).
  python eval/partial66/prefix.py   -> checks every writer line in the 61d facts files starts with its photo's prefix
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def prefixes():
    it = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet", columns=["item_id", "place", "taken"])
    it["item_id"] = it.item_id.astype(str)
    t_ = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")
    month = dict(zip(it.item_id, t_.dt.strftime("%B %Y")))
    out = {}
    for iid, pl in zip(it.item_id, it.place):
        parts = [p_.strip() for p_ in str(pl or "").split(",") if p_.strip() and not re.search(r"\d", p_)]
        parts = list(dict.fromkeys(parts))
        s = (f"place: {', '.join(parts)}. " if parts else "") + (f"taken: {month[iid]}. " if isinstance(month.get(iid), str) else "")
        out[iid] = s.rstrip()
    return out


if __name__ == "__main__":
    P = prefixes()
    for f in ("facts_w_t2b_final_rs2_xfitm", "facts_w_p2b_devmix_rs2m"):
        T = pd.read_parquet(ROOT / f"eval/facts_scorer/data/{f}.parquet")
        ok = sum(t.startswith(P[str(i)]) for i, t in zip(T.item_id, T.text))
        emp = sum(P[str(i)] == "" for i in T.item_id)
        pl = sum(P[str(i)].startswith("place:") for i in T.item_id)
        tk = sum("taken:" in P[str(i)] for i in T.item_id)
        print(f"{f}: {ok}/{len(T)} lines start with their prefix; prefix empty {emp}, with place {pl}, with taken {tk}; by split "
              f"{T.assign(e=[P[str(i)] == '' for i in T.item_id]).groupby('split').e.agg(['sum', 'size']).to_dict('index')}")
    sys.exit(0)
