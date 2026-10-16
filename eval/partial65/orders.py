"""RESULTS 65: WRITING ORDERS for the phone's facts writer (which photos get a facts line first). Public bench only.

Only signals the phone has BEFORE any facts line exist are used (capture time, photo id). Per library, every photo gets
a rank 0..n-1 in each order; at coverage c the photos with rank < ceil(c x n) have a facts line, the rest do not.
  N  newest-first: capture time descending (what a phone indexer naturally does: the camera roll's newest photos);
     undated photos last, among themselves in the random order R.
  R  random: walk_key(65, id) ascending (person_first.walk_key = fmix64(fnv1a64("65|<id>")), ties by id).
  E  one photo per EVENT first: events = agent.events (new event after a > 3 h gap; every undated photo is its own
     event); events ordered newest-first by their newest photo (undated singletons after all dated events, in R order);
     round-robin: pass 1 takes one photo of every event, pass 2 a second photo of every event that has one, ...;
     inside an event photos are taken in R order.
  python eval/partial65/orders.py  -> eval/partial65/data/orders.parquet (user, split, item_id, rank_N, rank_R, rank_E)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SEED = 65


def orders_for(sub: pd.DataFrame) -> dict:
    from findpics.agent import events
    from findpics.person_first import walk_key
    ids = sub.item_id.astype(str).tolist()
    key = np.array([walk_key(SEED, i) for i in ids], dtype=object)
    rnd = sorted(range(len(ids)), key=lambda k: (key[k], ids[k]))
    rpos = np.empty(len(ids), int); rpos[rnd] = np.arange(len(ids))
    t = pd.to_datetime(sub.taken, utc=True, errors="coerce", format="ISO8601").reset_index(drop=True)
    tn = np.array([x.value if not pd.isna(x) else np.nan for x in t], float)
    dated = ~np.isnan(tn)
    # N
    nd = sorted(np.where(dated)[0], key=lambda k: (-tn[k], rpos[k]))
    nu = sorted(np.where(~dated)[0], key=lambda k: rpos[k])
    N = nd + nu
    # E
    ev = events(sub.taken.reset_index(drop=True))
    per = {}
    for k in rnd:
        per.setdefault(int(ev[k]), []).append(k)
    def ekey(e):
        m = per[e]
        newest = np.nanmax(tn[m]) if dated[m].any() else np.nan
        return (0, -newest, rpos[m[0]]) if not np.isnan(newest) else (1, 0, rpos[m[0]])
    evs = sorted(per, key=ekey)
    E, r = [], 0
    while len(E) < len(ids):
        E += [per[e][r] for e in evs if len(per[e]) > r]
        r += 1
    out = {}
    for name, o in (("N", N), ("R", rnd), ("E", E)):
        assert sorted(o) == list(range(len(ids)))
        rk = np.empty(len(ids), int); rk[np.array(o)] = np.arange(len(ids))
        out[name] = rk
    return out


def main():
    from findpics import store
    sp = json.loads((ROOT / "eval/complex_bench/splits.json").read_text())
    split_of = {u: k for k in ("train", "dev") for u in sp[k]}          # TEST libraries never touched
    idx = store.load(ROOT / "data/public/index_disbench")
    it = idx.items.copy()
    it["item_id"] = it.item_id.astype(str)
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    it = it[it.user.isin(split_of)]
    rows = []
    for u, sub in it.groupby("user"):
        o = orders_for(sub)
        rows.append(pd.DataFrame(dict(user=u, split=split_of[u], item_id=sub.item_id.to_numpy(),
                                      rank_N=o["N"], rank_R=o["R"], rank_E=o["E"])))
    D = pd.concat(rows, ignore_index=True)
    out = ROOT / "eval/partial65/data"; out.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out / "orders.parquet")
    ev_n = {u: len(set(__import__("findpics.agent", fromlist=["events"]).events(s.taken.reset_index(drop=True))))
            for u, s in it.groupby("user")}
    print(D.groupby("split").size().to_dict(), "libraries", D.user.nunique(),
          "events per library median", int(np.median(list(ev_n.values()))))


if __name__ == "__main__":
    main()
