"""Complex-query benchmark, step 0: library splits + per-library metadata digest (events, places, face groups).

Splits are BY LIBRARY (DISBench user). The 24 users already used to tune the product (eval/everyday*: judge cuts,
stop rules, selfie rule) go to TRAIN; the 33 untouched users are split at random (seed 0) into TEST (20) and DEV (13).
(Caveat: the planner prompt was tuned on DISBench's own 122 queries, which span all 57 users; no library content was
used for training.)
Writes eval/complex_bench/splits.json and eval/complex_bench/libraries/<user>.json
Usage (main env, CPU): python eval/complex_bench/prep.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("eval/complex_bench")


def place_tail(addr: str) -> str:
    """'Y, Cherwell Street, ..., Oxford, Oxfordshire, England, OX4 1BQ, United Kingdom' -> 'Oxford, Oxfordshire, England,
    United Kingdom' (postcodes and house-level parts dropped)."""
    if not isinstance(addr, str) or not addr:
        return ""
    parts = [p.strip() for p in addr.split(",") if p.strip() and not re.search(r"\d", p)]
    return ", ".join(parts[-4:])


def main():
    from findpics import store
    from findpics.agent import events
    from findpics.people import face_groups
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")   # no torch import (slow CUDA-lib preload on Lustre)
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    used = set()
    for f in list(Path("eval").glob("everyday/part*.json")) + list(Path("eval").glob("everyday16*/part*.json")):
        used |= {r["user"] for r in json.load(open(f))}
    fresh = sorted(set(user_of) - used)
    rng = np.random.default_rng(0)
    test = sorted(rng.choice(fresh, size=20, replace=False).tolist())
    dev = sorted(set(fresh) - set(test))
    splits = dict(test=test, dev=dev, train=sorted(used),
                  note="by library; train = 24 users already used for product tuning (eval/everyday*); test/dev = the "
                       "33 untouched users, random seed 0. TEST is never used for training or tuning.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits.json").write_text(json.dumps(splits, indent=1))
    (OUT / "libraries").mkdir(exist_ok=True)
    for u in sorted(set(user_of)):
        rows = np.where(user_of == u)[0]
        sub = store.subset(idx, rows)
        it = sub.items.copy()
        t = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601").dt.tz_localize(None)
        it["t"] = t
        it["ev"] = events(it.taken)
        it["ptail"] = it.place.map(place_tail)
        evs = []
        for e, g in it.groupby("ev"):
            pc = Counter(g.ptail[g.ptail != ""]).most_common(2)
            evs.append(dict(event=int(e), start=str(g.t.min()), end=str(g.t.max()), n=len(g),
                            places=[p for p, _ in pc]))
        evs.sort(key=lambda x: x["start"])
        G = face_groups(sub, top=6) if len(sub.faces) else []
        groups = [dict(n_items=len(g["items"]), n_faces=len(g["faces"]),
                       item_ids=[str(sub.items.item_id.iloc[i]) for i in g["items"]],
                       face_rows=[int(f) for f in g["faces"]], rep=int(g["rep"])) for g in G]   # rows of store.subset(user)
        lib = dict(user=u, split=next(s for s in ("test", "dev", "train") if u in splits[s]), n=len(rows),
                   date_from=str(t.min()), date_to=str(t.max()), n_events=len(evs), events=evs,
                   top_places=Counter(it.ptail[it.ptail != ""]).most_common(15),
                   face_groups=groups, n_faces=int(len(sub.faces)))
        (OUT / "libraries" / f"{u}.json").write_text(json.dumps(lib, indent=1, default=str))
        print(u, lib["split"], len(rows), "events", len(evs), "groups", [g["n_items"] for g in groups], flush=True)


if __name__ == "__main__":
    main()
