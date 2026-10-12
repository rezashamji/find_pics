"""Complex-query benchmark, step 4c: SUPPLEMENTARY blind audit (same protocol, same 160 requests as audit/).

Why: audit/ sampled non-answers only INSIDE each program's time/place scope, so a program that narrows the request
with dates or places the owner never said (v1: "the guy playing the red guitar on stage" -> one day in Barcelona)
could not show those misses. v2 truth (build.py) drops such hidden constraints. This audit samples, per request:
  added   : up to 3 photos that are answers in v2 but were not in v1 (precision of what v2 added)
  out_hard: up to 3 non-answers OUTSIDE the v1 scope among the 30 most text-similar (misses outside the scope)
  out_rand: 1 other non-answer outside the v1 scope
  decoy   : 1 answer in both versions (keeps the labeller blind; also counts toward v1 precision)
Usage (main env): python eval/complex_bench/audit2.py make
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import scope_mask, sheet, tile  # noqa: E402
from build import Lib  # noqa: E402
from common import ROOT, city, load_items  # noqa: E402

A2 = ROOT / "audit2"


def main():
    from findpics import store
    from findpics.models import ImageTextEncoder
    key1 = json.loads((ROOT / "audit" / "key.json").read_text())
    V1, V2 = {}, {}
    for s in ("test", "dev"):
        V1.update({q["qid"]: q for q in json.loads((ROOT / "queries_v1" / f"{s}.json").read_text())["queries"]})
        V2.update({q["qid"]: q for q in json.loads((ROOT / "queries_v2" / f"{s}.json").read_text())["queries"]})
    it = load_items()
    idx = store.load("data/public/index_disbench")
    row_of = {str(i): k for k, i in enumerate(idx.items.item_id.astype(str))}
    enc = ImageTextEncoder(idx.clip_model, device="cpu")
    X = idx.clip.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    unit_item = idx.units.item_row.to_numpy()
    names = sorted(key1)
    TT = enc.texts([" ".join(key1[n]["turns"]) for n in names]).astype(np.float32)
    TT /= np.linalg.norm(TT, axis=1, keepdims=True)
    rng = random.Random(1)
    A2.mkdir(exist_ok=True); (A2 / "sheets").mkdir(exist_ok=True)
    key, libs = {}, {}
    for n, T in zip(names, TT):
        k1 = key1[n]; qid = k1["qid"]; u = k1["user"]
        q1, q2 = V1.get(qid), V2.get(qid)
        if q1 is None or q2 is None:
            print(n, qid, "not in both versions", flush=True); continue
        if u not in libs:
            libs.clear(); libs[u] = Lib(u, it)
        lib = libs[u]
        rows = np.array([row_of[i] for i in lib.ids])
        um = np.isin(unit_item, rows)
        simv = pd.DataFrame({"r": unit_item[um], "s": X[um] @ T}).groupby("r").s.max().reindex(rows).fillna(-1.0).to_numpy()
        a1 = set().union(*map(set, q1["answers"][-1])); a2 = set().union(*map(set, q2["answers"][-1]))
        progs1 = q1["program"][-1] if q1["cat"] == "h" else q1["program"]
        scope1 = np.zeros(len(lib.ids), bool)
        for a in progs1:
            if isinstance(a, dict):
                scope1 |= scope_mask(lib, a, q1["cat"])
        seen = {x["item_id"] for x in k1["items"]}
        items = []
        added = sorted(a2 - a1 - seen)
        for i in rng.sample(added, min(3, len(added))):
            items.append((i, "added"))
        out = np.where(~scope1 & ~np.isin(lib.ids, list(a2 | seen)))[0]
        out = out[np.argsort(-simv[out])]
        for r in rng.sample(list(out[:30]), min(3, len(out[:30]))):
            items.append((lib.ids[r], "out_hard"))
        for r in rng.sample(list(out[30:]), min(1, len(out[30:]))):
            items.append((lib.ids[r], "out_rand"))
        both = sorted((a1 & a2) - seen)
        if both:
            items.append((rng.choice(both), "decoy"))
        rng.shuffle(items)
        tiles = [tile(lib.L.path.iloc[lib.pos[i]], str(j), f"{lib.L.t.iloc[lib.pos[i]]:%Y-%m-%d %a %H:%M} | "
                      f"{city(lib.L.place.iloc[lib.pos[i]])}") for j, (i, _) in enumerate(items, 1)]
        for s in range(0, len(tiles), 6):
            sheet(tiles[s:s + 6]).save(A2 / "sheets" / f"{n}_{s // 6}.jpg", quality=92)
        key[n] = dict(qid=qid, cat=k1["cat"], user=u, turns=k1["turns"], today=k1["today"],
                      items=[dict(n=j, item_id=i, role=r) for j, (i, r) in enumerate(items, 1)],
                      n_added=len(added), n_out=int(len(out)), n_out_hard_pool=int(min(30, len(out))),
                      a1=len(a1), a2=len(a2))
        print(n, qid, len(items), flush=True)
    (A2 / "key.json").write_text(json.dumps(key, indent=1))
    with open(A2 / "requests.txt", "w") as fh:
        for n, k in key.items():
            if k["items"]:
                fh.write(f"{n} [{k['cat']}] (today {k['today']}) " + " || ".join(k["turns"]) + "\n")


if __name__ == "__main__":
    main()
