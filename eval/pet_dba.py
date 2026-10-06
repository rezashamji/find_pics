"""Neighbour smoothing (DBA k=2, converse._dba) for "find my dog" from 3 reference photos (RESULTS 17 protocol: 40 dogs
with >= 6 photos, seed 0, refs = 3 random photos, targets = the dog's other photos). Ranking only (no judge):
top-3 / top-5 precision and R-precision, plain mean-of-refs (product before 10-06) vs DBA.
Libraries: (a) all DogFaceNet photos (worst case: every photo a dog); (b) the 19,218 everyday test-library photos PLUS
all DogFaceNet photos (harder than RESULTS 17's mixed library, which added only the query dog's own photos).
Usage (GPU for the image encoder): python eval/pet_dba.py"""
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")
from findpics.converse import _dba


def metrics(L, V, lab, dogs, refs_of, dba):
    X = L
    if dba:
        X, _ = _dba(L, L[:1], k=2)
    res = {"top3": [0, 0], "top5": [0, 0], "rprec": []}
    for d in dogs:
        refs, tg = refs_of[d]
        Q = V[refs]
        if dba:
            _, Q = _dba(L, Q, k=2)
        s = (X @ Q.T).mean(1); s[refs] = -np.inf          # dog rows are the first len(V) rows of L
        o = np.argsort(-s)
        for k in (3, 5):
            res[f"top{k}"][0] += int(np.isin(o[:k], tg).sum()); res[f"top{k}"][1] += k
        res["rprec"].append(float(np.isin(o[:len(tg)], tg).mean()))
    return {"top3": f"{res['top3'][0]}/{res['top3'][1]}", "top5": f"{res['top5'][0]}/{res['top5'][1]}",
            "rprec": round(float(np.mean(res["rprec"])), 3)}


def main():
    from findpics import store
    from findpics.models import ImageTextEncoder
    fs = sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))
    df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
    col = [c for c in df.columns if c != "label"][0]
    raw = lambda v: v["bytes"] if isinstance(v, dict) else v
    df["h"] = [hashlib.md5(raw(v)).hexdigest() for v in df[col]]
    multi = df.groupby("h").label.nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h")
    ims, keep = [], []
    for v in df[col]:
        try:
            ims.append(Image.open(io.BytesIO(raw(v))).convert("RGB")); keep.append(True)
        except Exception:
            keep.append(False)
    df = df[keep].reset_index(drop=True); lab = df.label.to_numpy()
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336")
    V = np.concatenate([enc.images(ims[i:i + 128]) for i in range(0, len(ims), 128)]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    tl = store.load("data/public/index_testlib")
    first = tl.units.reset_index().groupby("item_row")["index"].first().to_numpy()
    BG = tl.clip[first].astype(np.float32); BG /= np.linalg.norm(BG, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    dogs = [d for d in pd.unique(lab) if (lab == d).sum() >= 6]
    dogs = list(rng.choice(dogs, min(40, len(dogs)), replace=False))
    refs_of = {}
    for d in dogs:
        mine = np.where(lab == d)[0]; refs = rng.choice(mine, 3, replace=False)
        refs_of[d] = (refs, np.setdiff1d(mine, refs))
    out = {}
    for name, L in (("all dogs", V), ("everyday + all dogs", np.concatenate([V, BG]))):
        out[name] = {"library": len(L), "plain": metrics(L, V, lab, dogs, refs_of, False),
                     "DBA k=2": metrics(L, V, lab, dogs, refs_of, True)}
        print(name, out[name], flush=True)
    json.dump(out, open("eval/pet_dba.json", "w"), indent=1)


if __name__ == "__main__":
    main()
