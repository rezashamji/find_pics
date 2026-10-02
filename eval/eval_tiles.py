"""Do tile vectors fix the cheap stage's blind spot for small background objects?
Evidence (journal 10-02 ~14:4x): oracle-confirmed bicycles/guitars missed by fast mode were small background objects
ranked 1,200-5,000 by the single whole-photo vector.
  embed  (main env, GPU): PE-Core vectors for every testlib photo: whole image + 2x2 grid + 3x3 grid (14 per photo)
  analyze (CPU): for each oracle query (eval/oracle/*.parquet), truth = judge yes@0.7 on every item; compare recall of
  that truth within the top-K items when ranking by (a) whole-image vector only, (b) max over whole + tiles,
  (c) max over whole + 2x2 only. K = 1,000 / 2,000 / 4,000 (fast mode's judged head is ~1,600-4,200).
Usage: python eval/eval_tiles.py embed | python eval/eval_tiles.py analyze
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("eval/tiles"); OUT.mkdir(parents=True, exist_ok=True)


def tiles(im):
    w, h = im.size; out = [im]
    for n in (2, 3):
        for i in range(n):
            for j in range(n):
                out.append(im.crop((int(j * w / n), int(i * h / n), int((j + 1) * w / n), int((i + 1) * h / n))))
    return out  # 1 + 4 + 9 = 14


def embed():
    import torch
    from findpics import store
    from findpics.media import load_image
    from findpics.models import ImageTextEncoder
    idx = store.load("data/public/index_testlib")
    enc = ImageTextEncoder(idx.clip_model)
    photos = np.where(idx.items.media.to_numpy() == "photo")[0]
    V = np.zeros((idx.n_items, 14, 1024), np.float16)
    buf, rows = [], []
    for k, r in enumerate(photos):
        try:
            buf += tiles(load_image(idx.items.path.iloc[r])); rows.append(r)
        except Exception:
            pass
        if len(rows) == 16 or k == len(photos) - 1:
            if rows:
                v = enc.images(buf).reshape(len(rows), 14, -1); V[rows] = v
            buf, rows = [], []
            if k % 1600 == 0:
                print(k, flush=True)
    np.save(OUT / "tile_vectors.npy", V)
    print("EMBED_DONE", V.shape)


def analyze():
    from findpics import store
    from findpics.models import ImageTextEncoder
    idx = store.load("data/public/index_testlib")
    V = np.load(OUT / "tile_vectors.npy", mmap_mode="r")
    enc = ImageTextEncoder(idx.clip_model, device="cpu")
    rows = []
    for f in sorted(Path("eval/oracle").glob("*.parquet")):
        qid = f.stem
        d = pd.read_parquet(f); truth = np.where(d.p.to_numpy() >= 0.7)[0]
        t = enc.texts([f"a photo of {qid.replace('_', ' ')}"])[0].astype(np.float32)
        S = np.asarray(V, dtype=np.float32) @ t          # [N, 14]
        variants = {"whole": S[:, 0], "whole+2x2": S[:, :5].max(1), "whole+2x2+3x3": S.max(1)}
        for name, s in variants.items():
            order = np.argsort(-s)
            for K in (1000, 2000, 4000):
                rows.append(dict(query=qid, variant=name, K=K, truth=len(truth),
                                 recall=float(np.isin(truth, order[:K]).mean())))
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["query", "truth"], columns=["K", "variant"], values="recall").round(3).to_string())
    print(df.groupby(["K", "variant"]).recall.mean().round(3).to_string())
    df.to_json("eval/results_tiles.json", orient="records", indent=1)


if __name__ == "__main__":
    {"embed": embed, "analyze": analyze}[sys.argv[1]]()
