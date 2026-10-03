"""How much does the image-text encoder's SIZE matter? (headroom with bigG; cost of phone-sized T/S/B)
For one PE-Core model per job:
  A. instance search ("this specific thing/place"): Stanford Online Products (2,000 ids max) + Google Landmarks v2 mini
     (1,500 ids), 3 refs, MEAN similarity over refs, R-precision (same protocol/seed as eval_instance.py).
  B. concept search (the cheap first stage of every query): all photos of the 19,218-item test library; for each of the
     20 oracle concepts (eval/oracle/<qid>.parquet: the 9B judge on every item), rank by the concept's text vector and
     measure recall of the judge-yes photos inside the top 2,000 (what streaming round 1 judges).
Output: eval/encoder_sizes/<model>.json.   Usage (main env, GPU): python eval/eval_encoder_sizes.py <PE-Core name>
Models: PE-Core-T-16-384, PE-Core-S-16-384, PE-Core-B-16, PE-Core-L-14-336 (current), PE-Core-bigG-14-448.
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def load_parquets(files, label_col, max_ids=None, min_n=4):
    rng = np.random.default_rng(0)
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    raw = lambda b: b["bytes"] if isinstance(b, dict) else b
    df["h"] = [hashlib.md5(raw(b)).hexdigest() for b in df["image"]]
    multi = df.groupby("h")[label_col].nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h")
    cnt = df[label_col].value_counts(); ids = cnt[cnt >= min_n].index
    if max_ids and len(ids) > max_ids:
        ids = rng.choice(ids, max_ids, replace=False)
    df = df[df[label_col].isin(set(ids))].reset_index(drop=True)
    ims, lab = [], []
    for b, l in zip(df["image"], df[label_col]):
        try:
            ims.append(Image.open(io.BytesIO(raw(b))).convert("RGB")); lab.append(l)
        except Exception:
            pass
    return ims, np.array(lab)


def embed(enc, ims, bs=64):
    V = np.concatenate([enc.images(ims[i:i + bs]) for i in range(0, len(ims), bs)]).astype(np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def rprec(V, labels):
    r = np.random.default_rng(0); out = []
    for l in [l for l in pd.unique(labels) if (labels == l).sum() >= 2]:
        mine = np.where(labels == l)[0]; ref = r.choice(mine, min(3, len(mine) - 1), replace=False)
        tgt = np.setdiff1d(mine, ref); s = (V @ V[ref].T).mean(1); s[ref] = -np.inf
        out.append(np.isin(np.argsort(-s)[:len(tgt)], tgt).mean())
    return round(float(np.mean(out)), 3)


def main():
    name = sys.argv[1]
    sys.path.insert(0, "eval")
    from findpics import store
    from findpics.media import load_image
    from findpics.models import ImageTextEncoder
    from eval_oracle import QUERIES
    enc = ImageTextEncoder(f"hf-hub:timm/{name}")
    res = dict(model=name)
    for kind, files, col, mx in (("things", sorted(Path("data/public/raw/sop/data").glob("*.parquet")), "item_id", 2000),
                                 ("places", sorted(Path("data/public/raw/gldv2/data").glob("train-*.parquet")), "label", 1500)):
        ims, lab = load_parquets(files, col, max_ids=mx)
        res[f"instance_{kind}"] = rprec(embed(enc, ims), lab); print(name, kind, res[f"instance_{kind}"], flush=True)
        del ims
    idx = store.load("data/public/index_testlib")
    photos = np.where(idx.items.media.to_numpy() == "photo")[0]
    V = []
    for i in range(0, len(photos), 64):
        V.append(embed(enc, [load_image(idx.items.path.iloc[int(r)]) for r in photos[i:i + 64]]))
    V = np.concatenate(V)
    rec = {}
    for Q in QUERIES:
        f = Path(f"eval/oracle/{Q['qid']}.parquet")
        if not f.exists():
            continue
        P = pd.read_parquet(f).set_index("item_row").p.reindex(photos).to_numpy()
        yes = set(np.where(P >= 0.7)[0])
        t = enc.texts(Q["looks"]).astype(np.float32).mean(0); t /= np.linalg.norm(t)
        top = set(np.argsort(-(V @ t))[:2000])
        rec[Q["qid"]] = round(len(yes & top) / max(len(yes), 1), 3)
    res["concept_recall_top2000"] = rec; res["concept_recall_median"] = float(np.median(list(rec.values())))
    print(json.dumps(res), flush=True)
    Path("eval/encoder_sizes").mkdir(exist_ok=True)
    json.dump(res, open(f"eval/encoder_sizes/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
