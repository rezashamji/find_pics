"""Can the VLM judge verify 'same individual dog' side by side? (It could NOT for people: 4/191 correct yeses.)
DogFaceNet (cross-ID duplicates removed). 300 same-dog pairs + 300 HARD different-dog pairs (the other dog whose photo
is most similar by PE-Core vector). Judge sees [photo A | photo B] and answers. Reports AUC and yes-rates.
Usage (vLLM env, GPU): python eval/eval_pet_judge.py
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def main():
    from findpics.engine import side_by_side
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    fs = sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))
    df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
    col = [c for c in df.columns if c != "label"][0]
    b = [v["bytes"] if isinstance(v, dict) else v for v in df[col]]
    df["h"] = [hashlib.md5(x).hexdigest() for x in b]
    multi = df.groupby("h").label.nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h").reset_index(drop=True)
    ims = []
    for v in df[col]:
        try:
            ims.append(Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v)).convert("RGB"))
        except Exception:
            ims.append(None)
    ok = np.array([im is not None for im in ims]); df = df[ok].reset_index(drop=True); ims = [im for im in ims if im is not None]
    lab = df.label.to_numpy()
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336")
    V = np.concatenate([enc.images(ims[i:i + 128]) for i in range(0, len(ims), 128)]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    del enc
    import torch; torch.cuda.empty_cache()
    rng = np.random.default_rng(0); pairs = []
    dogs = [d for d in pd.unique(lab) if (lab == d).sum() >= 2]
    for d in rng.choice(dogs, 300, replace=False):
        mine = np.where(lab == d)[0]; a, bb = rng.choice(mine, 2, replace=False); pairs.append((a, bb, 1))
        s = V @ V[a]; s[lab == d] = -9; pairs.append((a, int(np.argmax(s)), 0))
    J = VLLMJudge(gpu_mem=0.7)
    q = "The two panels show dogs. Is it the same individual dog in both panels (not just the same breed)?"
    p = np.array(J.p_yes([side_by_side(ims[a], ims[c]) for a, c, _ in pairs], q))
    y = np.array([t for *_, t in pairs]).astype(bool)
    from scipy.stats import rankdata
    r = rankdata(p); auc = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    res = dict(pairs=len(pairs), auc=float(auc), yes_rate_same=float((p[y] >= .5).mean()), yes_rate_diff=float((p[~y] >= .5).mean()),
               vector_auc=float((lambda s: (rankdata(s)[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))(
                   np.array([V[a] @ V[c] for a, c, _ in pairs]))))
    print(json.dumps(res, indent=1)); json.dump(res, open("eval/results_pet_judge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
