"""'Who' for non-humans: can reference photos of ONE specific dog find that dog's other photos among photos of OTHER
dogs (the hard case: every distractor is also a dog)? DogFaceNet_large: individual dogs with several photos each.
Compares general-purpose image vectors (no face model exists for dogs here):
  - PE-Core-L14-336 (text-image model; the product's current index vector)
  - DINOv2-base (self-supervised; known for fine visual detail)
Protocol per dog with >= 6 photos: 3 random photos = references; the rest = targets; all other dogs' photos = distractors.
Metric: R-precision (of the top-T ranked items, T = #targets, fraction that are this dog) and recall within top 2T.
Usage (main env, GPU): python eval/eval_pet_identity.py
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

fs = sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))
df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
img_col = [c for c in df.columns if c != "label"][0]
counts = df.label.value_counts()
keep = counts[counts >= 6].index
df = df[df.label.isin(keep)].reset_index(drop=True)
print(f"dogs with >=6 photos: {len(keep)}, photos: {len(df)}", flush=True)
def _load(v):
    try:
        return Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v)).convert("RGB")
    except Exception:
        return None
ims = [_load(v) for v in df[img_col]]
ok = [i for i, im in enumerate(ims) if im is not None]
print(f"unreadable images skipped: {len(ims) - len(ok)}", flush=True)
df = df.iloc[ok].reset_index(drop=True); ims = [ims[i] for i in ok]
counts = df.label.value_counts(); keep = counts[counts >= 6].index
sel = df.label.isin(keep).to_numpy(); df = df[sel].reset_index(drop=True); ims = [im for im, k in zip(ims, sel) if k]
print("image sizes (first 5):", [im.size for im in ims[:5]], flush=True)


def pe_vectors():
    from findpics.models import ImageTextEncoder
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336")
    V = np.concatenate([enc.images(ims[i:i + 128]) for i in range(0, len(ims), 128)]).astype(np.float32)
    del enc; torch.cuda.empty_cache(); return V


def dino_vectors():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    m = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    out = []
    with torch.inference_mode():
        for i in range(0, len(ims), 128):
            x = proc(images=ims[i:i + 128], return_tensors="pt")["pixel_values"].cuda().half()
            h = m(pixel_values=x).last_hidden_state
            v = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)          # CLS + mean patch token (common retrieval recipe)
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


labels = df.label.to_numpy()
rng = np.random.default_rng(0)
res = {}
for name, fn in [("PE-Core-L14-336", pe_vectors), ("DINOv2-base", dino_vectors)]:
    V = fn(); V /= np.linalg.norm(V, axis=1, keepdims=True)
    rp, r2 = [], []
    for dog in keep:
        mine = np.where(labels == dog)[0]
        ref = rng.choice(mine, 3, replace=False); tgt = np.setdiff1d(mine, ref)
        s = (V @ V[ref].T).max(1); s[ref] = -np.inf
        order = np.argsort(-s); T = len(tgt)
        rp.append(np.isin(order[:T], tgt).mean()); r2.append(np.isin(tgt, order[:2 * T]).mean())
    res[name] = dict(dogs=int(len(keep)), photos=int(len(df)), r_precision=float(np.mean(rp)), recall_top2T=float(np.mean(r2)),
                     dogs_rprec_ge_0_5=int((np.array(rp) >= 0.5).sum()))
    print(name, res[name], flush=True)
json.dump(res, open("eval/results_pet_identity.json", "w"), indent=1)


