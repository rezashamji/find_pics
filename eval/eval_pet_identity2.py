"""Individual-dog identity, v2: (a) crop the dog with an open-vocabulary detector so vectors describe the animal, not
the room (raw look of v1: matches followed setting + coat color); (b) add a dedicated animal re-ID model;
(c) remove exact-duplicate photos filed under different dog IDs (v1 raw look: 'other dog' at similarity 1.00).
Models:
  PE-Core-L14-336        text-image (current index vector)
  DINOv2-base            self-supervised
  MegaDescriptor-B-224   animal re-identification (BVRA). Leakage check: last modified 2024-01-05, BEFORE DogFaceNet was
                         added to the WildlifeDatasets toolkit (09/05/2024), so its weights cannot contain DogFaceNet.
Detector: OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0), query "a photo of a dog", best box + 10% margin.
Protocol as v1: dogs with >= 6 photos; 3 refs; targets = rest; distractors = all other dogs. R-precision, recall@top2T.
Usage (main env, GPU): python eval/eval_pet_identity2.py
"""
import hashlib
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
col = [c for c in df.columns if c != "label"][0]
raw = [v["bytes"] if isinstance(v, dict) else v for v in df[col]]
df["h"] = [hashlib.md5(b).hexdigest() for b in raw]
multi = df.groupby("h").label.nunique(); bad_h = set(multi[multi > 1].index)
print(f"photos {len(df)}; exact duplicates filed under >1 dog: {int(df.h.isin(bad_h).sum())} photos removed", flush=True)
keep_rows = ~df.h.isin(bad_h) & ~df.duplicated("h")


def load(b):
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None


ims = [load(b) if k else None for b, k in zip(raw, keep_rows)]
ok = np.array([im is not None for im in ims])
df = df[ok].reset_index(drop=True); ims = [im for im in ims if im is not None]
cnt = df.label.value_counts(); keep = cnt[cnt >= 6].index
sel = df.label.isin(keep).to_numpy(); df = df[sel].reset_index(drop=True); ims = [im for im, s in zip(ims, sel) if s]
print(f"after cleaning: {len(keep)} dogs, {len(df)} photos", flush=True)

# --- crop dogs with OWLv2
from transformers import Owlv2ForObjectDetection, Owlv2Processor
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
det = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval()
crops, found = [], 0
with torch.inference_mode():
    for i in range(0, len(ims), 32):
        batch = ims[i:i + 32]
        inp = proc(text=[["a photo of a dog"]] * len(batch), images=batch, return_tensors="pt").to("cuda")
        out = det(**inp)
        sizes = torch.tensor([[max(im.size)] * 2 for im in batch], device="cuda")  # OWLv2 pads to a square
        res = proc.post_process_grounded_object_detection(outputs=out, threshold=0.1, target_sizes=sizes)
        for im, r in zip(batch, res):
            if len(r["scores"]):
                x1, y1, x2, y2 = r["boxes"][int(r["scores"].argmax())].tolist(); w, h = x2 - x1, y2 - y1
                box = (max(0, x1 - .1 * w), max(0, y1 - .1 * h), min(im.width, x2 + .1 * w), min(im.height, y2 + .1 * h))
                if box[2] - box[0] > 16 and box[3] - box[1] > 16:
                    crops.append(im.crop(box)); found += 1; continue
            crops.append(im)
print(f"dog detected and cropped in {found}/{len(ims)} photos", flush=True)
del det; torch.cuda.empty_cache()
Path("eval/audits").mkdir(parents=True, exist_ok=True)


def pe(imgs):
    from findpics.models import ImageTextEncoder
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336")
    V = np.concatenate([enc.images(imgs[i:i + 128]) for i in range(0, len(imgs), 128)]).astype(np.float32)
    del enc; torch.cuda.empty_cache(); return V


def dino(imgs):
    from transformers import AutoImageProcessor, AutoModel
    p = AutoImageProcessor.from_pretrained("facebook/dinov2-base"); m = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    out = []
    with torch.inference_mode():
        for i in range(0, len(imgs), 128):
            h = m(pixel_values=p(images=imgs[i:i + 128], return_tensors="pt")["pixel_values"].cuda().half()).last_hidden_state
            out.append(torch.nn.functional.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float(), dim=-1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(out)


def mega(imgs):
    import timm
    import torchvision.transforms as T
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-B-224", pretrained=True).cuda().eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    out = []
    with torch.inference_mode():
        for i in range(0, len(imgs), 128):
            x = torch.stack([tf(im) for im in imgs[i:i + 128]]).cuda()
            out.append(torch.nn.functional.normalize(m(x).float(), dim=-1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(out)


labels = df.label.to_numpy()
res = {}
for vname, fn in [("PE-Core", pe), ("DINOv2", dino), ("MegaDescriptor-B-224", mega)]:
    for iname, imgs in [("full", ims), ("crop", crops)]:
        V = fn(imgs); V /= np.linalg.norm(V, axis=1, keepdims=True)
        rng = np.random.default_rng(0); rp, r2 = [], []
        for dog in keep:
            mine = np.where(labels == dog)[0]; ref = rng.choice(mine, 3, replace=False); tgt = np.setdiff1d(mine, ref)
            s = (V @ V[ref].T).max(1); s[ref] = -np.inf; order = np.argsort(-s); T_ = len(tgt)
            rp.append(np.isin(order[:T_], tgt).mean()); r2.append(np.isin(tgt, order[:2 * T_]).mean())
        k = f"{vname}/{iname}"
        res[k] = dict(r_precision=round(float(np.mean(rp)), 3), recall_top2T=round(float(np.mean(r2)), 3),
                      dogs_rprec_ge_0_5=int((np.array(rp) >= .5).sum()), dogs=int(len(keep)))
        print(k, res[k], flush=True)
        np.save(f"eval/oracle/_dog_{vname}_{iname}.npy", V) if Path("eval/oracle").exists() else None
json.dump(res, open("eval/results_pet_identity2.json", "w"), indent=1)
# save crops sample for raw look
for i in range(0, min(len(crops), 40)):
    crops[i].save(f"eval/audits/_dogcrop_{i}.jpg")
