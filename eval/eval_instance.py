"""'Find this specific X' for ANY kind of X, measured the same way. One reference set per identity -> rank everything
-> how many of that identity's other photos come back, against distractors of the SAME kind (the hard case).
Kinds and public sets:
  dogs     DogFaceNet_large (individual dogs; exact cross-ID duplicates removed)        crop: OWLv2 "a photo of a dog"
  things   Stanford Online Products, test shard 0 (same product, different photos)       no crop (product-centred)
  places   Google Landmarks v2 mini, test (same landmark, different visitors' photos)    no crop (the place is the scene)
  copies   self-generated "picture of a picture": 300 Open Images photos x 5 copies      no crop
           (photo of a print with perspective, framed on a wall, phone screenshot, crop + heavy JPEG, photo of a screen)
Models: PE-Core-L14-336 (index vector), DINOv2-base; for dogs also MegaDescriptor-B-224 (clean vs DogFaceNet: last
modified 2024-01-05, before DogFaceNet joined its toolkit 2024-09-05).
Metric per identity: refs = min(3, n-1) photos (copies: the original only); targets = the rest; R-precision and recall
within the top 2x. Usage (main env, GPU): python eval/eval_instance.py <kind>
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFilter

KIND = sys.argv[1]
rng = np.random.default_rng(0)


def _img(b):
    try:
        return Image.open(io.BytesIO(b["bytes"] if isinstance(b, dict) else b)).convert("RGB")
    except Exception:
        return None


def load_parquets(files, label_col, max_ids=None, min_n=4):
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    imcol = "image"
    df["h"] = [hashlib.md5((b["bytes"] if isinstance(b, dict) else b)).hexdigest() for b in df[imcol]]
    multi = df.groupby("h")[label_col].nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h")
    cnt = df[label_col].value_counts(); ids = cnt[cnt >= min_n].index
    if max_ids and len(ids) > max_ids:
        ids = rng.choice(ids, max_ids, replace=False)
    df = df[df[label_col].isin(set(ids))].reset_index(drop=True)
    ims = [_img(b) for b in df[imcol]]; ok = [i for i, im in enumerate(ims) if im is not None]
    return [ims[i] for i in ok], df[label_col].to_numpy()[ok]


def make_copies(n=300):
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    paths = sorted(Path("data/public/testlib/library/openimages").glob("*.jpg"))
    pick = rng.choice(len(paths), n + 3000, replace=False)
    origs = [Image.open(paths[i]).convert("RGB") for i in pick[:n]]
    distract = [Image.open(paths[i]).convert("RGB") for i in pick[n:]]
    for im in origs + distract:
        im.thumbnail((800, 800))
    walls = distract[:50]

    def print_photo(im):   # photo of a print: perspective + blur + warm cast
        w, h = im.size; d = 0.08
        q = [(rng.uniform(0, d) * w, rng.uniform(0, d) * h), (w - rng.uniform(0, d) * w, rng.uniform(0, d) * h),
             (w - rng.uniform(0, d) * w, h - rng.uniform(0, d) * h), (rng.uniform(0, d) * w, h - rng.uniform(0, d) * h)]
        out = im.transform((w, h), Image.QUAD, sum(q, ()), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.2))
        a = np.asarray(out).astype(np.float32); a[..., 0] *= 1.08; a[..., 2] *= 0.9
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    def framed(im):        # framed photo hanging on a wall, smaller in the scene
        wall = walls[rng.integers(len(walls))].copy().resize((1000, 750))
        f = im.copy(); f.thumbnail((420, 420)); fr = Image.new("RGB", (f.width + 40, f.height + 40), (40, 30, 20)); fr.paste(f, (20, 20))
        wall.paste(fr, (int(rng.uniform(50, 1000 - fr.width - 50)), int(rng.uniform(50, 750 - fr.height - 50)))); return wall

    def screenshot(im):    # phone screenshot: UI bars, letterbox
        s = im.copy(); s.thumbnail((390, 600)); c = Image.new("RGB", (390, 844), (250, 250, 250))
        c.paste(s, (0, (844 - s.height) // 2)); dr = ImageDraw.Draw(c); dr.rectangle([0, 0, 390, 90], fill=(240, 240, 240))
        dr.rectangle([0, 760, 390, 844], fill=(240, 240, 240)); return c

    def crop_jpeg(im):     # 60% crop + heavy compression
        w, h = im.size; x, y = rng.uniform(0, .4 * w), rng.uniform(0, .4 * h)
        b = io.BytesIO(); im.crop((x, y, x + .6 * w, y + .6 * h)).save(b, "JPEG", quality=25); return Image.open(b).convert("RGB")

    def screen_photo(im):  # photo of a screen: moire + gamma + slight rotation
        a = np.asarray(im).astype(np.float32); yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
        a *= (0.88 + 0.12 * np.sin(xx * 0.9 + yy * 0.35))[..., None]; a = 255 * (a / 255) ** 0.85
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).rotate(rng.uniform(-6, 6), expand=True, fillcolor=(10, 10, 10))

    ims, labels = [], []
    for k, o in enumerate(origs):
        for f in (lambda x: x, print_photo, framed, screenshot, crop_jpeg, screen_photo):
            ims.append(f(o)); labels.append(k)
    ims += distract; labels += list(range(n, n + len(distract)))     # distractors: unrelated photos, 1 each
    return ims, np.array(labels)


if KIND == "dogs":
    ims, labels = load_parquets(sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet")), "label", min_n=6)
elif KIND == "things":
    ims, labels = load_parquets(sorted(Path("data/public/raw/sop/data").glob("*.parquet")), "item_id", max_ids=2000)
elif KIND == "places":
    ims, labels = load_parquets(sorted(Path("data/public/raw/gldv2/data").glob("*.parquet")), "label", max_ids=1500)
elif KIND == "copies":
    ims, labels = make_copies()
print(f"{KIND}: {len(ims)} photos, {len(set(labels))} identities", flush=True)

crops = ims
if KIND == "dogs":
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    det = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval()
    crops = []
    with torch.inference_mode():
        for i in range(0, len(ims), 32):
            b = ims[i:i + 32]
            out = det(**proc(text=[["a photo of a dog"]] * len(b), images=b, return_tensors="pt").to("cuda"))
            res = proc.post_process_object_detection(out, threshold=0.1, target_sizes=torch.tensor([[max(im.size)] * 2 for im in b], device="cuda"))
            for im, r in zip(b, res):
                if len(r["scores"]):
                    x1, y1, x2, y2 = r["boxes"][int(r["scores"].argmax())].tolist(); w, h = x2 - x1, y2 - y1
                    bx = (max(0, x1 - .1 * w), max(0, y1 - .1 * h), min(im.width, x2 + .1 * w), min(im.height, y2 + .1 * h))
                    if bx[2] - bx[0] > 16 and bx[3] - bx[1] > 16:
                        crops.append(im.crop(bx)); continue
                crops.append(im)
    del det; torch.cuda.empty_cache()


def pe(x):
    from findpics.models import ImageTextEncoder
    e = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336"); V = np.concatenate([e.images(x[i:i + 128]) for i in range(0, len(x), 128)])
    del e; torch.cuda.empty_cache(); return V.astype(np.float32)


def dino(x):
    from transformers import AutoImageProcessor, AutoModel
    p = AutoImageProcessor.from_pretrained("facebook/dinov2-base"); m = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half(); o = []
    with torch.inference_mode():
        for i in range(0, len(x), 128):
            h = m(pixel_values=p(images=x[i:i + 128], return_tensors="pt")["pixel_values"].cuda().half()).last_hidden_state
            o.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(o)


def mega(x):
    import timm, torchvision.transforms as T
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-B-224", pretrained=True).cuda().eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)]); o = []
    with torch.inference_mode():
        for i in range(0, len(x), 128):
            o.append(m(torch.stack([tf(im) for im in x[i:i + 128]]).cuda()).float().cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(o)


models = [("PE-Core", pe), ("DINOv2", dino)] + ([("MegaDescriptor-B-224", mega)] if KIND == "dogs" else [])
variants = [("full", ims)] + ([("crop", crops)] if KIND == "dogs" else [])
res = {}
ids = [l for l in pd.unique(labels) if (labels == l).sum() >= 2]
for mn, fn in models:
    for vn, x in variants:
        V = fn(x); V /= np.linalg.norm(V, axis=1, keepdims=True); r = np.random.default_rng(0); rp, r2 = [], []
        for l in ids:
            mine = np.where(labels == l)[0]
            ref = mine[:1] if KIND == "copies" else r.choice(mine, min(3, len(mine) - 1), replace=False)
            tgt = np.setdiff1d(mine, ref); s = (V @ V[ref].T).max(1); s[ref] = -np.inf; o = np.argsort(-s); T_ = len(tgt)
            rp.append(np.isin(o[:T_], tgt).mean()); r2.append(np.isin(tgt, o[:2 * T_]).mean())
        k = f"{mn}/{vn}"; res[k] = dict(identities=len(ids), photos=len(x), r_precision=round(float(np.mean(rp)), 3),
                                        recall_top2T=round(float(np.mean(r2)), 3))
        print(KIND, k, res[k], flush=True)
json.dump(res, open(f"eval/results_instance_{KIND}.json", "w"), indent=1)
