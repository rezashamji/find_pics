"""Follow-up to eval/rendition480_parity.py (480 px JPEG rendition -> mean cosine ~0.955 vs the server's full-photo
vector). Which step does it: the downscale filter, the JPEG, or the size? And does it lose QUALITY or only differ?
(a) variants on 200 Open Images + 200 Pexels frames (cosine vs the server vector of the full photo);
(b) quality proxy: Open Images verified image-level labels (Confidence 1; noisy, but the same labels for both paths)
    on the 2000-photo pool: for every class with >= 15 positives, query "a photo of a <class>", average precision of
    the server path vs the 480 path. Labels are not truth (CLAUDE.md); this is a relative comparison only.
Usage (GPU job): python eval/rendition480_decompose.py -> eval/rendition480_decompose.json"""
import csv
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.path.insert(0, "eval")
from coreml_preproc_parity import dev, enc, load_image, model, pre, tok  # noqa: E402


def small(im, side=480, f=Image.LANCZOS, q=80, blur=0.0):
    im = im.copy(); im.thumbnail((side, side), f)
    if q:
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); im = Image.open(b).convert("RGB")
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    return im


def jpeg_only(im, q=80):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


V = {"phone sim: Lanczos 480 + JPEG80": lambda im: small(im),
     "Lanczos 480, no JPEG": lambda im: small(im, q=0),
     "PIL bilinear(AA) 480, no JPEG": lambda im: small(im, f=Image.BILINEAR, q=0),
     "PIL box 480, no JPEG": lambda im: small(im, f=Image.BOX, q=0),
     "PIL bicubic 480, no JPEG": lambda im: small(im, f=Image.BICUBIC, q=0),
     "JPEG80 at full size only": jpeg_only,
     "Lanczos 480 + JPEG95": lambda im: small(im, q=95),
     "Lanczos 448 + JPEG80": lambda im: small(im, side=448),
     "Lanczos 960 + JPEG80": lambda im: small(im, side=960),
     "Lanczos 480 + JPEG80 + blur 0.5": lambda im: small(im, blur=0.5),
     "Lanczos 480 + JPEG80 + blur 0.75": lambda im: small(im, blur=0.75),
     "Lanczos 480 + JPEG80 + blur 1.0": lambda im: small(im, blur=1.0)}

OI = sorted(Path("data/public/raw/openimages/images").glob("*.jpg"))
PX = sorted(Path("data/public/pexels_frames").glob("*.jpg"))
out = {"variants": {}, "label_ap": {}}
sets = {n: [load_image(str(p)) for p in random.Random(1).sample(ps, 2000)[:200]] for n, ps in [("openimages", OI), ("pexels", PX)]}
base = {n: enc(model, [pre(im) for im in ims]) for n, ims in sets.items()}
for vn, fn in V.items():
    row = {}
    for n, ims in sets.items():
        c = (enc(model, [pre(fn(im)) for im in ims], half=True) * base[n]).sum(1).numpy()
        row[n] = dict(mean=round(float(c.mean()), 4), p5=round(float(np.percentile(c, 5)), 4), min=round(float(c.min()), 4))
    out["variants"][vn] = row; print(vn, row, flush=True)
json.dump(out, open("eval/rendition480_decompose.json", "w"), indent=1)

pool = random.Random(1).sample(OI, 2000)
ids = [p.stem for p in pool]; pos = {}
names = {r["LabelName"]: r["DisplayName"] for r in csv.DictReader(open("data/public/raw/openimages/classes.csv"))}
idset = set(ids)
for r in csv.DictReader(open("data/public/raw/openimages/val_labels.csv")):
    if r["Confidence"] == "1" and r["ImageID"] in idset:
        pos.setdefault(r["LabelName"], set()).add(r["ImageID"])
classes = sorted(k for k, v in pos.items() if len(v) >= 15 and k in names)
ims = [load_image(str(p)) for p in pool]
S = enc(model, [pre(im) for im in ims]); P = enc(model, [pre(small(im)) for im in ims], half=True)
with torch.no_grad():
    T = F.normalize(model.encode_text(tok([f"a photo of a {names[c]}" for c in classes]).to(dev)).float(), dim=-1).cpu()


def ap(scores, y):
    o = np.argsort(-scores); y = y[o]; hits = np.cumsum(y)
    return float((hits / np.arange(1, len(y) + 1))[y == 1].mean())


rows = []
for i, c in enumerate(classes):
    y = np.array([1 if x in pos[c] else 0 for x in ids])
    a, b = ap((S @ T[i]).numpy(), y), ap((P @ T[i]).numpy(), y)
    rows.append((names[c], int(y.sum()), round(a, 3), round(b, 3)))
d = np.array([r[3] - r[2] for r in rows])
out["label_ap"] = dict(n_classes=len(rows), mean_ap_server=round(float(np.mean([r[2] for r in rows])), 4),
                       mean_ap_480=round(float(np.mean([r[3] for r in rows])), 4),
                       classes_480_better=int((d > 0.005).sum()), classes_480_worse=int((d < -0.005).sum()),
                       per_class=rows)
print({k: v for k, v in out["label_ap"].items() if k != "per_class"}, flush=True)
json.dump(out, open("eval/rendition480_decompose.json", "w"), indent=1)
print("R480D_OK", flush=True)
