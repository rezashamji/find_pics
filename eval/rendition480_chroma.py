"""Follow-up to eval/rendition480_decompose.py: is the JPEG damage at 480 px the 4:2:0 chroma subsampling (Pillow's
default; chroma then has 240 px, about the model's 224) or the quantization? Same 200 + 200 photos, cosine vs server.
Usage (GPU job): python eval/rendition480_chroma.py -> eval/rendition480_chroma.json"""
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "eval")
from coreml_preproc_parity import enc, load_image, model, pre  # noqa: E402


def small(im, q, sub, side=480):
    im = im.copy(); im.thumbnail((side, side), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, subsampling=sub); b.seek(0); return Image.open(b).convert("RGB")


OI = sorted(Path("data/public/raw/openimages/images").glob("*.jpg")); PX = sorted(Path("data/public/pexels_frames").glob("*.jpg"))
sets = {n: [load_image(str(p)) for p in random.Random(1).sample(ps, 2000)[:200]] for n, ps in [("openimages", OI), ("pexels", PX)]}
base = {n: enc(model, [pre(im) for im in ims]) for n, ims in sets.items()}
out = {}
for q, sub in [(80, 2), (80, 0), (95, 0), (60, 0), (80, 1)]:
    vn = f"Lanczos 480 + JPEG q{q} subsampling {['4:4:4', '4:2:2', '4:2:0'][sub]}"
    row = {}
    for n, ims in sets.items():
        c = (enc(model, [pre(small(im, q, sub)) for im in ims], half=True) * base[n]).sum(1).numpy()
        row[n] = dict(mean=round(float(c.mean()), 4), p5=round(float(np.percentile(c, 5)), 4), min=round(float(c.min()), 4))
    out[vn] = row; print(vn, row, flush=True)
json.dump(out, open("eval/rendition480_chroma.json", "w"), indent=1)
print("R480C_OK", flush=True)
