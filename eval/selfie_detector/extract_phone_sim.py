"""M50 follow-up: PE-Core-B-16 vectors of the Commons selfie set through the PHONE's indexing path (GPU job).

Why: the M50 weights were fitted on vectors of 1,280 px thumbnails. The phone stores the vector of the local ~448 px
PhotoKit rendition (PhotoLibrary.read side 448, resizeMode .fast, aspectFit), and on Reza's phone logit 0.5 gave a
23% false-positive rate vs 0.84% on Commons. Simulation (docs/PHONE_PARITY.md, eval/eval_tiles_phone.rendition):
long side -> S with Lanczos, JPEG round trip (Pillow, quality q, its default 4:2:0 chroma unless 444), then the
phone's preprocessing = open_clip's PE-Core-B-16 transform (Pillow bilinear squash to 224, no crop) and the fp16
image tower (ImageTextEncoder on GPU = fp16). PhotoKit's real encoder settings are unknown, hence several variants.

Variants (key -> (long side, JPEG quality, 4:4:4?)):
  r448q80   448, q80, 4:2:0   <- primary phone simulation
  r480q80   480, q80, 4:2:0   (the project's earlier rendition480 convention)
  r448q95   448, q95, 4:4:4   (the gentlest plausible encoder)
  r448raw   448, no JPEG      (separates SIZE from JPEG damage)
Source = the 1,280 px Commons thumbnail (data/public/raw/commons_iphone/img).

Usage: python eval/selfie_detector/extract_phone_sim.py LABELS.jsonl IMG_DIR OUT.npz
OUT.npz: pageid[N] + one [N,1024] fp16 array per variant (same row order as pageid).
"""
from __future__ import annotations

import faulthandler
faulthandler.dump_traceback_later(600, repeat=True)
print("start", flush=True)
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

VARIANTS = {"r448q80": (448, 80, False), "r480q80": (480, 80, False), "r448q95": (448, 95, True),
            "r448raw": (448, None, False)}


def rendition(im, side, q, full_chroma):
    im = im.copy()
    im.thumbnail((side, side), Image.LANCZOS)
    if q is None:
        return im
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=0 if full_chroma else 2)
    b.seek(0)
    return Image.open(b).convert("RGB")


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    img_dir, out = sys.argv[2], sys.argv[3]
    from findpics.models import ImageTextEncoder
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    print("PE-Core loaded", enc.device, enc.dtype, len(rows), "photos", flush=True)

    def load(r):
        fn = os.path.join(img_dir, f"{r['pageid']}.jpg")
        with Image.open(fn) as im:
            im = im.convert("RGB")
            return r["pageid"], {k: rendition(im, *v) for k, v in VARIANTS.items()}

    ids, vecs = [], {k: [] for k in VARIANTS}
    sizes = []
    t0 = time.time()
    B = 64
    with ThreadPoolExecutor(12) as ex:
        for s in range(0, len(rows), B):
            batch = list(ex.map(load, rows[s:s + B]))
            for k in VARIANTS:
                vecs[k].append(enc.images([v[k] for _, v in batch]))
            ids += [p for p, _ in batch]
            sizes += [v["r448q80"].size for _, v in batch]
            if s % (B * 20) == 0:
                print(f"{time.time() - t0:6.0f}s {s}/{len(rows)}", flush=True)
    np.savez(out, pageid=np.array(ids), sizes=np.array(sizes), **{k: np.concatenate(v) for k, v in vecs.items()})
    print("saved", out, len(ids), flush=True)


if __name__ == "__main__":
    main()
