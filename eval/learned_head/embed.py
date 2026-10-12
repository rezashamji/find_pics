"""M48 learned head: PE-Core-B-16 vectors, phone simulation, for every DISBench photo that eval/tiles_phone lacks
(whole + 2x2 tiles = 5 vectors per photo, exactly as eval_tiles_phone.embed16: 480 px Lanczos + JPEG q80 rendition,
crops cut from it), plus the text vectors of every question / look the learned-head experiment needs.

  python eval/learned_head/embed.py texts                -> eval/learned_head/texts.npz (+ texts.json)
  python eval/learned_head/embed.py images --shard=k/K   -> eval/learned_head/img_part{k}.npy + img_part{k}_ids.txt
Public data only (DISBench Flickr photos).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/learned_head")


def texts():
    from findpics.models import ImageTextEncoder
    T = json.load(open(OUT / "texts.json"))
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    V = np.concatenate([enc.texts(T[b:b + 256]) for b in range(0, len(T), 256)]).astype(np.float16)
    np.save(OUT / "texts.npy", V)
    print("TEXTS_DONE", V.shape, flush=True)


def images():
    from PIL import Image
    from eval_tiles_phone import rendition, crops
    from findpics.models import ImageTextEncoder
    k, K = map(int, next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")).split("/"))
    it = pd.read_parquet("data/public/index_disbench/items.parquet")
    it = it[it.media == "photo"]
    have = set(Path("eval/tiles_phone/ev16_ids.txt").read_text().split()) | set(Path("eval/tiles_phone/ids.txt").read_text().split())
    it = it[~it.item_id.astype(str).isin(have)].sort_values("item_id").reset_index(drop=True)
    it = it.iloc[k::K]
    ids, paths = it.item_id.astype(str).tolist(), it.path.tolist()
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    print("device", enc.device, enc.dtype, len(ids), flush=True)
    V = np.zeros((len(ids), 5, 1024), np.float16)
    for b in range(0, len(ids), 32):
        ims = [rendition(Image.open(p)) for p in paths[b:b + 32]]
        V[b:b + len(ims)] = enc.images([c for im in ims for c in crops(im)[:5]]).reshape(len(ims), 5, -1)
        if b % 2048 == 0:
            print(b, flush=True)
    np.save(OUT / f"img_part{k}.npy", V)
    (OUT / f"img_part{k}_ids.txt").write_text("\n".join(ids) + "\n")
    print("IMAGES_DONE", k, V.shape, flush=True)


if __name__ == "__main__":
    {"texts": texts, "images": images}[sys.argv[1]]()
