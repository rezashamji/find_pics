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


def images_pack():
    """Same vectors as images(), read from eval/learned_head/pack.py's packed files (one sequential read each) instead
    of 71k small-file opens (10-12 Lustre: ~0.3-0.4 s per open on some nodes). Writes every missing img_part{k}.npy
    (k = 0..7, embed.py's own order: sorted item_id, k::8). Decoding and crops run on all cores."""
    import io
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from eval_tiles_phone import rendition, crops
    from findpics.models import ImageTextEncoder
    sys.path.insert(0, "eval/learned_head")
    from pack import load_all, pool_items
    ids_all, _ = pool_items()
    todo = [k for k in range(8) if not (OUT / f"img_part{k}.npy").exists()]
    if not todo:
        print("IMAGES_DONE all exist", flush=True); return
    raw = load_all()
    print("pack loaded", len(raw), "todo", todo, flush=True)
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    ex = ThreadPoolExecutor(int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 8)))

    def prep(u):
        im = rendition(Image.open(io.BytesIO(bytes(raw[u]))))
        return [enc.preprocess(c) for c in crops(im)[:5]]
    import torch
    for k in todo:
        ids = ids_all[k::8]
        V = np.zeros((len(ids), 5, 1024), np.float16)
        for b in range(0, len(ids), 256):
            xs = list(ex.map(prep, ids[b:b + 256]))
            x = torch.stack([c for cs in xs for c in cs]).to(enc.device, enc.dtype)
            with torch.inference_mode():     # = ImageTextEncoder.images (normalise in fp32)
                f = torch.nn.functional.normalize(enc.model.encode_image(x).float(), dim=-1)
            V[b:b + len(xs)] = f.cpu().numpy().astype(np.float16).reshape(len(xs), 5, -1)
        np.save(OUT / f"img_part{k}.npy", V)
        (OUT / f"img_part{k}_ids.txt").write_text("\n".join(ids) + "\n")
        print("IMAGES_DONE", k, V.shape, flush=True)


def check_pack():
    """Parity: 64 photos of img_part1 (made by images() from the original files) re-embedded the images_pack() way."""
    import io
    from PIL import Image
    from eval_tiles_phone import rendition, crops
    from findpics.models import ImageTextEncoder
    sys.path.insert(0, "eval/learned_head")
    from pack import load_all
    ids = (OUT / "img_part1_ids.txt").read_text().split()[:64]
    V = np.load(OUT / "img_part1.npy")[:64].astype(np.float32)
    raw = load_all()
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    W = enc.images([c for u in ids for c in crops(rendition(Image.open(io.BytesIO(bytes(raw[u])))))[:5]]).reshape(64, 5, -1)
    cs = (V * W.astype(np.float32)).sum(-1)
    print(f"PACK_PARITY 64 photos x 5 crops: cosine min {cs.min():.5f} mean {cs.mean():.5f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check_pack":
        check_pack(); sys.exit()
    {"texts": texts, "images": images, "images_pack": images_pack}[sys.argv[1]]()
