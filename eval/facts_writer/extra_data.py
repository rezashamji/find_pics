"""RESULTS 61b: ~10x more PUBLIC photos outside the benchmark for distilling the short facts writer.
Sources (licences): Open Images V4 train (every image CC BY 2.0 per its own list; fetched from the public CVDF S3 mirror)
and the Wikimedia Commons iPhone photos already on disk (data/public/raw/commons_iphone/img; Commons only hosts free
licences). EXCLUDED: every DISBench photo id (all 57 users = all complex_bench splits incl. test) and any Open Images photo
by one of those 57 Flickr users; then a perceptual near-duplicate check (64-bit dHash, Hamming <= 3) against all
109,467 DISBench photos. Stored as the benchmark's photos are: longest side 500 px, JPEG q92.
  python eval/facts_writer/extra_data.py manifest [--n_oi=300000]
  python eval/facts_writer/extra_data.py fetch --shard=k/K          (download + resize + dHash)
  python eval/facts_writer/extra_data.py dhash_disbench --shard=k/K
  python eval/facts_writer/extra_data.py dedupe                       -> data/public/extra_photos/final.parquet
"""
import io
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

OUT = Path("data/public/extra_photos")
OI_CSV = Path("data/public/raw/openimages_train/train-images-boxable-with-rotation.csv")


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def dhash(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def shrink(im, side=500):
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = side / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im


def manifest():
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    dis_ids = set(it.item_id.astype(str))
    dis_users = set(it.path.str.extract(r"/images/images/([^/]+)/")[0].dropna())
    df = pd.read_csv(OI_CSV, usecols=["ImageID", "OriginalLandingURL", "AuthorProfileURL", "License", "Rotation"])
    df["flickr_id"] = df.OriginalLandingURL.str.extract(r"/photos/[^/]+/(\d+)")[0]
    df["author"] = df.AuthorProfileURL.str.extract(r"/people/([^/]+)/")[0]
    n0 = len(df)
    drop_id = df.flickr_id.isin(dis_ids); drop_user = df.author.isin(dis_users)
    df = df[~drop_id & ~drop_user & df.License.str.contains("creativecommons.org/licenses/by/2.0", na=False)]
    n = int(arg("n_oi", "300000"))
    oi = df.sample(n=min(n, len(df)), random_state=0)
    oi = pd.DataFrame(dict(src="oi", id=oi.ImageID, url="https://open-images-dataset.s3.amazonaws.com/train/" + oi.ImageID + ".jpg",
                           rotation=oi.Rotation.fillna(0).astype(float), local=""))
    cm = sorted(Path("data/public/raw/commons_iphone/img").glob("*.jpg"))
    cm = pd.DataFrame(dict(src="commons", id=[p.stem for p in cm], url="", rotation=0.0, local=[str(p) for p in cm]))
    M = pd.concat([oi, cm], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    M.to_parquet(OUT / "manifest.parquet")
    print(f"Open Images train {n0:,}: dropped {int(drop_id.sum())} DISBench photo ids, {int(drop_user.sum())} photos by DISBench "
          f"users; sampled {len(oi):,}. Commons iPhone {len(cm):,}. Manifest {len(M):,}")


def fetch():
    k, K = map(int, arg("shard", "0/1").split("/"))
    M = pd.read_parquet(OUT / "manifest.parquet").iloc[k::K]
    f_out = OUT / "shards" / f"part{k}.parquet"
    if f_out.exists():
        print("done already"); return
    (OUT / "oi").mkdir(parents=True, exist_ok=True); (OUT / "commons").mkdir(parents=True, exist_ok=True)

    def one(r):
        dst = OUT / r.src / f"{r.id}.jpg"
        try:
            if dst.exists():
                im = Image.open(dst); im.load()
            else:
                if r.local:
                    im = Image.open(r.local)
                else:
                    b = None
                    for att in range(3):
                        try:
                            b = urllib.request.urlopen(r.url, timeout=30).read(); break
                        except Exception:
                            time.sleep(1 + att)
                    if b is None:
                        return (r.src, r.id, "", 0, 0, 0, False)
                    im = Image.open(io.BytesIO(b))
                im = shrink(im)
                if r.rotation:
                    im = im.rotate(-r.rotation, expand=True)
                im.save(dst, format="JPEG", quality=92)
            return (r.src, r.id, str(dst), im.width, im.height, dhash(im), True)
        except Exception:
            return (r.src, r.id, "", 0, 0, 0, False)
    t0 = time.time()
    with ThreadPoolExecutor(48) as ex:
        rows = list(ex.map(one, M.itertuples()))
    D = pd.DataFrame(rows, columns=["src", "id", "path", "w", "h", "dhash", "ok"])
    D["dhash"] = D.dhash.astype("uint64")
    f_out.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(f_out)
    print(f"shard {k}: {int(D.ok.sum())}/{len(D)} ok in {time.time() - t0:.0f}s", flush=True)


def dhash_disbench():
    k, K = map(int, arg("shard", "0/1").split("/"))
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"]).iloc[k::K]

    def one(p):
        try:
            return dhash(Image.open(p))
        except Exception:
            return 0
    with ThreadPoolExecutor(32) as ex:
        h = list(ex.map(one, it.path))
    (OUT / "disbench_dhash").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(item_id=it.item_id.astype(str), dhash=np.array(h, np.uint64))).to_parquet(OUT / "disbench_dhash" / f"part{k}.parquet")
    print("dhash disbench", len(it), flush=True)


def dedupe():
    D = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "shards").glob("part*.parquet"))], ignore_index=True)
    B = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "disbench_dhash").glob("part*.parquet"))], ignore_index=True)
    D = D[D.ok].reset_index(drop=True)
    bh = B.dhash.to_numpy(np.uint64); dh = D.dhash.to_numpy(np.uint64)
    # pigeonhole: Hamming <= 3 over 4 x 16-bit chunks -> at least one chunk identical
    near = np.zeros(len(D), bool)
    for c in range(4):
        sh = np.uint64(16 * c)
        bk = (bh >> sh) & np.uint64(0xFFFF); dk = (dh >> sh) & np.uint64(0xFFFF)
        idx = {}
        for j, v in enumerate(bk.tolist()):
            idx.setdefault(v, []).append(j)
        for i, v in enumerate(dk.tolist()):
            if near[i] or v not in idx:
                continue
            for j in idx[v]:
                if bin(int(dh[i]) ^ int(bh[j])).count("1") <= 3:
                    near[i] = True; break
    # also exact duplicates inside the extra set
    dup_self = D.dhash.duplicated() & (D.dhash != 0)
    keep = D[~near & ~dup_self].reset_index(drop=True)
    keep.to_parquet(OUT / "final.parquet")
    print(f"downloaded ok {len(D):,}; near-duplicates of a DISBench photo {int(near.sum())}; duplicates inside the set "
          f"{int(dup_self.sum())}; kept {len(keep):,} ({keep.src.value_counts().to_dict()})")


if __name__ == "__main__":
    {"manifest": manifest, "fetch": fetch, "dhash_disbench": dhash_disbench, "dedupe": dedupe}[sys.argv[1]]()
