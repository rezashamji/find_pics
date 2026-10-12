"""Pack the DISBench photos of the 37 TRAINING libraries (every photo eval/tiles_phone lacks = the 20 test libraries'
photos are NOT here) into a few large files, because on 10-12 the lab Lustre served each small-file open at ~0.2-0.4 s
on some nodes (embed jobs ran at 2.3 photos/s) while large sequential reads were fast and parallel opens scaled.

  python eval/learned_head/pack.py [K=16] -> .cache/dpack/pack{k}.bin (raw JPEG bytes, concatenated), pack{k}.off.npy
                                             (int64 offsets, len n+1), pack{k}_ids.txt   (row order = embed.py order)
Read back: dpack.load(k) -> (ids, offsets, memoryview) ; dpack.image(buf, off, i) -> PIL image (bytes are the original
file, so decoding them is identical to Image.open(path)).
Public data only.
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics")
PACK = ROOT / ".cache/dpack"


def pool_items():
    import pandas as pd
    it = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet")
    it = it[it.media == "photo"]
    have = set((ROOT / "eval/tiles_phone/ev16_ids.txt").read_text().split()) | \
        set((ROOT / "eval/tiles_phone/ids.txt").read_text().split())
    it = it[~it.item_id.astype(str).isin(have)].sort_values("item_id").reset_index(drop=True)
    return it.item_id.astype(str).tolist(), it.path.tolist()


def load(k):
    """Pack k fully into RAM with one sequential read: (ids, offsets, bytes)."""
    ids = (PACK / f"pack{k}_ids.txt").read_text().split()
    off = np.load(PACK / f"pack{k}.off.npy")
    with open(PACK / f"pack{k}.bin", "rb") as f:
        buf = f.read()
    return ids, off, buf


def load_all(K=16):
    """{item_id: jpeg bytes} for every packed photo (~10 GB in RAM)."""
    out = {}
    for k in range(K):
        ids, off, buf = load(k)
        mv = memoryview(buf)
        for i, u in enumerate(ids):
            out[u] = mv[off[i]:off[i + 1]]
    return out


def image(b):
    from PIL import Image
    return Image.open(io.BytesIO(bytes(b)))


def read(p):
    for _ in range(3):
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            time.sleep(1)
    return b""


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ids, paths = pool_items()
    PACK.mkdir(parents=True, exist_ok=True)
    print("photos", len(ids), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(96) as ex:
        for k in range(K):
            if (PACK / f"pack{k}_ids.txt").exists():
                continue
            sel = list(range(k, len(ids), K))
            blobs = list(ex.map(read, [paths[i] for i in sel]))
            off = np.zeros(len(sel) + 1, np.int64); off[1:] = np.cumsum([len(b) for b in blobs])
            tmp = PACK / f".pack{k}.bin.tmp"
            with open(tmp, "wb") as f:
                for b in blobs:
                    f.write(b)
            tmp.rename(PACK / f"pack{k}.bin")
            np.save(PACK / f"pack{k}.off.npy", off)
            (PACK / f"pack{k}_ids.txt").write_text("\n".join(ids[i] for i in sel) + "\n")
            print(f"pack {k}: {len(sel)} photos, {off[-1] / 1e9:.2f} GB, empty {sum(len(b) == 0 for b in blobs)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print("PACK_DONE", flush=True)


if __name__ == "__main__":
    main()
