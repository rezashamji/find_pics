"""Build the search index: one pass over the library, done once (like Apple's overnight analysis).

Layout of an index dir:
  items.parquet                 one row per photo/video (from ingest.scan)
  shards/NNNN/units.parquet     one row per analyzed image unit (photo, or a sampled video frame)
  shards/NNNN/clip.npy          [n_units, D] float16 L2-normalized image vectors
  shards/NNNN/faces.parquet     one row per detected face (unit index, bbox, det score, size)
  shards/NNNN/face_emb.npy      [n_faces, 512] float16 L2-normalized identity vectors
  shards/NNNN/DONE              written last; a shard with DONE is skipped on rerun (idempotent, requeue-safe)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _iter_units(rows):
    from .media import load_image, sample_video_frames
    for r in rows:
        try:
            if r["media"] == "video":
                for t, im in sample_video_frames(r["path"]):
                    yield r["item_id"], t, im, None
            else:
                yield r["item_id"], -1.0, load_image(r["path"]), None
        except Exception as e:  # count + keep examples of failures; never crash the shard
            yield r["item_id"], -1.0, None, f"{type(e).__name__}: {e}"


class _UnitDataset:
    """Torch IterableDataset that splits a shard's rows across DataLoader workers (CPU decoding in parallel)."""

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        import torch.utils.data as tud
        wi = tud.get_worker_info()
        rows = self.rows if wi is None else self.rows[wi.id::wi.num_workers]
        for item_id, t, im, err in _iter_units(rows):
            yield item_id, t, (np.asarray(im) if im is not None else None), err


def index_shard(index_dir: Path, shard: int, n_shards: int, clip_name: str | None, face_name: str | None,
                batch: int = 64, workers: int = 8):
    import torch
    import torch.utils.data as tud
    from .models import ImageTextEncoder, FaceEncoder, DEFAULT_CLIP, DEFAULT_FACE
    from PIL import Image

    out = index_dir / "shards" / f"{shard:04d}"
    if (out / "DONE").exists():
        print(f"shard {shard} already done; skipping", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(index_dir / "items.parquet")
    rows = items.iloc[shard::n_shards].to_dict("records")
    print(f"shard {shard}/{n_shards}: {len(rows)} items", flush=True)

    names = (clip_name or DEFAULT_CLIP).split(",")
    encs = [ImageTextEncoder(n) for n in names]  # first = default (clip.npy); others -> clip_<k>.npy
    face = FaceEncoder(face_name or DEFAULT_FACE)
    class TorchDS(tud.IterableDataset):  # Linux fork start: no pickling of this local class needed
        def __iter__(self):
            return iter(_UnitDataset(rows))
    ds = TorchDS()
    dl = tud.DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=8 if workers else None)

    units, faces, face_vecs, errors = [], [], [], []
    clip_vecs = [[] for _ in encs]
    buf_ims, buf_meta = [], []
    t0 = time.time()

    def flush():
        if not buf_ims:
            return
        for k, e in enumerate(encs):
            clip_vecs[k].append(e.images(buf_ims))
        units.extend(buf_meta)
        buf_ims.clear(); buf_meta.clear()

    for item_id, t, arr, err in dl:
        if err is not None or arr is None:
            errors.append(dict(item_id=item_id, error=err))
            continue
        im = Image.fromarray(arr)
        u = len(units) + len(buf_meta)
        for f in face.faces(im):
            x1, y1, x2, y2 = f["bbox"]
            faces.append(dict(unit=u, item_id=item_id, frame_t=t, x1=x1, y1=y1, x2=x2, y2=y2,
                              det_score=f["det_score"], face_px=float(min(x2 - x1, y2 - y1)),
                              img_w=im.width, img_h=im.height))
            face_vecs.append(f["emb"])
        buf_ims.append(im); buf_meta.append(dict(item_id=item_id, frame_t=t, w=im.width, h=im.height))
        if len(buf_ims) >= batch:
            flush()
            n = len(units)
            if n % (batch * 10) == 0:
                print(f"  {n} units, {len(faces)} faces, {len(errors)} errors, {n/(time.time()-t0):.1f} units/s", flush=True)
    flush()

    pd.DataFrame(units).to_parquet(out / "units.parquet")
    for k, e in enumerate(encs):
        arr = np.concatenate(clip_vecs[k]) if clip_vecs[k] else np.zeros((0, 1), np.float16)
        np.save(out / ("clip.npy" if k == 0 else f"clip_{k}.npy"), arr)
    pd.DataFrame(faces).to_parquet(out / "faces.parquet")
    np.save(out / "face_emb.npy", np.stack(face_vecs) if face_vecs else np.zeros((0, 512), np.float16))
    pd.DataFrame(errors).to_parquet(out / "errors.parquet")
    stats = dict(shard=shard, items=len(rows), units=len(units), faces=len(faces), errors=len(errors),
                 seconds=round(time.time() - t0, 1), clip_models=names, face_model=face_name or DEFAULT_FACE)
    (out / "stats.json").write_text(json.dumps(stats))
    (out / "DONE").write_text("ok")
    print("DONE", stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--face", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    try:
        index_shard(Path(a.index_dir), a.shard, a.n_shards, a.clip, a.face, workers=a.workers)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
