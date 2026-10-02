"""Build a public test library that mimics a personal photo library, with known answers.

data/public/testlib/
  library/openimages/*.jpg   symlinks to Open Images val images (scenes, food, pets; human-verified labels)
  library/imdb/<year>/*.jpg  IMDB scene photos of people across years (name + photo year)
  library/videos/*.mp4       symlinks to Pexels clips
  library_metadata.json      osxphotos-style records (uuid, date, persons, labels, ismovie) -> exercises the Apple path.
                             `persons` simulates Apple's People tags: only ~50% of each family member's photos are
                             tagged (biased to big, clear faces, like a conservative real tagger).
  ground_truth.json          {"person:<name>": [uuids], "concept:<label>": {"pos": [...], "neg": [...]}}
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(os.environ["FP_ROOT"])
RAW = ROOT / "data/public/raw"
OUT = ROOT / "data/public/testlib"
LIB = OUT / "library"
rng = random.Random(0)
CONCEPTS = ['Bread', 'Baked goods', 'Pizza', 'Cake', 'Sandwich', 'Dog', 'Bicycle', 'Guitar', 'Christmas tree',
            'Swimming pool', 'Cat', 'Horse', 'Sunglasses', 'Wine glass', 'Coffee cup']


def rand_date(y0, y1):
    d0 = date(y0, 1, 1); span = (date(y1, 12, 31) - d0).days
    return d0 + timedelta(days=rng.randrange(span))


def main(n_family: int = 6, min_photos: int = 30, tag_frac: float = 0.5):
    meta, gt = [], {}
    (LIB / "openimages").mkdir(parents=True, exist_ok=True)
    (LIB / "imdb").mkdir(parents=True, exist_ok=True)
    (LIB / "videos").mkdir(parents=True, exist_ok=True)

    # ---- Open Images: concept ground truth
    oi = RAW / "openimages"
    cls = dict(csv.reader(open(oi / "classes.csv")))
    inv = {v: k for k, v in cls.items()}
    sel = [l.strip() for l in open(oi / "selected_ids.txt") if l.strip()]
    sel_set = set(sel)
    pos, neg = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(oi / "val_labels.csv")):
        if r["ImageID"] in sel_set and cls.get(r["LabelName"]) in CONCEPTS:
            (pos if r["Confidence"] == "1" else neg)[cls[r["LabelName"]]].append("oi_" + r["ImageID"])
    for iid in sel:
        src = oi / "images" / f"{iid}.jpg"
        if not src.exists():
            continue
        dst = LIB / "openimages" / f"oi_{iid}.jpg"
        if not dst.exists():
            dst.symlink_to(src)
        meta.append(dict(uuid=f"oi_{iid}", date=rand_date(2012, 2026).isoformat() + "T12:00:00", persons=[], labels=[], ismovie=False))
    for c in CONCEPTS:
        gt[f"concept:{c}"] = dict(pos=sorted(set(pos[c])), neg=sorted(set(neg[c])))

    # ---- IMDB scenes: identity across years
    shards = sorted((RAW / "imdb_wiki/imdb").glob("*.parquet"))
    cols = ["name", "photo_taken", "face_score", "second_face_score", "face_location", "full_path"]
    tabs = [pq.read_table(s, columns=cols).to_pandas().assign(shard=str(s), row=lambda d: np.arange(len(d))) for s in shards]
    df = pd.concat(tabs, ignore_index=True)
    df = df[np.isfinite(df["face_score"].astype(float))]
    df["year"] = pd.to_datetime(df["photo_taken"]).dt.year
    stats = df.groupby("name").agg(n=("year", "size"), y0=("year", "min"), y1=("year", "max"))
    stats["span"] = stats["y1"] - stats["y0"]
    fam = stats[(stats.n >= min_photos) & (stats.span >= 5)].sort_values(["span", "n"], ascending=False).head(n_family)
    print("family members:\n", fam)
    # write every IMDB image (family + everyone else as distractors)
    by_shard = df.groupby("shard")
    for s, g in by_shard:
        t = pq.read_table(s, columns=["image"]).column("image").to_pylist()
        for _, r in g.iterrows():
            uid = f"imdb_{Path(s).stem.split('-')[1]}_{int(r['row'])}"
            y = int(r["year"]) if 1950 < int(r["year"]) < 2030 else 2010
            d = LIB / "imdb" / str(y); d.mkdir(exist_ok=True)
            p = d / f"{uid}.jpg"
            if not p.exists():
                img = t[int(r["row"])]
                p.write_bytes(img["bytes"] if isinstance(img, dict) else img)
            is_fam = r["name"] in fam.index
            meta.append(dict(uuid=uid, date=f"{y}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T12:00:00",
                             persons=[], labels=[], ismovie=False, _name=r["name"] if is_fam else None,
                             _face_score=float(r["face_score"]), _second=float(r["second_face_score"]) if pd.notna(r["second_face_score"]) else None))
    # simulated Apple tags: tag the clearest tag_frac of each family member's photos
    for name in fam.index:
        recs = [m for m in meta if m.get("_name") == name]
        gt[f"person:{name}"] = sorted(m["uuid"] for m in recs)
        recs.sort(key=lambda m: -m["_face_score"])
        for m in recs[: int(len(recs) * tag_frac)]:
            m["persons"] = [name]
    for m in meta:
        for k in ("_name", "_face_score", "_second"):
            m.pop(k, None)

    # ---- Pexels videos
    vids = sorted((RAW / "pexels_videos").glob("*.mp4"))
    rng.shuffle(vids)
    for v in vids[:120]:
        uid = "px_" + v.stem
        dst = LIB / "videos" / f"{uid}.mp4"
        if not dst.exists():
            dst.symlink_to(v)
        meta.append(dict(uuid=uid, date=rand_date(2016, 2026).isoformat() + "T12:00:00", persons=[], labels=[], ismovie=True))

    (OUT / "library_metadata.json").write_text(json.dumps(meta))
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    print(f"items: {len(meta)}; family: {list(fam.index)}; concepts: {[ (c, len(gt['concept:'+c]['pos'])) for c in CONCEPTS]}")


if __name__ == "__main__":
    main()
