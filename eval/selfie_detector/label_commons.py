"""M46 step 1b: front/back label from EXIF for the harvested Commons iPhone photos.

MediaWiki drops EXIF LensModel, so the label is the lens the (Model, FocalLength, FNumber) triple identifies. Every
iPhone's front camera has its own focal length / aperture, distinct from its back lenses (the LensModel strings Apple
writes are e.g. "iPhone XR front camera 2.87mm f/2.2", "iPhone 12 front camera 2.71mm f/2.2",
"iPhone 14 front camera 2.69mm f/1.9"). Triples not in either table are EXCLUDED (unknown), never guessed.

Usage:
  python eval/selfie_detector/label_commons.py table META_DIR            # print (model, focal, f) counts
  python eval/selfie_detector/label_commons.py label META_DIR OUT.jsonl [max_per_user] [neg_per_pos]
"""
from __future__ import annotations

import collections
import glob
import json
import random
import re
import sys


def num(v):
    if v is None:
        return None
    try:
        a, b = str(v).split("/") if "/" in str(v) else (v, 1)
        return round(float(a) / float(b), 2)
    except Exception:
        return None


def norm_model(m):
    return re.sub(r"\s+", " ", (m or "").strip()).replace("Iphone", "iPhone")


def front_lens(model, f, fn):
    """True = front camera, False = a back camera, None = unknown (excluded). Measured on the harvest table (10-11):
    front = 2.6-2.9 mm at f/2.2 (iPhone 6 ... 13, SE 2/3), 2.0-3.0 mm at f/1.9 (iPhone 14 on), 2.1-2.2 mm at f/2.4
    (iPhone 5 / 5s / SE 1). Back = every lens >= 3.5 mm, the 1.5-1.6 mm ultra-wides (any aperture), and the
    2.22 mm f/2.2 ultra-wide (14 Pro ... 17; iPhone 16 / 17 non-Pro too: 87 of 247 iPhone 16 photos sit there, far
    above any front-camera share, so it is NOT the front lens)."""
    if f is None or fn is None or not model.startswith("iPhone"):
        return None
    if f >= 3.5 or 1.5 <= f <= 1.6 or (abs(f - 2.22) < 0.015 and abs(fn - 2.2) < 0.015):
        return False
    if abs(fn - 2.2) < 0.015 and 2.6 <= f <= 2.9:
        return True
    if abs(fn - 1.9) < 0.015 and 2.0 <= f <= 3.0:
        return True
    if abs(fn - 2.4) < 0.015 and (2.1 <= f <= 2.2 or abs(f - 1.85) < 0.015):   # 5/5s/SE1; 4/4S VGA front
        return True
    return None


def load(meta_dir):
    rows = {}
    for fn in glob.glob(f"{meta_dir}/*.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            rows[r["pageid"]] = r
    return list(rows.values())


def main():
    mode, meta_dir = sys.argv[1], sys.argv[2]
    rows = load(meta_dir)
    if mode == "table":
        c = collections.Counter()
        for r in rows:
            e = r["exif"]
            if e.get("Make") != "Apple":
                continue
            m = norm_model(e.get("Model"))
            c[(m, num(e.get("FocalLength")), num(e.get("FNumber")))] += 1
        for k, v in sorted(c.items(), key=lambda x: (x[0][0], -x[1])):
            if v >= 2:
                print(v, k, front_lens(*k))
        print("files", len(rows), "apple", sum(c.values()))
        return
    out_fn = sys.argv[3]
    max_per_user = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    neg_per_pos = float(sys.argv[5]) if len(sys.argv) > 5 else 8
    lab = []
    unknown = collections.Counter()
    for r in rows:
        e = r["exif"]
        if e.get("Make") != "Apple" or not r.get("thumb"):
            continue
        m = norm_model(e.get("Model"))
        k = front_lens(m, num(e.get("FocalLength")), num(e.get("FNumber")))
        if k is None:
            unknown[(m, num(e.get("FocalLength")), num(e.get("FNumber")))] += 1
            continue
        lab.append(dict(pageid=r["pageid"], title=r["title"], user=r["user"], label=int(k), model=m,
                        focal=num(e.get("FocalLength")), fnum=num(e.get("FNumber")), w=r["w"], h=r["h"],
                        thumb=r["thumb"], cat=r["cat"], date=e.get("DateTimeOriginal")))
    random.seed(0)
    random.shuffle(lab)
    per = collections.Counter()
    pos, neg = [], []
    for r in lab:                      # cap each uploader per label so a few heavy uploaders do not dominate
        if per[(r["user"], r["label"])] >= max_per_user:
            continue
        per[(r["user"], r["label"])] += 1
        (pos if r["label"] else neg).append(r)
    # MODEL-MATCHED negatives: up to neg_per_pos back-camera photos per front-camera photo OF THE SAME MODEL, so the
    # detector cannot win by recognising camera generation (old 3GS/4/4S backs vs newer fronts) instead of selfies.
    pos_by_model = collections.Counter(r["model"] for r in pos)
    taken = collections.Counter()
    neg_kept = []
    for r in neg:
        if taken[r["model"]] < neg_per_pos * pos_by_model[r["model"]]:
            taken[r["model"]] += 1
            neg_kept.append(r)
    neg = neg_kept
    with open(out_fn, "w") as f:
        for r in pos + neg:
            f.write(json.dumps(r) + "\n")
    print(f"labelled {len(lab)} (front {sum(r['label'] for r in lab)}), unknown {sum(unknown.values())}; "
          f"kept pos {len(pos)} neg {len(neg)} users {len(set(r['user'] for r in pos + neg))}")
    print("top unknown:", unknown.most_common(15))


if __name__ == "__main__":
    main()
