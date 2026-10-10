"""Package the recall-audit test set (RESULTS 34: 4 DISBench libraries, 7,886 photos) for the Apple Photos head-to-head.

Writes
  data/public/apple_photos_test/images/<item_id>.jpg   the original bytes (copied, never re-encoded; git-ignored)
  data/public/apple_photos_test/manifest.tsv           filename, item_id, library, bytes, sha256, url
  eval/apple_photos/manifest.tsv                       the same manifest, committed: the Mac fetches with it
                                                       (scripts/fetch_apple_photos_test.py), no rsync / 2FA needed
The label key (eval/recall_audit/key*.json, labels.txt) is NOT copied into the package.
url = YFCC100M's public S3 bucket (multimedia-commons), the source DISBench's own download_images.py uses; checked
byte-identical to our copy by sha256 (--verify-s3 downloads every url and compares).

  python scripts/build_apple_photos_test.py [--verify-s3]
"""
import concurrent.futures as cf
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path("data/public/raw/disbench")
PKG = Path("data/public/apple_photos_test")
MAN = Path("eval/apple_photos/manifest.tsv")


def s3_url(h):
    return f"https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/{h[:3]}/{h[3:6]}/{h}.jpg"


def build():
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores").glob("part*.parquet"))])
    items = sc.drop_duplicates("item_id")[["item_id", "user", "path"]].sort_values(["user", "item_id"])
    assert len(items) == 7886 and items.item_id.is_unique, len(items)
    hashes = {}
    for u in items.user.unique():
        for line in open(RAW / "photo_ids" / f"{u}.txt"):
            p = line.rstrip("\n").split("\t")
            hashes[p[0]] = p[1]
    (PKG / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in items.itertuples():
        dst = PKG / "images" / f"{r.item_id}.jpg"
        if not dst.exists():
            shutil.copyfile(r.path, dst)
        b = dst.read_bytes()
        rows.append(dict(filename=dst.name, item_id=r.item_id, library=r.user, bytes=len(b),
                         sha256=hashlib.sha256(b).hexdigest(), url=s3_url(hashes[r.item_id])))
    m = pd.DataFrame(rows)
    m.to_csv(PKG / "manifest.tsv", sep="\t", index=False)
    MAN.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(MAN, sep="\t", index=False)
    print(f"{len(m)} photos, {m.bytes.sum() / 1e6:.1f} MB, libraries: {m.library.value_counts().to_dict()}")
    return m


def verify_s3(m):
    def one(r):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(r.url, timeout=30) as f:
                    return r.filename, hashlib.sha256(f.read()).hexdigest() == r.sha256
            except Exception as e:
                err = e
        return r.filename, f"error {err}"
    bad = []
    with cf.ThreadPoolExecutor(16) as ex:
        for i, (fn, ok) in enumerate(ex.map(one, m.itertuples())):
            if ok is not True:
                bad.append((fn, ok))
            if (i + 1) % 1000 == 0:
                print(f"verified {i + 1}/{len(m)}, mismatched/failed so far {len(bad)}", flush=True)
    print(f"S3 vs our bytes: {len(m) - len(bad)}/{len(m)} identical; bad: {bad[:20]}")


if __name__ == "__main__":
    man = build()
    if "--verify-s3" in sys.argv:
        verify_s3(man)
