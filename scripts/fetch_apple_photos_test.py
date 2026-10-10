#!/usr/bin/env python3
"""Mac side of the Apple Photos head-to-head (MAC_INBOX M29): fetch the 7,886 test photos from the public YFCC100M S3
bucket (no login, no cluster, no 2FA) and check every file's sha256 against the manifest built on the cluster.

Python standard library only (macOS /usr/bin/python3 is enough). Resumable: files already present with the right
sha256 are skipped. Run from the repo root:
    python3 scripts/fetch_apple_photos_test.py            # -> data/public/apple_photos_test/images/<item_id>.jpg
Exit code 0 only if all 7,886 files are present and byte-identical to the cluster's copy.
"""
import concurrent.futures as cf
import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

MAN = Path("eval/apple_photos/manifest.tsv")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/public/apple_photos_test/images")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def one(r):
    dst = OUT / r["filename"]
    if dst.exists() and sha(dst.read_bytes()) == r["sha256"]:
        return "have"
    err = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(r["url"], timeout=30) as f:
                b = f.read()
            if sha(b) != r["sha256"]:
                err = "sha256 mismatch"
                continue
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(b)
            tmp.replace(dst)
            return "got"
        except Exception as e:   # network hiccup: retry with backoff
            err = repr(e)
            time.sleep(2 * (attempt + 1))
    return f"FAILED {r['filename']}: {err}"


def main():
    rows = list(csv.DictReader(open(MAN), delimiter="\t"))
    OUT.mkdir(parents=True, exist_ok=True)
    t0, counts, failed = time.time(), {"have": 0, "got": 0}, []
    with cf.ThreadPoolExecutor(16) as ex:
        for i, s in enumerate(ex.map(one, rows)):
            if s in counts:
                counts[s] += 1
            else:
                failed.append(s)
            if (i + 1) % 500 == 0 or i + 1 == len(rows):
                print(f"{i + 1}/{len(rows)}  already had {counts['have']}, downloaded {counts['got']}, "
                      f"failed {len(failed)}  ({time.time() - t0:.0f} s)", flush=True)
    for f in failed[:20]:
        print(f)
    n_ok = counts["have"] + counts["got"]
    total = sum(int(r["bytes"]) for r in rows)
    print(f"DONE: {n_ok}/{len(rows)} files present and sha256-identical ({total / 1e6:.0f} MB) in {OUT}")
    sys.exit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
