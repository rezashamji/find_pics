"""M46 step 1c: download the 1280-px Commons thumbnails (EXIF-rotated by Commons) for the labelled set.

Usage: python eval/selfie_detector/download_thumbs.py LABELS.jsonl IMG_DIR [threads]
Each line of LABELS.jsonl needs pageid + thumb. Output IMG_DIR/<pageid>.jpg. Resumable (skips existing files).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

UA = "find_pics-research/0.1 (selfie-detector eval; rezamshamji@gmail.com) python-requests"
S = requests.Session()
S.headers["User-Agent"] = UA


def fetch(row, img_dir):
    fn = os.path.join(img_dir, f"{row['pageid']}.jpg")
    if os.path.exists(fn):
        return "skip"
    for k in range(6):
        try:
            r = S.get(row["thumb"], timeout=60)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                tmp = fn + ".part"
                open(tmp, "wb").write(r.content)
                os.replace(tmp, fn)
                return "ok"
            if r.status_code in (404, 410):
                return f"http{r.status_code}"
            time.sleep(min(60, 3 * 2 ** k))
        except Exception:
            time.sleep(min(60, 3 * 2 ** k))
    return "fail"


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    img_dir = sys.argv[2]
    th = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    os.makedirs(img_dir, exist_ok=True)
    t0, cnt = time.time(), {}
    with ThreadPoolExecutor(th) as ex:
        for i, s in enumerate(ex.map(lambda r: fetch(r, img_dir), rows)):
            cnt[s] = cnt.get(s, 0) + 1
            if i % 500 == 0:
                print(f"{time.time() - t0:6.0f}s {i}/{len(rows)} {cnt}", flush=True)
    print("done", cnt, flush=True)


if __name__ == "__main__":
    main()
