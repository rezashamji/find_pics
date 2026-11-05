"""M82 cluster anchor: a NEW Wikimedia Commons iPhone photo set that no model of ours has seen.

Why new: all 17,594 Commons iPhone photos on disk (data/public/raw/commons_iphone/img) went into the trained writer's
extra training data (RESULTS 61b/61d), so they are not held out. The 2026-10-11 harvest's metadata
(data/public/raw/commons_iphone/meta) lists ~197k Commons iPhone files; only 17,594 were downloaded.

SELECTION RULE (label-independent, fixed before any model output was seen):
  file in a "Category:Taken with (Apple) iPhone ..." category (not a "front camera" category), JPEG,
  pageid NOT among the 17,594 on disk, EXIF DateTimeOriginal year 2019-2026 (rules out Open Images V4 and DISBench /
  YFCC100M photos, which are all older), long side >= 1280 px;
  shuffled with seed 82, at most 20 photos per uploader, first N (default 3,000).
Then the Commons API adds licence, author, GPS (EXIF) and the date; the 1280 px standard thumbnail is downloaded.

  python eval/device_anchor82/commons_new.py [--n=3000]
  -> data/public/device_anchor82/commons_manifest.parquet, data/public/device_anchor82/commons_img/<pageid>.jpg
"""
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

RAW = Path("data/public/raw/commons_iphone")
OUT = Path("data/public/device_anchor82")
API = "https://commons.wikimedia.org/w/api.php"
UA = "find_pics-research/0.1 (device anchor eval; rezamshamji@gmail.com) python-requests"
S = requests.Session()
S.headers["User-Agent"] = UA
_last = [0.0]


def api(params, tries=10):
    params = {**params, "format": "json", "maxlag": 5}
    for k in range(tries):
        try:
            time.sleep(max(0.0, _last[0] + 0.4 - time.time()))
            _last[0] = time.time()
            r = S.get(API, params=params, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if "error" in d and d["error"].get("code") == "maxlag":
                    time.sleep(5 + 5 * k); continue
                return d
            time.sleep(min(60, 2 ** k))
        except Exception as e:
            print("retry", k, e, flush=True); time.sleep(min(60, 2 ** k))
    raise RuntimeError("API failed")


def year(r):
    m = re.match(r"(\d{4}):\d\d:\d\d", (r.get("exif") or {}).get("DateTimeOriginal", "") or "")
    return int(m.group(1)) if m else None


def strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def main():
    n = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--n=")), 3000))
    disk = {int(f[:-4]) for f in os.listdir(RAW / "img")}
    rows = {}
    for f in sorted(glob.glob(str(RAW / "meta" / "*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.setdefault(r["pageid"], r)
    pool = [r for k, r in sorted(rows.items()) if k not in disk and r["cat"].startswith("Category:Taken with")
            and "front camera" not in r["cat"] and 2019 <= (year(r) or 0) <= 2026 and max(r["w"], r["h"]) >= 1280]
    random.Random(82).shuffle(pool)
    per, pick = {}, []
    for r in pool:
        if per.get(r["user"], 0) >= 20:
            continue
        per[r["user"]] = per.get(r["user"], 0) + 1
        pick.append(r)
        if len(pick) >= n:
            break
    print(f"pool {len(pool)} photos ({len({r['user'] for r in pool})} uploaders); picked {len(pick)} "
          f"({len(per)} uploaders)", flush=True)
    meta = {}
    for s in range(0, len(pick), 50):
        ids = "|".join(str(r["pageid"]) for r in pick[s:s + 50])
        d = api(dict(action="query", pageids=ids, prop="imageinfo", iiprop="url|size|extmetadata", iiurlwidth=1280,
                     iiextmetadatafilter="LicenseShortName|Artist|DateTimeOriginal|GPSLatitude|GPSLongitude"))
        for p in d["query"]["pages"].values():
            ii = (p.get("imageinfo") or [{}])[0]
            em = {k: v.get("value") for k, v in (ii.get("extmetadata") or {}).items()}
            meta[p["pageid"]] = dict(url=ii.get("url"), descurl=ii.get("descriptionurl"), thumb1280=ii.get("thumburl"),
                                     license=em.get("LicenseShortName"), artist=strip_html(em.get("Artist")),
                                     gps_lat=em.get("GPSLatitude"), gps_lon=em.get("GPSLongitude"))
    print("api metadata", len(meta), flush=True)
    img = OUT / "commons_img"
    img.mkdir(parents=True, exist_ok=True)

    def fetch(r):
        dst = img / f"{r['pageid']}.jpg"
        if dst.exists() and dst.stat().st_size > 0:
            return True
        u = (meta.get(r["pageid"]) or {}).get("thumb1280")
        if not u:
            return False
        for k in range(5):
            try:
                x = S.get(u, timeout=60)
                if x.status_code == 200 and x.headers.get("content-type", "").startswith("image/jpeg"):
                    tmp = dst.with_suffix(".part")
                    tmp.write_bytes(x.content); tmp.rename(dst)
                    return True
                if x.status_code == 429:
                    time.sleep(10 * (k + 1)); continue
                time.sleep(2 ** k)
            except Exception:
                time.sleep(2 ** k)
        return False
    t0 = time.time(); ok = []
    with ThreadPoolExecutor(4) as ex:
        for i, o in enumerate(ex.map(fetch, pick)):
            ok.append(o)
            if (i + 1) % 250 == 0:
                print(f"{time.time() - t0:6.0f}s {i + 1}/{len(pick)} ok {sum(ok)}", flush=True)
    M = []
    for r, o in zip(pick, ok):
        m = meta.get(r["pageid"]) or {}
        M.append(dict(key=f"c{r['pageid']}", pageid=r["pageid"], title=r["title"], user=r["user"], w=r["w"], h=r["h"],
                      model=(r.get("exif") or {}).get("Model"), taken_exif=(r.get("exif") or {}).get("DateTimeOriginal"),
                      cat=r["cat"], path=str((img / f"{r['pageid']}.jpg").resolve()) if o else "", ok=o, **m))
    M = pd.DataFrame(M)
    M.to_parquet(OUT / "commons_manifest.parquet")
    print(f"COMMONS_DONE {int(M.ok.sum())}/{len(M)} downloaded, GPS {int(M.gps_lat.notna().sum())}", flush=True)


if __name__ == "__main__":
    main()
