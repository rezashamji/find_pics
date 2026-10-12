"""M46 step 1a: harvest EXIF metadata of iPhone photos on Wikimedia Commons (public, free-licensed).

Why Commons: the local public sets (DISBench / Open Images) are Flickr resizes with EXIF stripped (0/150 sampled files
had any EXIF). Commons keeps the original's EXIF and serves it through the API without downloading the file.
MediaWiki does NOT keep EXIF LensModel, so the front/back label is derived later (label_commons.py) from
Make + Model + FocalLength + FNumber, each iPhone model's front camera having its own focal length.

Usage: python eval/selfie_detector/harvest_commons.py OUT_DIR [per_category_cap]
Writes OUT_DIR/meta/<category>.jsonl (one line per file: title, uploader, size, thumb url, EXIF subset).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = "find_pics-research/0.1 (selfie-detector eval; rezamshamji@gmail.com) python-requests"
KEEP = ["Make", "Model", "FocalLength", "FNumber", "FocalLengthIn35mmFilm", "DateTimeOriginal", "Orientation",
        "Software", "DigitalZoomRatio"]
S = requests.Session()
S.headers["User-Agent"] = UA


_last = [0.0]
MIN_GAP = 0.4   # <= 150 requests/min: Wikimedia's limit for an identified unauthenticated client is 200/min


def get(params, tries=12):
    params = {**params, "format": "json", "maxlag": 5}
    for k in range(tries):
        try:
            time.sleep(max(0.0, _last[0] + MIN_GAP - time.time()))
            _last[0] = time.time()
            r = S.get(API, params=params, timeout=60)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                w = float(ra) if ra and ra.isdigit() else min(120, 5 * 2 ** k)
                print("429, waiting", w, flush=True); time.sleep(w); continue
            if r.status_code == 200:
                d = r.json()
                if "error" in d and d["error"].get("code") == "maxlag":
                    time.sleep(5 + 5 * k); continue
                return d
            time.sleep(min(60, 2 ** k))
        except Exception as e:  # network blip
            print("retry", k, e, flush=True); time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"API failed: {params}")


def categories():
    cats, off = set(), 0
    while True:
        d = get(dict(action="query", list="search", srnamespace=14, srlimit=50, sroffset=off,
                     srsearch='intitle:"Taken with" intitle:iPhone'))
        for c in d["query"]["search"]:
            cats.add(c["title"])
        if "continue" not in d:
            break
        off = d["continue"]["sroffset"]
    keep = sorted(c for c in cats if re.match(r"Category:Taken with (Apple )?iPhone", c))
    return keep + ["Category:Taken with front camera"] + subcats("Category:Selfies", 3)


def subcats(root, depth):
    """root + its subcategories down to `depth` levels (selfie categories: labels still come from EXIF only)."""
    seen, frontier = [root], [root]
    for _ in range(depth):
        nxt = []
        for c in frontier:
            cont = {}
            while True:
                d = get(dict(action="query", list="categorymembers", cmtitle=c, cmtype="subcat", cmlimit=500, **cont))
                for m in d["query"]["categorymembers"]:
                    if m["title"] not in seen:
                        seen.append(m["title"]); nxt.append(m["title"])
                if "continue" not in d:
                    break
                cont = {k: v for k, v in d["continue"].items() if k != "continue"}
        frontier = nxt
        if len(seen) > 400:
            break
    return seen[:400]


def harvest(cat, out_dir, cap):
    fn = os.path.join(out_dir, "meta", re.sub(r"[^A-Za-z0-9_.-]+", "_", cat[9:]) + ".jsonl")
    if os.path.exists(fn + ".done"):
        return cat, -1
    n, cont = 0, {}
    with open(fn, "w") as f:
        while n < cap:
            d = get(dict(action="query", generator="categorymembers", gcmtitle=cat, gcmtype="file", gcmlimit=50,
                         prop="imageinfo", iiprop="url|size|metadata|user|mime", iiurlwidth=1024, **cont))
            for p in (d.get("query") or {}).get("pages", {}).values():
                ii = (p.get("imageinfo") or [None])[0]
                if not ii or ii.get("mime") != "image/jpeg":
                    continue
                md = {m["name"]: m["value"] for m in (ii.get("metadata") or []) if m["name"] in KEEP}
                f.write(json.dumps(dict(title=p["title"], pageid=p["pageid"], user=ii.get("user"), w=ii.get("width"),
                                        h=ii.get("height"), thumb=ii.get("thumburl"), cat=cat, exif=md)) + "\n")
                n += 1
            if "continue" not in d:
                break
            cont = {k: v for k, v in d["continue"].items() if k != "continue"}
    open(fn + ".done", "w").write(str(n))
    return cat, n


def main():
    out_dir = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    cats = categories()
    print(len(cats), "categories:", cats, flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(1) as ex:
        for cat, n in ex.map(lambda c: harvest(c, out_dir, cap), cats):
            print(f"{time.time() - t0:7.0f}s {cat}: {n}", flush=True)


if __name__ == "__main__":
    main()
