"""M46 step 1a (more fronts): Commons full-text search by FRONT-CAMERA PIXEL SIZE and by the word "selfie".

Why: in the "Taken with iPhone ..." categories only ~0.2% of files are front-camera shots (MEASURED 10-11 on the
first 50k). The 7-MP front camera of iPhone 7 / 8 / X / XS / XR / SE 2-3 writes 3088x2316 (or 3088x2320), a size no
back camera of those phones uses, and CirrusSearch can filter on filew/fileh. Every hit is still labelled ONLY from
its EXIF (label_commons.py), so a non-iPhone 3088x2316 photo is simply dropped.

Usage: python eval/selfie_detector/harvest_search.py OUT_DIR   (appends OUT_DIR/meta/search_<k>.jsonl)
"""
from __future__ import annotations

import json
import os
import sys

from harvest_commons import KEEP, get

QUERIES = ["filew:3088 fileh:2316", "filew:2316 fileh:3088", "filew:3088 fileh:2320", "filew:2320 fileh:3088",
           "filew:2576 fileh:1932 iPhone", "filew:1932 fileh:2576 iPhone",
           "filew:1280 fileh:960 iPhone", "filew:960 fileh:1280 iPhone",
           "selfie filetype:bitmap", "selfie iPhone filetype:bitmap", "self-portrait iPhone filetype:bitmap"]


def titles(q, cap=10000):
    out, off = [], 0
    while off < cap:
        d = get(dict(action="query", list="search", srnamespace=6, srlimit=500, sroffset=off, srsearch=q,
                     srinfo="totalhits", srprop=""))
        out += [c["title"] for c in d["query"]["search"]]
        if "continue" not in d:
            break
        off = d["continue"]["sroffset"]
    return out


def main():
    out_dir = sys.argv[1]
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    for k, q in enumerate(QUERIES):
        fn = os.path.join(out_dir, "meta", f"search_{k}.jsonl")
        if os.path.exists(fn + ".done"):
            continue
        ts = titles(q)
        n = 0
        with open(fn, "w") as f:
            for s in range(0, len(ts), 20):
                d = get(dict(action="query", titles="|".join(ts[s:s + 20]), prop="imageinfo",
                             iiprop="url|size|metadata|user|mime", iiurlwidth=1024))
                for p in (d.get("query") or {}).get("pages", {}).values():
                    ii = (p.get("imageinfo") or [None])[0]
                    if not ii or ii.get("mime") != "image/jpeg":
                        continue
                    md = {m["name"]: m["value"] for m in (ii.get("metadata") or []) if m["name"] in KEEP}
                    f.write(json.dumps(dict(title=p["title"], pageid=p["pageid"], user=ii.get("user"),
                                            w=ii.get("width"), h=ii.get("height"), thumb=ii.get("thumburl"),
                                            cat="search:" + q, exif=md)) + "\n")
                    n += 1
        open(fn + ".done", "w").write(str(n))
        print(q, len(ts), "hits", n, "jpeg", flush=True)


if __name__ == "__main__":
    main()
