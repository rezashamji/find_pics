"""Blind eye-audit of disagreements between the full-precision 9B (REF) and the phone's
4-bit 9B (Q9) / 4-bit 4B (Q4) on the everyday queries (DISBench, public Flickr data).

Writes 2x2 sheets (1000 px tiles, number only) to eval/everyday_q_audit/<query>_<n>.jpg and the
key (sheet, number -> item_id, query, sets) to eval/everyday_q_audit/key.json.
Images are public Flickr data: never commit the sheets.
"""
import glob
import json
import os
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from findpics.media import load_image

import os
OUT = os.environ.get("FP_QA_OUT", "eval/everyday_q_audit")
SKIP = {"photos taken at night", "selfies"}
TILE = 1000
PER_SET = 4


def load(d):
    r = {}
    for f in sorted(glob.glob(d + "/part*.json")):
        for x in json.load(open(f)):
            r[(x["user"], x["query"])] = set(x["exhaustive"] or [])
    return r


def tile(path, num, font):
    im = load_image(path, max_side=1000).convert("RGB")
    s = TILE / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (TILE, TILE), (40, 40, 40))
    canvas.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 110, 90], fill=(255, 255, 0))
    d.text((20, 5), str(num), fill=(0, 0, 0), font=font)
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    # FP_QA_RUNS="ref,a,b": compare two candidate runs against a reference (default: the 8-library phone-weight audit)
    ref, a, b = os.environ.get("FP_QA_RUNS", "eval/everyday_v2,eval/everyday_q9b,eval/everyday_q4b").split(",")
    R, Q9, Q4 = load(ref), load(a), load(b)
    paths = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(paths.item_id.astype(str), paths.path))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    rng = random.Random(5)
    key = []
    for q in sorted({k[1] for k in R} - SKIP):
        pools = {s: set() for s in ["ref_not_q9", "q9_not_ref", "ref_not_q4", "q4_not_ref"]}
        for (u, qq), r in R.items():
            if qq != q:
                continue
            a, b = Q9[(u, qq)], Q4[(u, qq)]
            pools["ref_not_q9"] |= r - a
            pools["q9_not_ref"] |= a - r
            pools["ref_not_q4"] |= r - b
            pools["q4_not_ref"] |= b - r
        chosen = []
        for s, pool in pools.items():
            pick = rng.sample(sorted(pool), min(PER_SET, len(pool)))
            for i in pick:
                if i not in chosen:
                    chosen.append(i)
        rng.shuffle(chosen)  # blind: no ordering by set
        for n0 in range(0, len(chosen), 4):
            sheet = f"{q.replace(' ', '_')}_{n0 // 4 + 1}.jpg"
            grid = Image.new("RGB", (2 * TILE, 2 * TILE), (0, 0, 0))
            for j, iid in enumerate(chosen[n0:n0 + 4]):
                grid.paste(tile(path[iid], j + 1, font), ((j % 2) * TILE, (j // 2) * TILE))
                key.append({"sheet": sheet, "number": j + 1, "item_id": iid, "query": q,
                            "sets": [s for s, p in pools.items() if iid in p]})
            grid.save(os.path.join(OUT, sheet), quality=90)
        print(q, {s: len(p) for s, p in pools.items()}, "audited", len(chosen))
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
