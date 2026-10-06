"""Eye-audit of search precision on the 16 fresh DISBench libraries (eval/everyday16_9b).

For each query (except "photos taken at night") draws 12 returned photos stratified by user with
random.Random(6): cycle through users that returned >=1 photo (shuffled), one random photo from
each in turn, until 12 or the pool runs out. Writes 2x2 sheets of 1000-px numbered tiles to
eval/everyday16_audit/<query>_<n>.jpg and the key to eval/everyday16_audit/key.json.
Images are public Flickr data: never commit the sheets.
"""
import glob
import json
import os
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from findpics.media import load_image

SRC = "eval/everyday16_9b"
OUT = "eval/everyday16_audit"
SKIP = {"photos taken at night"}
TILE = 1000
N = 12


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
    rows = [x for f in sorted(glob.glob(SRC + "/part*.json")) for x in json.load(open(f))]
    paths = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(paths.item_id.astype(str), paths.path))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    rng = random.Random(6)
    key = []
    for q in sorted({x["query"] for x in rows} - SKIP):
        per_user = {}
        for x in rows:
            if x["query"] == q and x["exhaustive"]:
                per_user[x["user"]] = sorted(set(map(str, x["exhaustive"])))
        users = sorted(per_user)
        rng.shuffle(users)
        chosen = []
        while len(chosen) < N and any(per_user[u] for u in users):
            for u in users:
                if len(chosen) >= N:
                    break
                if per_user[u]:
                    i = per_user[u].pop(rng.randrange(len(per_user[u])))
                    chosen.append((i, u))
        for n0 in range(0, len(chosen), 4):
            sheet = f"{q.replace(' ', '_')}_{n0 // 4 + 1}.jpg"
            grid = Image.new("RGB", (2 * TILE, 2 * TILE), (0, 0, 0))
            for j, (iid, u) in enumerate(chosen[n0:n0 + 4]):
                grid.paste(tile(path[iid], j + 1, font), ((j % 2) * TILE, (j // 2) * TILE))
                key.append({"sheet": sheet, "number": j + 1, "item_id": iid, "user": u, "query": q})
            grid.save(os.path.join(OUT, sheet), quality=90)
        print(q, "users_with_hits", len(users), "audited", len(chosen),
              "distinct_users", len({u for _, u in chosen}))
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
