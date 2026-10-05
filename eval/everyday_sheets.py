"""Audit sheets for eval_everyday: per query, 8 random photos of the exhaustive set (pooled over users), 2x2 at 800 px
tiles (judge-resolution look; small thumbnails misled before). Public data (DISBench / Flickr) only."""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw
from findpics import store
from findpics.media import load_image

rows = json.load(open("eval/everyday/all.json"))
idx = store.load("data/public/index_disbench")
path = dict(zip(idx.items.item_id.astype(str), idx.items.path))
out = Path("eval/everyday/audit"); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
for q in sorted({r["query"] for r in rows}):
    if q in ("selfies", "photos taken at night"):
        continue
    pool = [i for r in rows if r["query"] == q for i in r["exhaustive"]]
    pick = rng.sample(pool, min(8, len(pool)))
    for s in range(0, len(pick), 4):
        S = Image.new("RGB", (1620, 1620), (30, 30, 30)); d = ImageDraw.Draw(S)
        for j, iid in enumerate(pick[s:s + 4]):
            im = load_image(path[iid], max_side=800); x, y = (j % 2) * 810, (j // 2) * 810
            S.paste(im, (x + (800 - im.width) // 2, y + (800 - im.height) // 2))
            d.rectangle([x, y, x + 60, y + 26], fill=(0, 0, 0)); d.text((x + 5, y + 5), str(s + j + 1), fill=(255, 255, 0))
        S.save(out / f"{q.replace(' ', '_')}_{s // 4 + 1}.jpg", quality=85)
    print(q, len(pool), "sampled", len(pick))
