"""Eye-audit sheets for eval_everyday v2: per query (except night), 10 random photos of the exhaustive set pooled over
users (random.Random(1)), plus 'dog_user': 10 random of the 47642109 user's 'photos with a dog' results.
Sheets: 2 photos side by side, each tile max side 1000 px (judge resolution). Public data (DISBench / Flickr) only."""
import glob
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from findpics import store
from findpics.media import load_image

T = 1000
rows = [r for f in sorted(glob.glob("eval/everyday_v2/part*.json")) for r in json.load(open(f))]
idx = store.load("data/public/index_disbench")
path = dict(zip(idx.items.item_id.astype(str), idx.items.path))
out = Path("eval/everyday_v2/audit"); out.mkdir(parents=True, exist_ok=True)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
except OSError:
    font = ImageFont.load_default()

sets = {}
for q in sorted({r["query"] for r in rows}):
    if q == "photos taken at night":
        continue
    pool = [i for r in rows if r["query"] == q for i in r["exhaustive"]]
    sets[q] = (q, random.Random(1).sample(pool, min(10, len(pool))))
dog = [i for r in rows if r["user"].startswith("47642109") and r["query"] == "photos with a dog" for i in r["exhaustive"]]
sets["dog_user"] = ("photos with a dog", random.Random(1).sample(dog, min(10, len(dog))))

key = []
for name, (q, pick) in sets.items():
    for s in range(0, len(pick), 2):
        S = Image.new("RGB", (2 * T + 10, T), (30, 30, 30)); d = ImageDraw.Draw(S)
        sheet = f"{name.replace(' ', '_')}_{s // 2 + 1}.jpg"
        for j, iid in enumerate(pick[s:s + 2]):
            im = load_image(path[iid], max_side=T); x = j * (T + 10)
            if max(im.size) < T:  # DISBench originals are 500 px; upscale to the tile (no new detail, easier to read)
                f = T / max(im.size); im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
            S.paste(im, (x + (T - im.width) // 2, (T - im.height) // 2))
            d.rectangle([x, 0, x + 70, 50], fill=(0, 0, 0)); d.text((x + 8, 4), str(s + j + 1), fill=(255, 255, 0), font=font)
            key.append({"set": name, "sheet": sheet, "number": s + j + 1, "item_id": iid, "query": q})
        S.save(out / sheet, quality=90)
    print(name, len(pick))
json.dump(key, open(out / "key.json", "w"), indent=1)
