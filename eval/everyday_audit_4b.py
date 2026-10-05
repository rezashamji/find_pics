"""Blind eye-audit of 9B (eval/everyday_v2) vs 4B (eval/everyday_4b) exhaustive-set disagreements.
Per query (except night): pool only9b / only4b over users, sample <=6 each (Random(2)), shuffle, 2-up sheets at
1000 px tiles. Sheets carry no model info; key -> eval/everyday_4b/audit/key.json. Public DISBench data only."""
import glob
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw
from findpics import store
from findpics.media import load_image


def load(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/part*.json")):
        rows += json.load(open(f))
    return {(r["user"], r["query"]): set(map(str, r["exhaustive"] or [])) for r in rows}


a9, a4 = load("eval/everyday_v2"), load("eval/everyday_4b")
idx = store.load("data/public/index_disbench")
path = dict(zip(idx.items.item_id.astype(str), idx.items.path))
out = Path("eval/everyday_4b/audit"); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(2)
key = []
T = 1000
for q in sorted({k[1] for k in a9}):
    if q == "photos taken at night":
        continue
    o9 = sorted(i for k in a9 if k[1] == q and k in a4 for i in a9[k] - a4[k])
    o4 = sorted(i for k in a9 if k[1] == q and k in a4 for i in a4[k] - a9[k])
    pick = [(i, "only9b") for i in rng.sample(o9, min(6, len(o9)))] + \
           [(i, "only4b") for i in rng.sample(o4, min(6, len(o4)))]
    rng.shuffle(pick)
    qn = q.replace(" ", "_")
    for s in range(0, len(pick), 2):
        S = Image.new("RGB", (2 * T + 10, T), (30, 30, 30)); d = ImageDraw.Draw(S)
        for j, (iid, side) in enumerate(pick[s:s + 2]):
            im = load_image(path[iid], max_side=T); x = j * (T + 10)
            S.paste(im, (x + (T - im.width) // 2, (T - im.height) // 2))
            n = s + j + 1
            d.rectangle([x, 0, x + 70, 30], fill=(0, 0, 0)); d.text((x + 8, 8), str(n), fill=(255, 255, 0))
            key.append({"sheet": f"{qn}_{s // 2 + 1}.jpg", "n": n, "item_id": iid, "query": q, "side": side})
        S.save(out / f"{qn}_{s // 2 + 1}.jpg", quality=88)
    print(f"{q}: only9b={len(o9)} only4b={len(o4)} sampled={len(pick)}")
json.dump(key, open(out / "key.json", "w"), indent=1)
