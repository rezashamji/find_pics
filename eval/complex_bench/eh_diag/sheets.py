# RESULTS 51: native-pixel 2x2 sheets of a dev request's photos (public DISBench), numbered; ids/dates printed.
# Usage: python sheets.py <qid> <turn> <kind: fn|fp|tp|ids:ID,ID> <n> [seed]
import json, glob, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
qid, ti, kind, n = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 51
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {r["qid"]: r for f in glob.glob("eval/complex_bench/runs/cmp/A4oldq4_search/part*.json") for r in json.load(open(f))}
q, r = Q[qid], R[qid]
u = q["user"]
gt = set().union(*map(set, q["answers"][ti]))
got = set().union(*map(set, r["turns"][ti]["final"]["albums"])) if r["turns"][ti]["final"] else set()
if kind.startswith("ids:"):
    ids = kind[4:].split(",")
else:
    pool = sorted({"fn": gt - got, "fp": got - gt, "tp": got & gt}[kind])
    rng = np.random.default_rng(seed)
    ids = list(rng.choice(pool, size=min(n, len(pool)), replace=False)) if pool else []
import pandas as pd
it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path", "taken", "place"])
it = it[it.path.str.contains(f"/{u}/", regex=False)]
meta = {str(a): dict(taken=b, place=c) for a, b, c in zip(it.item_id, it.taken, it.place)}
out = Path(".cache/cb51/sheets"); out.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22) if Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf").exists() else ImageFont.load_default()
for s in range(0, len(ids), 4):
    chunk = ids[s:s + 4]
    ims = [Image.open(f"data/public/raw/disbench/images/images/{u}/{i}.jpg").convert("RGB") for i in chunk]
    W = max(im.width for im in ims); H = max(im.height for im in ims)
    sheet = Image.new("RGB", (2 * W + 10, 2 * H + 10), "white")
    d = ImageDraw.Draw(sheet)
    for k, im in enumerate(ims):
        x, y = (k % 2) * (W + 10), (k // 2) * (H + 10)
        sheet.paste(im, (x, y)); d.rectangle([x, y, x + 34, y + 30], fill="yellow"); d.text((x + 6, y + 2), str(s + k + 1), fill="black", font=font)
    f = out / f"{qid.replace('@', '_').replace(':', '_')}_t{ti}_{kind[:3]}_{s // 4}.jpg"
    sheet.save(f, quality=92)
    print(f)
for k, i in enumerate(ids):
    m = meta.get(str(i), {})
    print(k + 1, i, m.get("taken", ""), str(m.get("place", ""))[:60])
