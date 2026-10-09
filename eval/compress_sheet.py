"""Contact sheets of the eye-labeled public (DISBench) photos where a compression candidate and the q4 baseline disagree
(eval/compress/flips_<tag>.json from compress_report.py). Each photo at NATIVE pixels (only shrunk if > 900 px), two per
row, caption = query, eye label, P(base) -> P(candidate). Usage: python eval/compress_sheet.py q3vl4b_vis4 [cut ...]
"""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1]; cuts = sys.argv[2:] or ["0.7", "0.95"]
flips = json.load(open(ROOT / f"eval/compress/flips_{tag}.json"))
items = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet")
path = dict(zip(items.item_id.astype(str), items.path))
rows = {}
for c in cuts:
    for f in flips[c]:
        rows.setdefault((f["query"], f["item_id"]), dict(f, cuts=[]))["cuts"].append(c)
rows = list(rows.values())
out = ROOT / "eval/compress/sheets"; out.mkdir(parents=True, exist_ok=True)
PER, CAP = 6, 900
for s in range(0, len(rows), PER):
    chunk = rows[s:s + PER]
    ims = []
    for r in chunk:
        im = Image.open(path[r["item_id"]]).convert("RGB")
        if max(im.size) > CAP:
            im.thumbnail((CAP, CAP))
        ims.append(im)
    colw = max(i.width for i in ims)
    pairs = [ims[k:k + 2] for k in range(0, len(ims), 2)]
    H = sum(max(i.height for i in p) + 40 for p in pairs)
    canvas = Image.new("RGB", (2 * colw + 10, H), "white"); d = ImageDraw.Draw(canvas)
    y = 0; n = s
    for p in pairs:
        for k, im in enumerate(p):
            r = chunk[n - s]
            canvas.paste(im, (k * (colw + 10), y + 38))
            d.text((k * (colw + 10) + 4, y + 4), f"#{n} {r['query']} | eye {r['eye']} | cut {','.join(r['cuts'])}",
                   fill="black")
            d.text((k * (colw + 10) + 4, y + 20), f"P base {r['p_base']:.3f} -> cand {r['p_cand']:.3f} "
                   f"({im.width}x{im.height})", fill="black")
            n += 1
        y += max(i.height for i in p) + 40
    canvas.save(out / f"{tag}_{s // PER:02d}.png")
print(len(rows), "photos ->", out)
