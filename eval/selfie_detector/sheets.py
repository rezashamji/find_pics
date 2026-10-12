"""Contact sheets for eyeballing (public Commons data only): 6 photos per sheet, 3x2, ~620 px tiles, caption = id.

Usage: python eval/selfie_detector/sheets.py LABELS.jsonl IMG_DIR OUT_DIR {front|back|ids:<file>} N [seed]
"""
from __future__ import annotations

import json
import os
import random
import sys

from PIL import Image, ImageDraw


def main():
    labels, img_dir, out_dir, which, n = sys.argv[1:6]
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rows = [json.loads(l) for l in open(labels)]
    rows = [r for r in rows if os.path.exists(f"{img_dir}/{r['pageid']}.jpg")]
    if which.startswith("ids:"):
        ids = [l.split()[0] for l in open(which[4:]) if l.strip()]
        by = {str(r["pageid"]): r for r in rows}
        pick = [by[i] for i in ids if i in by][: int(n)]
        tag = os.path.basename(which[4:]).split(".")[0]
    else:
        lab = 1 if which == "front" else 0
        pool = [r for r in rows if r["label"] == lab]
        random.Random(seed).shuffle(pool)
        pick = pool[: int(n)]
        tag = which
    os.makedirs(out_dir, exist_ok=True)
    T = 620
    for s in range(0, len(pick), 6):
        sheet = Image.new("RGB", (3 * T, 2 * (T + 24)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(pick[s:s + 6]):
            im = Image.open(f"{img_dir}/{r['pageid']}.jpg").convert("RGB")
            im.thumbnail((T, T))
            x, y = (k % 3) * T, (k // 3) * (T + 24)
            sheet.paste(im, (x + (T - im.size[0]) // 2, y + 24))
            d.text((x + 4, y + 4), f"{s + k}: {r['pageid']} {r['model']} {'FRONT' if r['label'] else 'back'}", fill="black")
        sheet.save(f"{out_dir}/{tag}_{s // 6:02d}.jpg", quality=90)
    with open(f"{out_dir}/{tag}_key.txt", "w") as f:
        for k, r in enumerate(pick):
            f.write(f"{k} {r['pageid']} {r['label']} {r['model']} {r['user']} {r['title']}\n")
    print(len(pick), "photos ->", out_dir)


if __name__ == "__main__":
    main()
