"""RESULTS 49c: eye-check the dev c truth sets the build.py person_need fix changed. Native pixels (DISBench <= 500 px),
6 photos per sheet, numbered; no scores shown. Per request: every new truth photo (up to 36, else a uniform sample of
36) and a uniform sample of 6 photos the fix REMOVED from the truth.
Usage: python eval/complex_bench/c_diag/s5_truthfix_sheets.py <old dev.json> <new dev.json>
  -> eval/complex_bench/c_diag/truthfix/<qid>_{new,gone}_NN.jpg (+ index.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path("eval/complex_bench/c_diag/truthfix")


def sheets(paths, stem, font):
    names = []
    for b in range(0, len(paths), 6):
        ims = [Image.open(p).convert("RGB") for p in paths[b:b + 6]]
        W = max(i.width for i in ims); H = max(i.height for i in ims)
        sh = Image.new("RGB", (2 * W + 6, 3 * H + 12), (255, 255, 255)); dr = ImageDraw.Draw(sh)
        for k, im in enumerate(ims):
            x, y = (k % 2) * (W + 6), (k // 2) * (H + 6)
            sh.paste(im, (x, y)); dr.rectangle([x, y, x + 24, y + 26], fill=(255, 255, 0))
            dr.text((x + 6, y + 1), str(k + 1), fill=(0, 0, 0), font=font)
        n = f"{stem}_{b // 6 + 1:02d}.jpg"; sh.save(OUT / n, quality=95); names.append(n)
    return names


def main():
    old = {q["qid"]: q for q in json.load(open(sys.argv[1]))["queries"]}
    new = {q["qid"]: q for q in json.load(open(sys.argv[2]))["queries"]}
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(it.item_id.astype(str), it.path))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4949)
    index = {}
    for qid in sorted(set(old) | set(new)):
        if qid.split(":")[1][0] != "c":
            continue
        a = set().union(*map(set, old[qid]["answers"][-1])) if qid in old else set()
        b = set().union(*map(set, new[qid]["answers"][-1])) if qid in new else set()
        if a == b:
            continue
        nb = sorted(b); gone = sorted(a - b)
        show = list(rng.choice(nb, size=min(36, len(nb)), replace=False)) if nb else []
        gs = list(rng.choice(gone, size=min(6, len(gone)), replace=False)) if gone else []
        s = qid.replace("@", "_").replace(":", "_")
        index[qid] = dict(words=(new.get(qid) or old.get(qid))["turns"], old=len(a), new=len(b), shown_new=show,
                          shown_gone=gs, sheets_new=sheets([path[i] for i in show], f"{s}_new", font),
                          sheets_gone=sheets([path[i] for i in gs], f"{s}_gone", font))
        print(qid, index[qid]["words"], len(a), "->", len(b), index[qid]["sheets_new"], index[qid]["sheets_gone"])
    json.dump(index, open(OUT / "index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
