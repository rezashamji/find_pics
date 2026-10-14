"""Contact sheets of eye-set rows (the judge's own input image at native pixels: DISBench <= 500 px, person rows with
the person crop + red box), captioned with the eye label and each judge's P. For reading disagreements, not labelling.
Usage: python eval/judge_teach/sheet.py <out prefix> <judgeA> <judgeB> <filter: a_yes_b_no|a_no_b_yes> [--groups=D1,D2,D3]
       [--cutb=0.5] [--n=24]   -> .cache/tmp/r58_sheets/<prefix>_NN.jpg (git-ignored; public photos)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/judge_teach"))


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def main():
    from gate import GROUPS
    from findpics import store
    from findpics.engine import _frame_for, person_crop_boxed
    pre, a, b, mode = sys.argv[1:5]
    E = pd.read_parquet(ROOT / "eval/judge_teach/eyeset.parquet")
    pa = pd.read_parquet(ROOT / f"eval/judge_teach/eye_p/{a}.parquet").p.to_numpy()
    pb = pd.read_parquet(ROOT / f"eval/judge_teach/eye_p/{b}.parquet").p.to_numpy()
    g = np.zeros(len(E), bool)
    for k in arg("groups", "D1,D2,D3").split(","):
        g |= next(f for n, f in GROUPS.items() if n.startswith(k))(E).to_numpy()
    g &= (E.lib == arg("lib", "heldout")).to_numpy()
    ya = pa >= E.cut.to_numpy()
    yb = pb >= float(arg("cutb", "0.5"))
    sel = g & (ya & ~yb if mode == "a_yes_b_no" else ~ya & yb)
    R = E[sel].assign(pa=pa[sel], pb=pb[sel]).sample(frac=1, random_state=0).head(int(arg("n", "24")))
    idx = store.load(str(ROOT / "data/public/index_disbench"))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default(size=15)
    out = ROOT / ".cache/tmp/r58_sheets"; out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in R.itertuples():
        im = _frame_for(idx, int(r.item_row), int(r.face_row))
        if r.set == "person":
            im = person_crop_boxed(idx, im, int(r.face_row))
        im = im.copy(); im.thumbnail((896, 896))
        cap = [f"{r.search[:40]} | EYE {r.label}", f"{a} {r.pa:.4f} | {b} {r.pb:.4f}", r.question[:70]]
        tiles.append((im, cap))
    print(len(sel.nonzero()[0]), "rows match;", len(tiles), "drawn")
    for s in range(0, len(tiles), 6):
        T = tiles[s:s + 6]
        W = max(t.width for t, _ in T); H = max(t.height for t, _ in T) + 60
        sheet = Image.new("RGB", (3 * W, 2 * H), "white"); d = ImageDraw.Draw(sheet)
        for j, (im, cap) in enumerate(T):
            x, y = (j % 3) * W, (j // 3) * H
            sheet.paste(im, (x, y + 60))
            for li, c in enumerate(cap):
                d.text((x + 4, y + 2 + 19 * li), c, fill="black", font=font)
        sheet.save(out / f"{pre}_{s // 6:02d}.jpg", quality=90)


if __name__ == "__main__":
    main()
