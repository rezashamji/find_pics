"""RESULTS 65 eye check (pre-registered): for a config, 4 requests (seed 65) among those whose final ALL F1 drops > .10
vs 100_N_A (TRAIN + dev pooled); for each, up to 4 LOST truth photos (truth, returned at 100%, not returned at the config)
at 900 px (bench originals are 500 px: upscaled), captioned with whether the photo had a facts line at that config and
its facts line. Public bench photos only; sheets under eval/partial65/sheets/ (*.jpg git-ignored).
  python eval/partial65/sheets.py <config, e.g. 0_N_V or 25_N_V>
"""
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/dedup64"))
from sheets import W, font, load, tile  # noqa: E402

OUTD = ROOT / "eval/partial65/sheets"
FACTS = {"train": "eval/facts_scorer/data/facts_w_t2b_final_rs2_xfitm.parquet",
         "dev": "eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet"}


def got(d):
    out = {}
    for f in sorted((ROOT / d).glob("part*.json")):
        for r in json.loads(f.read_text()):
            fin = (r["turns"][-1].get("final") if r.get("turns") and not r.get("error") else None) or {}
            out[r["qid"]] = {str(x) for al in fin.get("albums", []) for x in al}
    return out


def f1(ret, gt):
    tp = len(ret & gt)
    return 0.0 if not tp else 2 * tp / (len(ret) + len(gt))


def main():
    cfg = sys.argv[1]
    cov, o, pol = cfg.split("_")
    OUTD.mkdir(parents=True, exist_ok=True)
    path, _ = load()
    ORD = pd.read_parquet(ROOT / "eval/partial65/data/orders.parquet")
    ORD["item_id"] = ORD.item_id.astype(str)
    hasf = set()
    for u, g in ORD.groupby("user"):
        m = math.ceil(int(cov) / 100 * len(g))
        hasf |= set(g.item_id[g[f"rank_{o}"].to_numpy() < m])
    cand = []
    for sp in ("train", "dev"):
        own = dict(zip(*[pd.read_parquet(ROOT / FACTS[sp]).query("split == @sp")[c].astype(str) for c in ("item_id", "text")]))
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{sp}.json").read_text())["queries"]}
        A = got(f"eval/partial65/replay/{sp}/100_N_A"); B = got(f"eval/partial65/replay/{sp}/{cfg}")
        for q in sorted(A):
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            if f1(A[q], gt) - f1(B[q], gt) > .10:
                lost = sorted((A[q] - B[q]) & gt)
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), lost, len(B[q] - A[q] - gt), own))
    rng = np.random.default_rng(65)
    pick = [cand[k] for k in sorted(rng.choice(len(cand), min(4, len(cand)), replace=False))]
    L = [f"{cfg}: {len(cand)} requests drop > .10 vs 100%; picked {len(pick)} (seed 65)"]
    F = font(20)
    for j, (sp, q, Qq, fa, fb, lost, added_wrong, own) in enumerate(pick):
        show = [lost[k] for k in sorted(rng.choice(len(lost), min(4, len(lost)), replace=False))] if lost else []
        head = (f"#{j + 1} {sp} {q} ({Qq['cat']}): {' || '.join(Qq['turns'])} | F1 100% {fa:.3f} -> {cfg} {fb:.3f}; "
                f"lost truth {len(lost)}, added non-truth {added_wrong}")
        L.append(head)
        if not show:
            continue
        H = W + 160
        S = Image.new("RGB", (2 * W + 20, H * math.ceil(len(show) / 2) + 80), "white")
        d = ImageDraw.Draw(S)
        y0 = 4
        for line in textwrap.wrap(head, 170)[:3]:
            d.text((4, y0), line, fill="black", font=F); y0 += 23
        for t, i in enumerate(show):
            x, y = (t % 2) * (W + 20), 80 + (t // 2) * H
            im, n = tile(path[i]); S.paste(im, (x, y))
            cap = f"{i} facts at {cfg}: {'YES' if i in hasf else 'NO'} | native {n} px | LINE: {own.get(i, '')[:330]}"
            L.append("   " + cap)
            yy = y + W + 4
            for line in textwrap.wrap(cap, 80)[:6]:
                d.text((x + 4, yy), line, fill="black", font=F); yy += 23
        S.save(OUTD / f"lost_{cfg}_{j}.jpg", quality=92)
    (OUTD / f"lost_{cfg}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
