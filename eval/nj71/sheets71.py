"""RESULTS 71 eye check (pre-registered): photos all-judge (AJ) returns that NJ misses ("missed") and photos NJ returns that
AJ does not ("extra"), TRAIN + dev pooled, requests in seed-71 order, <= 4 photos per request (seed 71), until >= 10 tiles
over >= 3 requests. Tiles 900 px (bench originals ~500 px: upscaled). Caption: in the RESULTS-63 truth or not + the photo's
facts line (what NJ's millisecond scorer read).
  python eval/nj71/sheets71.py <missed|extra>   -> eval/nj71/sheets/<mode>{.txt,_<k>.jpg}"""
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

OUTD = ROOT / "eval/nj71/sheets"
FACTS = {"train": "eval/facts_scorer/data/facts_w_t2b_final_rs2_xfitm.parquet", "dev": "eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet"}


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
    mode = sys.argv[1]
    T = (ROOT / "eval/nj71/threshold.txt").read_text().split()[0]
    OUTD.mkdir(parents=True, exist_ok=True)
    path, _ = load()
    cand, FT = [], {}
    for sp in ("train", "dev"):
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{sp}.json").read_text())["queries"]}
        A, B = got(f"eval/scope69/base/{sp}/alljudge"), got(f"eval/nj71/replay/s1/{sp}/NJ_fp32_T{T}")
        X = pd.read_parquet(ROOT / FACTS[sp], columns=["item_id", "text"]); FT.update(zip(X.item_id.astype(str), X.text))
        for q in sorted(B):
            if q not in Q or q not in A:
                continue
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            pool = sorted(A[q] - B[q]) if mode == "missed" else sorted(B[q] - A[q])
            if pool:
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), pool, gt))
    rng = np.random.default_rng(71)
    order = rng.permutation(len(cand))
    what = "AJ returns, NJ misses" if mode == "missed" else "NJ returns, AJ does not"
    L = [f"{mode} (NJ t=.{T}): {len(cand)} requests have >= 1 photo that {what}; seed-71 order until >= 10 tiles over >= 3 requests"]
    Fn = font(20)
    n_tiles, j = 0, 0
    for k in order:
        if n_tiles >= 10 and j >= 3:
            break
        sp, q, Qq, fa, fb, pool, gt = cand[k]
        show = [pool[i] for i in sorted(rng.choice(len(pool), min(4, len(pool)), replace=False))]
        head = (f"#{j + 1} {sp} {q} ({Qq['cat']}): {' || '.join(Qq['turns'])} | F1 vs truth AJ {fa:.3f} / NJ {fb:.3f}; "
                f"{len(pool)} photos {what}")
        L.append(head)
        j += 1
        H = W + 260
        S = Image.new("RGB", (2 * W + 20, H * math.ceil(len(show) / 2) + 100), "white")
        d = ImageDraw.Draw(S)
        y0 = 4
        for line in textwrap.wrap(head, 170)[:4]:
            d.text((4, y0), line, fill="black", font=Fn); y0 += 23
        for t, i in enumerate(show):
            x, y = (t % 2) * (W + 20), 100 + (t // 2) * H
            im, n = tile(path[i]); S.paste(im, (x, y))
            cap = f"{i} | truth {'YES' if i in gt else 'no'} | native {n} px | facts: {FT.get(i, '(none)')}"
            L.append("   " + cap)
            yy = y + W + 4
            for line in textwrap.wrap(cap, 80)[:10]:
                d.text((x + 4, yy), line, fill="black", font=Fn); yy += 23
        S.save(OUTD / f"{mode}_{j - 1}.jpg", quality=92)
        n_tiles += len(show)
    L.insert(1, f"tiles shown {n_tiles} over {j} requests")
    (OUTD / f"{mode}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
