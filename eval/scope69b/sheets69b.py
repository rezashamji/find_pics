"""RESULTS 69b eye check (pre-registered): truth photos DRY fp32 at coverage c LOSES that DRY fp32 at 100% (RESULTS 69)
returned. Requests (TRAIN + dev pooled) with >= 1 such photo, seed-69 order, <= 4 photos per request, until >= 8 tiles over
>= 3 requests. Tiles 900 px (bench originals ~500 px: upscaled). Caption: whether the photo had a facts line at c, and
whether it was judged per leaf at 100% / at c.
  python eval/scope69b/sheets69b.py <cov>   -> eval/scope69b/sheets/lost_<cov>{.txt,_<k>.jpg}
"""
import gzip
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/dedup64"))
sys.path.insert(0, str(ROOT / "eval/scope69"))
from sheets import W, font, load, tile  # noqa: E402
from sheets69 import f1, got  # noqa: E402

OUTD = ROOT / "eval/scope69b/sheets"


def recs(d):
    out = {}
    for f in sorted((ROOT / d).glob("rec*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); out[r["qid"]] = r
    return out


def main():
    cov = sys.argv[1]
    OUTD.mkdir(parents=True, exist_ok=True)
    path, _ = load()
    cand = []
    for sp in ("train", "dev"):
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{sp}.json").read_text())["queries"]}
        A, B = got(f"eval/scope69/replay/{sp}/DRY_fp32"), got(f"eval/scope69b/replay/{sp}/DRY_fp32_{cov}")
        for q in sorted(B):
            if q not in Q or q not in A:
                continue
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            lost = sorted((A[q] - B[q]) & gt)
            if lost:
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), lost))
    rng = np.random.default_rng(69)
    order = rng.permutation(len(cand))
    L = [f"lost at {cov}%: {len(cand)} requests lose >= 1 truth photo that DRY fp32 at 100% returned; seed-69 order until "
         f">= 8 tiles over >= 3 requests"]
    F = font(20)
    n_tiles, j, RC = 0, 0, {}
    for k in order:
        if n_tiles >= 8 and j >= 3:
            break
        sp, q, Qq, fa, fb, pool = cand[k]
        if sp not in RC:
            lib = {}
            for f in (ROOT / "data/public/scope69/record" / sp).glob("lib_*.json"):
                x = json.loads(f.read_text()); lib[x["user"]] = x["ids"]
            RC[sp] = (recs(f"data/public/scope69/record/{sp}/DRY_fp32"),
                      recs(f"data/public/scope69b/record/{sp}/DRY_fp32_{cov}"), lib)
        ra, rb = RC[sp][0][q], RC[sp][1][q]
        ids = RC[sp][2][Qq["user"]]
        ja = {(ids[x[0]], x[1]) for x in ra["judged"]}; jb = {(ids[x[0]], x[1]) for x in rb["judged"]}
        hasf = {ids[x] for x in rb["hasf"]}
        leaves = sorted({x[1] for x in ra["judged"]} | {x[1] for x in rb["judged"]})
        show = [pool[i] for i in sorted(rng.choice(len(pool), min(4, len(pool)), replace=False))]
        head = (f"#{j + 1} {sp} {q} ({Qq['cat']}): {' || '.join(Qq['turns'])} | F1 100% {fa:.3f} -> {cov}% {fb:.3f}; lost "
                f"truth {len(pool)} | judge calls 100% {ra['n_judge']} / {cov}% {rb['n_judge']}")
        L.append(head)
        j += 1
        H = W + 210
        S = Image.new("RGB", (2 * W + 20, H * math.ceil(len(show) / 2) + 100), "white")
        d = ImageDraw.Draw(S)
        y0 = 4
        for line in textwrap.wrap(head, 170)[:4]:
            d.text((4, y0), line, fill="black", font=F); y0 += 23
        for t, i in enumerate(show):
            x, y = (t % 2) * (W + 20), 100 + (t // 2) * H
            im, n = tile(path[i]); S.paste(im, (x, y))
            jl = "; ".join(f"[{lq[:50]}] 100% {'J' if (i, lq) in ja else '-'} / {cov}% {'J' if (i, lq) in jb else '-'}"
                           for lq in leaves)
            cap = f"{i} | native {n} px | line at {cov}%: {'YES' if i in hasf else 'no'} | judged per leaf: {jl}"
            L.append("   " + cap)
            yy = y + W + 4
            for line in textwrap.wrap(cap, 80)[:8]:
                d.text((x + 4, yy), line, fill="black", font=F); yy += 23
        S.save(OUTD / f"lost_{cov}_{j - 1}.jpg", quality=92)
        n_tiles += len(show)
    L.insert(1, f"tiles shown {n_tiles} over {j} requests")
    (OUTD / f"lost_{cov}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
