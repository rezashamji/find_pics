"""RESULTS 70 eye check (pre-registered): truth photos the capped line at <tag> LOSES that the uncapped control (same jobs)
returned. Requests (TRAIN + dev pooled) with >= 1 such photo, seed-70 order, <= 4 photos per request, until >= 10 tiles over
>= 3 requests. Tiles 900 px (bench originals ~500 px: upscaled). Caption: capped line, full line, judged per leaf.
  python eval/trunc70/sheets70.py <tag>   -> eval/trunc70/sheets/lost_<tag>{.txt,_<k>.jpg}
"""
import gzip
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
sys.path.insert(0, str(ROOT / "eval/scope69"))
from sheets import W, font, load, tile  # noqa: E402
from sheets69 import f1, got  # noqa: E402

OUTD = ROOT / "eval/trunc70/sheets"
SRC = {"train": "t2b_final_rs2_xfit", "dev": "p2b_devmix_rs2"}


def recs(d):
    out = {}
    for f in sorted((ROOT / d).glob("rec*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); out[r["qid"]] = r
    return out


def main():
    tag = sys.argv[1]
    OUTD.mkdir(parents=True, exist_ok=True)
    path, _ = load()
    cand = []
    for sp in ("train", "dev"):
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{sp}.json").read_text())["queries"]}
        A, B = got(f"eval/trunc70/replay/{sp}/DRY_fp32"), got(f"eval/trunc70/replay/{sp}/DRY_fp32_F{tag}")
        for q in sorted(B):
            if q not in Q or q not in A:
                continue
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            lost = sorted((A[q] - B[q]) & gt)
            if lost:
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), lost))
    rng = np.random.default_rng(70)
    order = rng.permutation(len(cand))
    L = [f"lost at {tag}: {len(cand)} requests lose >= 1 truth photo that the uncapped control returned; seed-70 order until "
         f">= 10 tiles over >= 3 requests"]
    F = font(20)
    n_tiles, j, RC, TX = 0, 0, {}, {}
    for k in order:
        if n_tiles >= 10 and j >= 3:
            break
        sp, q, Qq, fa, fb, pool = cand[k]
        if sp not in RC:
            lib = {}
            for f in (ROOT / "data/public/trunc70/record" / sp).glob("lib_*.json"):
                x = json.loads(f.read_text()); lib[x["user"]] = x["ids"]
            RC[sp] = (recs(f"data/public/trunc70/record/{sp}/DRY_fp32"),
                      recs(f"data/public/trunc70/record/{sp}/DRY_fp32_F{tag}"), lib)
            TX[sp] = tuple(pd.read_parquet(ROOT / p).assign(item_id=lambda d: d.item_id.astype(str)).set_index("item_id").text
                           for p in (f"eval/facts_scorer/data/facts_w_r70_{SRC[sp]}_{tag}m.parquet",
                                     "eval/facts_scorer/data/" + ("facts_w_t2b_final_rs2_xfitm.parquet" if sp == "train"
                                                                  else "facts_w_p2b_devmix_rs2m.parquet")))
        ra, rb = RC[sp][0][q], RC[sp][1][q]
        ids = RC[sp][2][Qq["user"]]
        ja = {(ids[x[0]], x[1]) for x in ra["judged"]}; jb = {(ids[x[0]], x[1]) for x in rb["judged"]}
        leaves = sorted({x[1] for x in ra["judged"]} | {x[1] for x in rb["judged"]})
        show = [pool[i] for i in sorted(rng.choice(len(pool), min(4, len(pool)), replace=False))]
        head = (f"#{j + 1} {sp} {q} ({Qq['cat']}): {' || '.join(Qq['turns'])} | F1 full {fa:.3f} -> {tag} {fb:.3f}; lost "
                f"truth {len(pool)} | judge calls full {ra['n_judge']} / {tag} {rb['n_judge']}")
        L.append(head)
        j += 1
        H = W + 420
        S = Image.new("RGB", (2 * W + 20, H * math.ceil(len(show) / 2) + 100), "white")
        d = ImageDraw.Draw(S)
        y0 = 4
        for line in textwrap.wrap(head, 170)[:4]:
            d.text((4, y0), line, fill="black", font=F); y0 += 23
        for t, i in enumerate(show):
            x, y = (t % 2) * (W + 20), 100 + (t // 2) * H
            im, n = tile(path[i]); S.paste(im, (x, y))
            jl = "; ".join(f"[{lq[:40]}] full {'J' if (i, lq) in ja else '-'} / {tag} {'J' if (i, lq) in jb else '-'}"
                           for lq in leaves)
            capped, full = TX[sp][0][i], TX[sp][1][i]
            cut = full[len(capped):] if full.startswith(capped) else "(reordered)"
            cap = f"{i} | {n} px | judged: {jl} | LINE AT {tag}: {capped} || CUT PART: {cut}"
            L.append("   " + cap)
            yy = y + W + 4
            for line in textwrap.wrap(cap, 85)[:17]:
                d.text((x + 4, yy), line, fill="black", font=F); yy += 23
        S.save(OUTD / f"lost_{tag}_{j - 1}.jpg", quality=92)
        n_tiles += len(show)
    L.insert(1, f"tiles shown {n_tiles} over {j} requests")
    (OUTD / f"lost_{tag}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
