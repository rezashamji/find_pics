"""RESULTS 69 eye check (pre-registered): truth photos the phone-feasible variant LOSES that the measured harness (AJ
control) found. Requests (TRAIN + dev pooled) with >= 1 such photo, in seed-69 order, up to 4 photos per request, until
>= 12 tiles over >= 3 requests. Tiles 900 px (bench originals ~500 px: upscaled). Caption: whether the photo was judged
for each leaf in AJ / the variant (from the recordings) and its leaf scope sizes.
  python eval/scope69/sheets69.py <DRY_bf16|ON_bf16|...>   -> eval/scope69/sheets/lost_<cfg>{.txt,_<k>.jpg}
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
from sheets import W, font, load, tile  # noqa: E402

OUTD = ROOT / "eval/scope69/sheets"


def got(d):
    out = {}
    for f in sorted((ROOT / d).glob("part*.json")):
        for r in json.loads(f.read_text()):
            fin = (r["turns"][-1].get("final") if r.get("turns") and not r.get("error") else None) or {}
            out[r["qid"]] = {str(x) for al in fin.get("albums", []) for x in al}
    return out


def recs(sp, cfg):
    out = {}
    for f in sorted((ROOT / "data/public/scope69/record" / sp / cfg).glob("rec*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); out[r["qid"]] = r
    return out


def f1(ret, gt):
    tp = len(ret & gt)
    return 0.0 if not tp else 2 * tp / (len(ret) + len(gt))


def main():
    cfg = sys.argv[1]
    OUTD.mkdir(parents=True, exist_ok=True)
    path, _ = load()
    cand = []
    for sp in ("train", "dev"):
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{sp}.json").read_text())["queries"]}
        A, B = got(f"eval/scope69/replay/{sp}/AJ_bf16"), got(f"eval/scope69/replay/{sp}/{cfg}")
        for q in sorted(B):
            if q not in Q or q not in A:
                continue
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            lost = sorted((A[q] - B[q]) & gt)
            if lost:
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), lost))
    rng = np.random.default_rng(69)
    order = rng.permutation(len(cand))
    L = [f"lost {cfg}: {len(cand)} requests lose >= 1 truth photo the AJ control returned; drawn in seed-69 order until "
         f">= 12 tiles over >= 3 requests"]
    F = font(20)
    n_tiles, j = 0, 0
    RC = {}
    for k in order:
        if n_tiles >= 12 and j >= 3:
            break
        sp, q, Qq, fa, fb, pool = cand[k]
        if (sp, "AJ") not in RC:
            RC[(sp, "AJ")], RC[(sp, cfg)] = recs(sp, "AJ_bf16"), recs(sp, cfg)
            lib = {}
            for f in (ROOT / "data/public/scope69/record" / sp).glob("lib_*.json"):
                x = json.loads(f.read_text()); lib[x["user"]] = x["ids"]
            RC[(sp, "lib")] = lib
        ra, rb = RC[(sp, "AJ")][q], RC[(sp, cfg)][q]
        ids = RC[(sp, "lib")][Qq["user"]]
        ja = {(ids[x[0]], x[1]) for x in ra["judged"]}; jb = {(ids[x[0]], x[1]) for x in rb["judged"]}
        leaves = sorted({x[1] for x in ra["judged"]} | {x[1] for x in rb["judged"]})
        show = [pool[i] for i in sorted(rng.choice(len(pool), min(4, len(pool)), replace=False))]
        sa = {k_.rsplit("|", 1)[0]: len(v) for k_, v in ra["scope"].items()}
        sb = {k_.rsplit("|", 1)[0]: len(v) for k_, v in rb["scope"].items()}
        head = (f"#{j + 1} {sp} {q} ({Qq['cat']}): {' || '.join(Qq['turns'])} | F1 AJ {fa:.3f} -> {cfg} {fb:.3f}; lost truth "
                f"{len(pool)} | judge calls AJ {ra['n_judge']} / {cfg} {rb['n_judge']} | leaf scopes AJ {sa} / {cfg} {sb}")
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
            jl = "; ".join(f"[{lq[:50]}] AJ {'J' if (i, lq) in ja else '-'} / {cfg.split('_')[0]} {'J' if (i, lq) in jb else '-'}"
                           for lq in leaves)
            cap = f"{i} | native {n} px | judged per leaf: {jl}"
            L.append("   " + cap)
            yy = y + W + 4
            for line in textwrap.wrap(cap, 80)[:8]:
                d.text((x + 4, yy), line, fill="black", font=F); yy += 23
        S.save(OUTD / f"lost_{cfg}_{j - 1}.jpg", quality=92)
        n_tiles += len(show)
    L.insert(1, f"tiles shown {n_tiles} over {j} requests")
    (OUTD / f"lost_{cfg}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
