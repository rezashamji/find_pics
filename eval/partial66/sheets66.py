"""RESULTS 66 eye check (pre-registered, JOURNAL "RESULTS 66 PRE-REGISTRATION").
  python eval/partial66/sheets66.py lost <cov>_N_<pol>      requests (TRAIN + dev pooled) whose final F1 drops > .10 vs 100%,
      in seed-66 order, until >= 12 LOST truth tiles (truth, returned at 100%, not returned at the config; up to 4 per request)
  python eval/partial66/sheets66.py vs <cov>_N_<pol>        photos the config returns that <cov>_N_V does not (gains if truth,
      new false positives if not): >= 12 tiles, seed-66 order over requests, up to 4 per request
Tiles at 900 px (bench originals 500 px: upscaled), captioned with facts-at-config, the prefix line and the writer line.
Public bench photos only; sheets under eval/partial66/sheets/ (*.jpg git-ignored).
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
sys.path.insert(1, str(ROOT / "eval/partial66"))
from prefix import prefixes  # noqa: E402
from sheets import W, font, load, tile  # noqa: E402

OUTD = ROOT / "eval/partial66/sheets"
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
    mode, cfg = sys.argv[1], sys.argv[2]
    cov, o, pol = cfg.split("_")
    OUTD.mkdir(parents=True, exist_ok=True)
    path, _ = load()
    PRE = prefixes()
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
        B = got(f"eval/partial66/replay/{sp}/{cfg}")
        A = got(f"eval/partial65/replay/{sp}/100_N_A") if mode == "lost" else got(f"eval/partial66/replay/{sp}/{cov}_N_V")
        for q in sorted(B):
            if q not in Q:
                continue
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            if mode == "lost" and f1(A[q], gt) - f1(B[q], gt) > .10:
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), sorted((A[q] - B[q]) & gt), own, gt))
            if mode == "vs" and B[q] - A[q]:
                cand.append((sp, q, Q[q], f1(A[q], gt), f1(B[q], gt), sorted(B[q] - A[q]), own, gt))
    rng = np.random.default_rng(66)
    order = rng.permutation(len(cand))
    ref = "100%" if mode == "lost" else f"{cov}_N_V"
    L = [f"{mode} {cfg}: {len(cand)} candidate requests ({'F1 drop > .10 vs 100%' if mode == 'lost' else 'returns photos V does not'})"
         f"; drawn in seed-66 order until >= 12 tiles"]
    F = font(20)
    n_tiles, j = 0, 0
    for k in order:
        if n_tiles >= 12:
            break
        sp, q, Qq, fa, fb, pool, own, gt = cand[k]
        show = [pool[i] for i in sorted(rng.choice(len(pool), min(4, len(pool)), replace=False))] if pool else []
        head = (f"#{j + 1} {sp} {q} ({Qq['cat']}): {' || '.join(Qq['turns'])} | F1 {ref} {fa:.3f} -> {cfg} {fb:.3f}; "
                f"{'lost truth' if mode == 'lost' else 'extra photos'} {len(pool)}")
        L.append(head)
        j += 1
        if not show:
            continue
        H = W + 210
        S = Image.new("RGB", (2 * W + 20, H * math.ceil(len(show) / 2) + 80), "white")
        d = ImageDraw.Draw(S)
        y0 = 4
        for line in textwrap.wrap(head, 170)[:3]:
            d.text((4, y0), line, fill="black", font=F); y0 += 23
        for t, i in enumerate(show):
            x, y = (t % 2) * (W + 20), 80 + (t // 2) * H
            im, n = tile(path[i]); S.paste(im, (x, y))
            tag = "" if mode == "lost" else (" TRUTH" if i in gt else " NOT-truth")
            cap = (f"{i}{tag} facts at {cfg}: {'YES' if i in hasf else 'NO'} | native {n} px | PREFIX: {PRE.get(i, '')} | "
                   f"LINE: {own.get(i, '')[len(PRE.get(i, '')):][:260]}")
            L.append("   " + cap)
            yy = y + W + 4
            for line in textwrap.wrap(cap, 80)[:8]:
                d.text((x + 4, yy), line, fill="black", font=F); yy += 23
        S.save(OUTD / f"{mode}_{cfg}_{j - 1}.jpg", quality=92)
        n_tiles += len(show)
    L.insert(1, f"tiles shown {n_tiles} over {j} requests")
    (OUTD / f"{mode}_{cfg}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
