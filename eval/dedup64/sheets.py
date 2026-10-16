"""RESULTS 64 eye check: (rep, member) pairs side by side, each photo 900 px on its long side, with cosine, time gap and the
rep's shared writer line printed under them. Public bench photos only; sheets are written under eval/dedup64/sheets/
(git-ignored: *.jpg).
  python eval/dedup64/sheets.py pairs --s=0.96 [--n=15 per split] [--seed=0]
  python eval/dedup64/sheets.py lost --s=0.96 --split=train   (truth photos returned per-photo, lost when shared, that
       are group members; one sheet per sampled loss: member | rep, the request words, both lines)
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "eval/dedup64/sheets"
W = 900


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def font(n):
    for f in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, n)
    return ImageFont.load_default()


def tile(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((W, W)) if max(im.size) > W else None
    if max(im.size) < W:     # small originals: upscale to 900 px long side so the sheet is at the judge's resolution
        r = W / max(im.size); im = im.resize((round(im.width * r), round(im.height * r)))
    c = Image.new("RGB", (W, W), "white"); c.paste(im, ((W - im.width) // 2, (W - im.height) // 2))
    return c, max(Image.open(path).size)


def sheet(pairs, out, captions):
    F = font(20)
    H = W + 200
    S = Image.new("RGB", (2 * W + 20, H * len(pairs)), "white")
    d = ImageDraw.Draw(S)
    for k, ((a, b), cap) in enumerate(zip(pairs, captions)):
        ta, na = tile(a); tb, nb = tile(b)
        S.paste(ta, (0, k * H)); S.paste(tb, (W + 20, k * H))
        y = k * H + W + 4
        for line in textwrap.wrap(cap + f" | native long side {na} / {nb} px", 170)[:8]:
            d.text((4, y), line, fill="black", font=F); y += 23
    S.save(out, quality=92)


def load():
    it = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet", columns=["item_id", "path", "taken"])
    it["item_id"] = it.item_id.astype(str)
    return dict(zip(it.item_id, it.path)), dict(zip(it.item_id, pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")))


def main():
    mode, s = sys.argv[1], arg("s")
    OUTD.mkdir(parents=True, exist_ok=True)
    path, taken = load()
    G = pd.read_parquet(ROOT / f"eval/dedup64/data/groups_s{s}.parquet")
    T = pd.read_parquet(ROOT / f"eval/dedup64/data/facts_shared_sinf.parquet")
    own = dict(zip(T.item_id.astype(str), T.text))
    if mode == "pairs":
        rng = np.random.default_rng(int(arg("seed", "0"))); n = int(arg("n", "15"))
        P = []
        for sp in ("train", "dev"):
            m = G[(G.split == sp) & (G.item_id != G.rep_id)]
            P += [r for r in m.iloc[rng.choice(len(m), n, replace=False)].itertuples()]
        L = []
        for j in range(0, len(P), 2):
            ch = P[j:j + 2]
            caps = [f"#{j + i + 1} {r.split} {r.user} rep {r.rep_id} | member {r.item_id} | cos {r.cos:.3f} | "
                    f"dt {(taken[r.item_id] - taken[r.rep_id]).total_seconds() / 60:.1f} min | REP LINE: {own[r.rep_id][:600]}"
                    for i, r in enumerate(ch)]
            sheet([(path[r.rep_id], path[r.item_id]) for r in ch], OUTD / f"pairs_s{s}_{j // 2:02d}.jpg", caps)
            L += caps + [f"   MEMBER OWN LINE #{j + i + 1}: {own[r.item_id][:600]}" for i, r in enumerate(ch)]
        (OUTD / f"pairs_s{s}.txt").write_text("\n".join(L) + "\n")
        print(f"{len(P)} pairs -> {OUTD}/pairs_s{s}_*.jpg")
    elif mode == "lost":
        sp = arg("split", "train")
        sys.path.insert(0, str(ROOT / "eval/complex_bench"))
        rep = dict(zip(G.item_id.astype(str), G.rep_id.astype(str)))

        def got(d):
            out = {}
            for f in sorted((ROOT / d).glob("part*.json")):
                for r in json.loads(f.read_text()):
                    fin = (r["turns"][-1].get("final") if r.get("turns") and not r.get("error") else None) or {}
                    out[r["qid"]] = (r, {str(x) for al in fin.get("albums", []) for x in al})
            return out
        A = got(f"eval/dedup64/replay/{sp}_perphoto/r59d"); B = got(f"eval/dedup64/replay/{sp}_s{s}/r59d")
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{sp}.json").read_text())["queries"]}
        rows = []
        for q in A:
            if q not in B:
                continue
            gt = {str(x) for al in Q[q]["answers"][-1] for x in al}
            for i in (A[q][1] - B[q][1]) & gt:
                if rep.get(i, i) != i:
                    rows.append((q, i, rep[i]))
        print(f"{len(rows)} lost truth photos that are group members ({sp}, s={s})")
        rng = np.random.default_rng(0)
        pick = [rows[k] for k in rng.choice(len(rows), min(len(rows), int(arg("n", "12"))), replace=False)] if rows else []
        L = []
        for j, (q, i, r) in enumerate(pick):
            cap = f"LOST #{j + 1} {q}: {' || '.join(Q[q]['turns'])} | member {i} (left) rep {r} (right) | MEMBER OWN: {own[i][:300]} | REP: {own[r][:300]}"
            sheet([(path[i], path[r])], OUTD / f"lost_{sp}_s{s}_{j:02d}.jpg", [cap]); L.append(cap)
        (OUTD / f"lost_{sp}_s{s}.txt").write_text(f"{len(rows)} lost member truth photos\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
