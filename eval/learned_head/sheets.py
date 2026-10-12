"""Contact sheets at NATIVE pixels (DISBench ~500 px long side = what the judge sees) of examples picked by
pick_failures(): photos where the learned head and the eye label / teacher disagree. Public data; *.jpg is git-ignored.
  python eval/learned_head/sheets.py   -> eval/learned_head/sheets/<name>.jpg + sheets/index.tsv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/learned_head")


def render(rows, name, cols=3):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    ims = []
    for r in rows:
        im = Image.open(r["path"]).convert("RGB")
        if max(im.size) > 1000:
            im.thumbnail((1000, 1000))
        ims.append(im)
    W = max(im.width for im in ims); H = max(im.height for im in ims) + 44
    n = len(ims); rr = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 8, rr * H + (rr - 1) * 8), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for k, (im, r) in enumerate(zip(ims, rows)):
        x, y = (k % cols) * (W + 8), (k // cols) * (H + 8)
        sheet.paste(im, (x, y + 44))
        dr.text((x + 4, y + 2), f"{k + 1}. {r['caption'][:70]}", fill=(0, 0, 0), font=font)
        dr.text((x + 4, y + 22), r["caption"][70:140], fill=(0, 0, 0), font=font)
    (OUT / "sheets").mkdir(exist_ok=True)
    sheet.save(OUT / "sheets" / f"{name}.jpg", quality=92)


def main():
    A = pd.read_parquet(OUT / "pairs.parquet")
    it = pd.read_parquet("data/public/index_disbench/items.parquet")
    path = dict(zip(it.item_id.astype(str), it.path))
    rng = np.random.default_rng(48)
    idx = []
    # (1) A4: eye-labelled real matches the held-out head ranks lowest (misses), per object search
    S = pd.read_parquet(OUT / "scores_E1.parquet")
    a4 = S[(S.src == "A4") & (S.eye == "match")]
    for q in ("photos with a car", "photos with a bicycle", "photos with a dog"):
        d = a4[a4["query"] == q].sort_values("mlp").head(6)
        idx.append((f"miss_{q.split()[-1]}", [dict(path=path[i], caption=f"{q}: eye=match head={m:.3f} judge P={p:.3f} look={l:.3f}")
                                              for i, m, p, l in zip(d.item, d.mlp, d.p, d.look)]))
    # (2) A4 scene searches: eye no_match photos the head scores highest
    for q in ("beach photos", "food photos", "sunset photos"):
        d = S[(S.src == "A4") & (S["query"] == q) & (S.eye == "no_match")].sort_values("mlp", ascending=False).head(6)
        idx.append((f"fp_{q.split()[0]}", [dict(path=path[i], caption=f"{q}: eye=NO head={m:.3f} judge P={p:.3f}")
                                           for i, m, p in zip(d.item, d.mlp, d.p)]))
    # (3) E2 held-out JD questions: counting / relations / text / person-look, teacher-yes vs teacher-no pairs the head
    # orders wrongly with the largest margin, inside one question's top-24 photos
    S2 = pd.read_parquet(OUT / "scores_E2.parquet")
    S2 = S2[S2.jd_part == "top"]
    import re
    pats = {"count": r"(?i)\b(two|three|four|five|more than|at least|exactly)\b", "relation": r"(?i)\b(next to|in front of|behind|on top of|holding|under|beside)\b",
            "text": r"(?i)\b(text|written|writing|sign|reads|says|receipt|menu)\b", "personlook": r"(?i)\b(smiling|wearing|hair|laughing|crying|tired|happy|sad)\b"}
    for nm, pat in pats.items():
        cand = []
        for q, d in S2[S2["query"].str.contains(pat)].groupby("query"):
            if 2 <= d.y.sum() <= len(d) - 2:
                yes, no = d[d.y], d[~d.y]
                worst_no = no.sort_values("mlp", ascending=False).iloc[0]; worst_yes = yes.sort_values("mlp").iloc[0]
                if worst_no.mlp > worst_yes.mlp:
                    cand.append((worst_no.mlp - worst_yes.mlp, q, worst_yes, worst_no))
        cand.sort(key=lambda x: -x[0])
        pick = [cand[k] for k in sorted(rng.choice(len(cand), size=min(4, len(cand)), replace=False))] if cand else []
        rows = []
        for _, q, ys, ns in pick:
            rows.append(dict(path=path[ys["item"]], caption=f"Q: {q} | teacher YES p9={ys.p:.2f} head={ys.mlp:.3f}"))
            rows.append(dict(path=path[ns["item"]], caption=f"Q: {q} | teacher NO p9={ns.p:.2f} head={ns.mlp:.3f}"))
        if rows:
            idx.append((f"jd_{nm}", rows))
    lines = []
    for name, rows in idx:
        render(rows, name, cols=2 if name.startswith("jd_") else 3)
        lines += [f"{name}\t{k + 1}\t{r['caption']}\t{r['path']}" for k, r in enumerate(rows)]
    (OUT / "sheets" / "index.tsv").write_text("\n".join(lines) + "\n")
    print("SHEETS_DONE", [n for n, _ in idx])


if __name__ == "__main__":
    main()
