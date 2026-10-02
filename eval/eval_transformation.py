"""Closest public proxy for Reza's query: people with DOCUMENTED weight transformations (IMDB-WIKI face crops + photo year).
Eras from published reports (sources in JOURNAL 2026-10-02):
  Chris Pratt : ~300 lb, lost 60 lb in 6 months for Guardians of the Galaxy (filmed 2013) -> heavy <=2012, lean >=2014
  Jonah Hill  : lost ~40 lb in 2011 (21 Jump Street)                                    -> heavy <=2010, lean 2011-2012
  Seth Rogen  : lost 30 lb for The Green Hornet (shot 2009-10)                           -> heavy <=2008, lean 2010-2011
IMDB-WIKI photo years and name labels are noisy; results are read alongside the images, not alone.
Two of Reza's real risks:
  faces (main env): references from the LEAN era only -> are the HEAVY-era photos still matched? wrong matches among
        6,000 other people's faces?
  judge (vLLM env): within each person, does P(yes) rank heavy-era above lean-era photos (AUC)?
Usage: python eval/eval_transformation.py faces   |   python eval/eval_transformation.py judge
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ERAS = {"Chris Pratt": (2012, 2014, 2099), "Jonah Hill": (2010, 2011, 2012), "Seth Rogen": (2008, 2010, 2011)}
OUT = Path("data/public/transformation")


def era(name, y):
    h, l0, l1 = ERAS[name]
    return "heavy" if y <= h else ("lean" if l0 <= y <= l1 else None)


def load_meta():
    fs = sorted(Path("data/public/raw/imdb_wiki/face").glob("train-*.parquet"))
    parts = []
    for f in fs:
        t = pq.read_table(f, columns=["name", "photo_taken", "face_score"]).to_pandas()
        t["f"] = str(f); t["r"] = np.arange(len(t)); parts.append(t)
    m = pd.concat(parts, ignore_index=True)
    m = m[np.isfinite(m.face_score.astype(float))].copy()
    m["year"] = pd.to_datetime(m.photo_taken, errors="coerce").dt.year
    return m.dropna(subset=["year"])


def read_crops(rows):
    out = {}
    for f, g in rows.groupby("f"):
        col = pq.read_table(f, columns=["face_image"]).column("face_image").to_pylist()
        for k, r in g.iterrows():
            v = col[int(r.r)]
            out[k] = Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v)).convert("RGB")
    return out


def part_faces():
    from findpics.models import FaceEncoder
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_meta()
    tgt = m[m.name.isin(list(ERAS))].copy()
    tgt["era"] = [era(n, int(y)) for n, y in zip(tgt.name, tgt.year)]
    tgt = tgt[tgt.era.notna()]
    print(tgt.groupby(["name", "era"]).size().to_string(), flush=True)
    dist = m[~m.name.isin(list(ERAS))].sample(6000, random_state=0)
    ims = read_crops(pd.concat([tgt, dist]))
    fe = FaceEncoder()
    emb = {}
    for k, im in ims.items():
        c = Image.new("RGB", (448, 448), (127, 127, 127)); c.paste(im.resize((224, 224)), (112, 112))
        f = fe.faces(c)
        if f:
            emb[k] = max(f, key=lambda d: d["det_score"])["emb"].astype(np.float32)
    keys = list(emb); E = np.stack([emb[k] for k in keys]); ix = {k: i for i, k in enumerate(keys)}
    others = [ix[k] for k in dist.index if k in emb]
    res = {}
    for name in ERAS:
        t = tgt[(tgt.name == name) & tgt.index.isin(list(emb))]
        lean = [ix[k] for k in t[t.era == "lean"].index]; heavy = [ix[k] for k in t[t.era == "heavy"].index]
        if len(lean) < 3 or len(heavy) < 3:
            res[name] = dict(note="too few photos", lean=len(lean), heavy=len(heavy)); continue
        ref = np.random.default_rng(0).choice(lean, min(10, len(lean)), replace=False)
        S = E @ E[ref].T; S[S > 0.999] = -1; s = S.max(1)
        res[name] = dict(refs_from_lean_era=int(len(ref)), heavy_photos=len(heavy), lean_photos=len(lean))
        for th in (0.4, 0.3):
            res[name][f"heavy_era_found@{th}"] = int((s[heavy] >= th).sum())
            res[name][f"wrong_among_{len(others)}@{th}"] = int((s[others] >= th).sum())
        res[name]["heavy_era_median_sim"] = round(float(np.median(s[heavy])), 3)
    print(json.dumps(res, indent=1)); json.dump(res, open("eval/results_transformation_faces.json", "w"), indent=1)
    tgt[["name", "year", "era"]].to_csv(OUT / "rows.csv")
    for k in tgt.index:
        ims[k].save(OUT / f"{k}.jpg", quality=92)


def part_judge():
    from scipy.stats import rankdata
    from findpics.vlm import VLLMJudge
    rows = pd.read_csv(OUT / "rows.csv", index_col=0)
    ims = [Image.open(OUT / f"{k}.jpg").convert("RGB") for k in rows.index]
    J = VLLMJudge()
    res = {}
    for qn, q in [("heavier", "Does this person's face look heavier or fuller than average?"),
                  ("overweight", "Does this person look overweight?")]:
        rows[f"p_{qn}"] = J.p_yes(ims, q)
        for name, g in rows.groupby("name"):
            y = (g.era == "heavy").to_numpy(); p = g[f"p_{qn}"].to_numpy()
            if y.sum() and (~y).sum():
                r = rankdata(p); n1 = int(y.sum()); n0 = len(y) - n1
                res.setdefault(name, {})[qn] = dict(
                    auc_heavy_vs_lean=round(float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 3), heavy=n1, lean=n0,
                    mean_p_heavy=round(float(p[y].mean()), 3), mean_p_lean=round(float(p[~y].mean()), 3))
    print(json.dumps(res, indent=1)); json.dump(res, open("eval/results_transformation_judge.json", "w"), indent=1)
    rows.to_csv(OUT / "scored.csv")


if __name__ == "__main__":
    {"faces": part_faces, "judge": part_judge}[sys.argv[1]]()
