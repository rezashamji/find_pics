"""Which face model can the app SHIP? InsightFace buffalo_l (current) is licensed for non-commercial research only;
AuraFace-v1 (fal, Apache-2.0, trained on commercially usable data) is a drop-in InsightFace-format pack
(scrfd detector + glintr100 recognizer). Same protocol as eval_unseen_faces.py (DigiFace-1M: 300 rendered identities
no model has seen, 8 references each, the rest targets, every other identity a distractor).
The two models' similarity scales differ, so they are compared at EQUAL wrong-match rates (the rate buffalo_l has at
its product threshold 0.40, and 10x lower), not at the same number.
Usage (main env, GPU): python eval/eval_face_models.py
"""
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Z = "data/public/raw/digiface/subjects_0-1999_72_imgs.zip"
N_ID, N_REF = 300, 8


def ensure_auraface():
    d = Path(os.environ["INSIGHTFACE_HOME"]) / "models" / "auraface"
    if not (d / "glintr100.onnx").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("fal/AuraFace-v1", local_dir=str(d), allow_patterns=["glintr100.onnx", "scrfd_10g_bnkps.onnx", "LICENSE.md"])
    return d


def embed(name):
    from findpics.models import FaceEncoder
    fe = FaceEncoder(name=name)
    z = zipfile.ZipFile(Z)
    emb, who, miss, t0 = [], [], 0, time.time()
    for i in range(N_ID):
        for k in range(72):
            im = Image.open(io.BytesIO(z.read(f"{i}/{k}.png"))).convert("RGB")
            canvas = Image.new("RGB", (448, 448), (127, 127, 127)); canvas.paste(im.resize((224, 224)), (112, 112))
            f = fe.faces(canvas)
            if not f:
                miss += 1; continue
            emb.append(max(f, key=lambda d: d["det_score"])["emb"].astype(np.float32)); who.append(i)
    return np.stack(emb), np.array(who), miss, (time.time() - t0) / (N_ID * 72)


def scores(E, who, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for i in range(N_ID):
        mine = np.where(who == i)[0]
        if len(mine) < N_REF + 5:
            continue
        ref = rng.choice(mine, N_REF, replace=False); tgt = np.setdiff1d(mine, ref); others = np.where(who != i)[0]
        s = (E @ E[ref].T).max(1)
        pos.append(s[tgt]); neg.append(s[others])
    return np.concatenate(pos), np.concatenate(neg)


def main():
    ensure_auraface()
    res = {}
    for name in ["buffalo_l", "auraface"]:
        E, who, miss, sec = embed(name)
        pos, neg = scores(E, who)
        res[name] = dict(detected=int(len(E)), not_detected=miss, sec_per_image=round(sec, 4), pos=pos, neg=neg)
        print(name, "detected", len(E), "missed", miss, f"{sec*1000:.1f} ms/image", flush=True)
    b = res["buffalo_l"]
    fmr0 = float((b["neg"] >= 0.40).mean())                 # buffalo's wrong-match rate at the product threshold
    out = {}
    for name, r in res.items():
        row = dict(detected=r["detected"], not_detected=r["not_detected"], ms_per_image=round(r["sec_per_image"] * 1000, 1))
        for lbl, fmr in [("at_buffalo_0.40_rate", fmr0), ("at_10x_lower_rate", fmr0 / 10)]:
            t = float(np.quantile(r["neg"], 1 - fmr))          # threshold giving that wrong-match rate
            row[lbl] = dict(threshold=round(t, 3), recall=round(float((r["pos"] >= t).mean()), 4),
                            wrong_match_rate=round(float((r["neg"] >= t).mean()), 6))
        row["recall_at_0.40"] = round(float((r["pos"] >= 0.40).mean()), 4)
        row["wrong_rate_at_0.40"] = round(float((r["neg"] >= 0.40).mean()), 6)
        out[name] = row
    print(json.dumps(out, indent=1))
    json.dump(out, open("eval/results_face_models.json", "w"), indent=1)


if __name__ == "__main__":
    main()
