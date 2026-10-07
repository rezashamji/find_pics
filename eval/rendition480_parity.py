"""Since 8ded039 the phone indexes every photo from PhotoKit's LOCAL ~480 px rendition (448 px ask, resizeMode .fast)
instead of the full photo. How far does that move the vectors from the server's (which embeds the full photo, capped at
1600 px by findpics.media.load_image)? Public data only.

Phone rendition, simulated: long side -> 480 with Lanczos, JPEG round trip at quality 80 (PhotoKit derivatives are
JPEG/HEIC), then the server preprocessing (Pillow bilinear squash to 224 = the phone's PILResize since 10-07) and fp16
weights (the phone's image tower since 10-07). Reference: the server vector of the full photo (fp32).
Sets: Open Images (1024 px) and Pexels video frames (1600x900, the highest-resolution real scenes we have locally).
(1) image cosine vs server: mean / p5 / min, 200 photos per set;
(2) top-k agreement (overlap@600, @100, @20) for the 20 everyday queries on 2000-photo pools, all via the 480 path;
(3) faces: InsightFace SCRFD detection (server's detector, det 0.5; the phone uses Apple Vision, not simulated) and
    AuraFace+flip embeddings (findpics.face_profiles.SHIPPED) at full size vs at 480: faces found, faces >= 40 px (the
    min face size of face_groups / refs_from_items, measured in the INDEXED image's pixels), cosine of faces found in both.
Usage (GPU job, main env): python eval/rendition480_parity.py -> eval/rendition480_parity.json"""
import csv
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "eval")
sys.path.insert(0, "src")
from coreml_preproc_parity import QUERIES, dev, enc, load_image, model, pre, tok  # noqa: E402
import torch.nn.functional as F  # noqa: E402

R = 480


def rendition(im, side=R, q=80):
    im = im.copy(); im.thumbnail((side, side), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def stats(c):
    return dict(mean=round(float(np.mean(c)), 4), p5=round(float(np.percentile(c, 5)), 4), min=round(float(np.min(c)), 4), n=len(c))


OI = sorted(Path("data/public/raw/openimages/images").glob("*.jpg"))
PX = sorted(Path("data/public/pexels_frames").glob("*.jpg"))
out = {"rendition": f"Lanczos long side {R}, JPEG q80", "image": {}, "retrieval": {}, "faces": {}}
with torch.no_grad():
    T = F.normalize(model.encode_text(tok(QUERIES).to(dev)).float(), dim=-1).cpu()

for name, paths in [("openimages", OI), ("pexels_frames", PX)]:
    pool = random.Random(1).sample(paths, 2000)
    full = [load_image(str(p)) for p in pool]
    S = enc(model, [pre(im) for im in full])
    P = enc(model, [pre(rendition(im)) for im in full], half=True)
    c = (S * P).sum(1).numpy()
    out["image"][name] = dict(first200=stats(c[:200]), all2000=stats(c),
                              full_long_side=stats([max(im.size) for im in full[:200]]))
    res = {}
    for k in (600, 100, 20):
        ov = [len(set(torch.topk(S @ T[i], k).indices.tolist()) & set(torch.topk(P @ T[i], k).indices.tolist())) / k
              for i in range(len(QUERIES))]
        res[f"overlap@{k}"] = dict(mean=round(float(np.mean(ov)), 3), min=round(float(np.min(ov)), 3), n_queries=len(ov))
        if k == 20:
            res["per_query@20"] = dict(zip(QUERIES, [round(x, 2) for x in ov]))
    out["retrieval"][name + "2000"] = res
    print(name, out["image"][name], res, flush=True)
    json.dump(out, open("eval/rendition480_parity.json", "w"), indent=1)

# faces
from findpics.face_profiles import SHIPPED  # noqa: E402
from findpics.models import FaceEncoder  # noqa: E402
fe = FaceEncoder(SHIPPED)
face_ids = {r["ImageID"] for r in csv.DictReader(open("data/public/raw/openimages/val_labels.csv"))
            if r["LabelName"] == "/m/0dzct" and r["Confidence"] == "1"}
oi_faces = [p for p in OI if p.stem in face_ids]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


for name, paths in [("openimages_human_face", random.Random(2).sample(oi_faces, min(400, len(oi_faces)))),
                    ("pexels_frames", random.Random(2).sample(PX, 400))]:
    n_full = n_small = n_full40 = n_small40 = n_both = 0
    cos, cos40, missed_px = [], [], []
    for p in paths:
        im = load_image(str(p)); sm = rendition(im); s = im.size[0] / sm.size[0]
        ff, fs = fe.faces(im), fe.faces(sm)
        n_full += len(ff); n_small += len(fs)
        n_full40 += sum(min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= 40 for f in ff)
        n_small40 += sum(min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= 40 for f in fs)
        used = set()
        for f in ff:
            best, bj = 0.0, -1
            for j, g in enumerate(fs):
                if j in used:
                    continue
                v = iou(f["bbox"], [x * s for x in g["bbox"]])
                if v > best:
                    best, bj = v, j
            px_small = min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) / s
            if best >= 0.5:
                used.add(bj); n_both += 1
                c = float(np.dot(f["emb"].astype(np.float32), fs[bj]["emb"].astype(np.float32)))
                cos.append(c)
                if px_small >= 40:
                    cos40.append(c)
            else:
                missed_px.append(px_small)
    mp = np.array(missed_px) if missed_px else np.zeros(0)
    out["faces"][name] = dict(
        photos=len(paths), faces_full=n_full, faces_480=n_small, matched=n_both,
        faces_full_ge40px=n_full40, faces_480_ge40px=n_small40,
        missed_at_480_by_size_at_480=dict(lt20=int((mp < 20).sum()), px20_40=int(((mp >= 20) & (mp < 40)).sum()),
                                          ge40=int((mp >= 40).sum())),
        cos_matched=stats(cos) if cos else None, cos_matched_ge40px_at_480=stats(cos40) if cos40 else None)
    print(name, out["faces"][name], flush=True)
    json.dump(out, open("eval/rendition480_parity.json", "w"), indent=1)
print("R480_OK", flush=True)
