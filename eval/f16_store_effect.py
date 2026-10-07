"""How much does storing vectors as Float16 (the phone's binary index store, FindPicsCore/IndexStore.swift) change
cosine scores and rankings?  Public data only.

image: N DISBench photos embedded in FLOAT32 (PE-Core-L-14-336 run in fp32; the server index itself is already fp16,
       so it cannot answer this), queries = text vectors (fp32, as on the phone). Compare cos(q, v32) vs cos(q, f16(v32)):
       |delta| distribution, top-k overlap, and image-image cosines (bursts 0.90, subject neighbours).
faces: AuraFace+flip fp32 vectors of CelebA / DigiFace (data/public/derived/face_free_*.npz): pair cosines and how many
       same/different-person decisions flip at the shipped profile's cuts (accept 0.53, group/expand 0.62, maybe 0.30).
Usage: python eval/f16_store_effect.py images N   (GPU)   |   python eval/f16_store_effect.py faces   (CPU)
"""
import json
import sys

import numpy as np

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics"
QUERIES = ["a dog on the beach", "birthday cake with candles", "a red car", "people hiking in the mountains",
           "a plate of sushi", "a cat sleeping", "snow on trees", "a city at night", "a bicycle", "flowers in a vase",
           "a selfie", "a wedding", "a screenshot of a phone", "a sunset over the ocean", "children playing soccer",
           "a coffee cup", "a bookshelf", "fireworks", "a horse", "a pizza"]


def f16(x):
    return x.astype(np.float16).astype(np.float32)


def stats(d):
    a = np.abs(d.ravel())
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)), "p99": float(np.quantile(a, 0.99)),
            "max": float(a.max())}


def images(n):
    import pandas as pd
    import torch
    from PIL import Image
    sys.path.insert(0, ROOT + "/src")
    from findpics.models import ImageTextEncoder
    items = pd.read_parquet(ROOT + "/data/public/index_disbench/items.parquet")
    items = items[items.media == "photo"].sample(n=n, random_state=0)
    enc = ImageTextEncoder(dtype=torch.float32)
    vs = []
    paths = items.path.tolist()
    for i in range(0, len(paths), 64):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
        x = torch.stack([enc.preprocess(im) for im in ims]).to(enc.device, torch.float32)
        with torch.inference_mode():
            f = torch.nn.functional.normalize(enc.model.encode_image(x).float(), dim=-1)
        vs.append(f.cpu().numpy())
    V = np.concatenate(vs).astype(np.float32)
    Q = enc.texts(QUERIES).astype(np.float32)
    np.save(ROOT + "/.cache/f16_store_effect_img32.npy", V)
    np.save(ROOT + "/.cache/f16_store_effect_q32.npy", Q)
    report_images(V, Q)


def report_images(V, Q):
    V16 = f16(V)
    S32, S16 = Q @ V.T, Q @ V16.T
    out = {"images": len(V), "queries": len(Q), "query_image_cos_delta": stats(S16 - S32)}
    for k in (10, 50, 150):
        ov = [len(set(np.argsort(-S32[i])[:k]) & set(np.argsort(-S16[i])[:k])) / k for i in range(len(Q))]
        out[f"top{k}_overlap_mean"] = float(np.mean(ov)); out[f"top{k}_overlap_min"] = float(np.min(ov))
    sub = V[:2000]; G32 = sub @ V.T; G16 = f16(sub) @ V16.T
    out["image_image_cos_delta"] = stats(G16 - G32)
    out["burst_0.90_flips"] = int(((G32 >= 0.90) != (G16 >= 0.90)).sum()); out["burst_pairs"] = int(G32.size)
    out["burst_0.90_pairs_f32"] = int((G32 >= 0.90).sum())
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT + "/eval/f16_store_effect_images.json", "w"), indent=1)


def faces():
    out = {}
    for name in ("celeba", "digiface"):
        z = np.load(f"{ROOT}/data/public/derived/face_free_{name}.npz")
        E = z["auraface_flip"].astype(np.float32); who = z["who"]
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        rng = np.random.default_rng(0); idx = rng.choice(len(E), size=min(4000, len(E)), replace=False)
        A, W = E[idx], who[idx]
        S32 = A @ E.T; S16 = f16(A) @ f16(E).T
        same = W[:, None] == who[None, :]
        r = {"faces": int(len(E)), "pairs": int(S32.size), "cos_delta": stats(S16 - S32)}
        for cut in (0.30, 0.53, 0.62):
            flip = (S32 >= cut) != (S16 >= cut)
            r[f"flips_at_{cut}"] = int(flip.sum())
            r[f"flips_at_{cut}_same_person"] = int((flip & same).sum())
            r[f"pairs_above_{cut}_f32"] = int((S32 >= cut).sum())
        out[name] = r
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT + "/eval/f16_store_effect_faces.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "images":
        images(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
    elif sys.argv[1] == "report_images":
        report_images(np.load(ROOT + "/.cache/f16_store_effect_img32.npy"), np.load(ROOT + "/.cache/f16_store_effect_q32.npy"))
    else:
        faces()
