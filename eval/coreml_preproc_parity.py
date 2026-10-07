"""Why the phone's PE-Core-B-16 image vectors disagree with the server's (Self-check 10-07: scene.png 0.9345,
stripes.png 0.9765, bar > 0.99). Reproduce each candidate difference in PyTorch and measure cosine vs the server vector.

Server (what the index and SelfCheck/refs.json use): findpics.media.load_image (JPEG draft >= 1600, EXIF transpose,
bicubic thumbnail to 1600) -> open_clip preprocess for PE-Core-B-16 = PIL im.resize((224, 224), BILINEAR) (squash, no
crop; PIL's bilinear widens its triangle filter by the shrink factor = antialiased) -> /255 -> (x - 0.5) / 0.5.
Phone before the fix (Embedder.swift): CIImage.transformed(by: scale) + CIContext.render = Core Image's linear sampler,
one 2x2 bilinear tap per output pixel and no widening (= torch interpolate bilinear, antialias=False); int8 weights.

Parts: (1) per-variant cosines on the two Self-check images, 200 DISBench photos (<= 500 px) and 200 Open Images photos
(1024 px, nearer the phone's 1280 px read); (2) top-k overlap (fast-mode head 600) for everyday text queries on one
DISBench library and a 2000-photo Open Images pool, server vs phone-like vectors.
Usage (GPU job): python eval/coreml_preproc_parity.py  -> eval/coreml_preproc_parity.json"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "src")
from findpics.media import load_image  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
NAME = "hf-hub:timm/PE-Core-B-16"
model, _, pre = open_clip.create_model_and_transforms(NAME); model = model.to(dev).eval()
tok = open_clip.get_tokenizer(NAME)
S = 224


def to_t(a):                                   # uint8 HxWx3 -> normalized 3x224x224 (already 224)
    return (torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255 - 0.5) / 0.5


def ci_bilinear(im, aa=False):                 # Core Image affine + linear sampler (one bilinear tap), round to RGBA8
    x = torch.from_numpy(np.asarray(im.convert("RGB"))).permute(2, 0, 1)[None].float()
    y = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=aa)
    return to_t(y[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy())


def pil(im, f):
    return to_t(np.asarray(im.convert("RGB").resize((S, S), f)))


def center_crop(im):                           # resize shorter side to 224 (bicubic) + center crop (classic CLIP)
    w, h = im.size; s = S / min(w, h); im = im.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BICUBIC)
    w, h = im.size; l, t = (w - S) // 2, (h - S) // 2
    return to_t(np.asarray(im.crop((l, t, l + S, t + S))))


def srgb_to_lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin_to_srgb(c): return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def linear_light(im):                          # antialiased resize done in linear light (CI's default working space)
    a = srgb_to_lin(np.asarray(im.convert("RGB")).astype(np.float64) / 255)
    x = torch.from_numpy(a).permute(2, 0, 1)[None].float()
    y = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=True)[0].permute(1, 2, 0).numpy()
    return to_t((lin_to_srgb(y) * 255).round().clip(0, 255).astype(np.uint8))


P3_TO_SRGB = np.array([[1.2249, -0.2247, 0.0], [-0.0420, 1.0419, 0.0], [-0.0197, -0.0786, 1.0979]])


def p3_managed(im):                            # raw values tagged Display P3, colour-managed to sRGB (what a CIContext
    a = srgb_to_lin(np.asarray(im.convert("RGB")).astype(np.float64) / 255)   # rendering to sRGB does; PIL ignores ICC)
    a = np.clip(a @ P3_TO_SRGB.T, 0, 1)
    return pre(Image.fromarray((lin_to_srgb(a) * 255).round().clip(0, 255).astype(np.uint8)))


VARIANTS = {
    "server (PIL bilinear AA squash)": lambda im: pre(im),
    "CI affine bilinear, no AA (phone before fix)": ci_bilinear,
    "torch bilinear AA (sanity)": lambda im: ci_bilinear(im, aa=True),
    "PIL nearest": lambda im: pil(im, Image.NEAREST),
    "PIL bicubic": lambda im: pil(im, Image.BICUBIC),
    "PIL lanczos": lambda im: pil(im, Image.LANCZOS),
    "center crop bicubic": center_crop,
    "linear-light AA resize": linear_light,
    "P3-tagged -> sRGB managed": p3_managed,
    "vertical flip": lambda im: pre(im.transpose(Image.FLIP_TOP_BOTTOM)),
    "horizontal flip": lambda im: pre(im.transpose(Image.FLIP_LEFT_RIGHT)),
}


@torch.no_grad()
def enc(m, xs, half=False):
    out = []
    for i in range(0, len(xs), 128):
        x = torch.stack(xs[i:i + 128]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=half and dev == "cuda"):
            out.append(F.normalize(m.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(out)


def int8_copy(m):                              # coremltools linear_quantize_weights(linear_symmetric, per_channel,
    import copy                                #   weight_threshold=2048) as in scripts/quantize_coreml.py, simulated:
    q = copy.deepcopy(m)                       #   every weight tensor with > 2048 elements, int8 per output channel,
    n = 0                                      #   scale = max|w| / 127 (symmetric)
    with torch.no_grad():
        for name, p in q.visual.named_parameters():
            if p.dim() >= 2 and p.numel() > 2048:
                w = p.data.float(); s = w.abs().flatten(1).amax(1).clamp_min(1e-12) / 127
                s = s.view(-1, *[1] * (w.dim() - 1)); p.data.copy_((w / s).round().clamp(-127, 127) * s); n += 1
    print("int8-simulated weight tensors:", n, flush=True)
    return q


def selfcheck_images():
    d = Path("ios/FindPicsApp.swiftpm/Sources/SelfCheck"); refs = json.load(open(d / "refs.json"))["images"]
    return {n: Image.open(d / n).convert("RGB") for n in refs}, {n: torch.tensor(v) for n, v in refs.items()}


def main():
    random.seed(0)
    out = {"variants": {}, "notes": __doc__.split("\n")[0]}
    sc, sc_ref = selfcheck_images()
    db_dir = Path("data/public/raw/disbench/images/images")
    db = sorted(p for u in sorted(db_dir.iterdir()) for p in u.glob("*.jpg"))
    db = random.sample(db, 200)
    oi = random.sample(sorted(Path("data/public/raw/openimages/images").glob("*.jpg")), 200)
    sets = {"selfcheck": [sc[n] for n in sc], "disbench200": [load_image(str(p)) for p in db],
            "openimages200": [load_image(str(p)) for p in oi]}
    q8 = int8_copy(model)
    base = {k: enc(model, [pre(im) for im in ims]) for k, ims in sets.items()}
    # the server ref in refs.json was made on CPU fp32: check our server vectors reproduce it
    out["refs_json_vs_recomputed"] = {n: float(base["selfcheck"][i] @ sc_ref[n].float()) for i, n in enumerate(sc)}
    for vname, fn in VARIANTS.items():
        xs = {k: [fn(im) for im in ims] for k, ims in sets.items()}
        for mname, m, half in [("fp32", model, False), ("fp16", model, True), ("int8", q8, False), ("int8+fp16", q8, True)]:
            if mname != "fp32" and vname not in ("server (PIL bilinear AA squash)", "CI affine bilinear, no AA (phone before fix)"):
                continue
            row = {}
            for k in sets:
                c = (enc(m, xs[k], half) * base[k]).sum(1).numpy()
                row[k] = {n: round(float(c[i]), 4) for i, n in enumerate(sc)} if k == "selfcheck" else \
                    dict(mean=round(float(c.mean()), 4), p10=round(float(np.percentile(c, 10)), 4),
                         min=round(float(c.min()), 4), n=len(c))
            out["variants"][f"{vname} | {mname}"] = row
            print(f"{vname} | {mname}: {row}", flush=True)
    json.dump(out, open("eval/coreml_preproc_parity.json", "w"), indent=1)
    retrieval(out)
    json.dump(out, open("eval/coreml_preproc_parity.json", "w"), indent=1)
    print("PARITY_OK", flush=True)


QUERIES = ["a dog", "a cat", "food on a plate", "a birthday cake", "a beach", "a sunset", "flowers", "a car",
           "a bicycle", "people at a party", "a mountain", "a city street at night", "a baby", "a church",
           "a christmas tree", "a boat", "a bird", "a selfie", "a group photo", "snow"]


def retrieval(out):
    lib = Path("data/public/raw/disbench/images/images/69099808@N00")
    pools = {"disbench_69099808@N00": sorted(lib.glob("*.jpg")),
             "openimages2000": random.Random(1).sample(sorted(Path("data/public/raw/openimages/images").glob("*.jpg")), 2000)}
    with torch.no_grad():
        T = F.normalize(model.encode_text(tok(QUERIES).to(dev)).float(), dim=-1).cpu()
    q8 = int8_copy(model)
    out["retrieval"] = {}
    for pname, paths in pools.items():
        ims = [load_image(str(p)) for p in paths]
        V = {"server": enc(model, [pre(im) for im in ims]),
             "phone_before (CI no-AA, int8)": enc(q8, [ci_bilinear(im) for im in ims]),
             "CI no-AA, fp32 weights": enc(model, [ci_bilinear(im) for im in ims]),
             "server preprocess, int8": enc(q8, [pre(im) for im in ims])}
        cos = {k: round(float((v * V["server"]).sum(1).mean()), 4) for k, v in V.items()}
        res = {"n_photos": len(paths), "mean_cos_vs_server": cos, "queries": {}}
        for k in [600, 100, 20]:
            kk = min(k, len(paths))
            for vn, v in V.items():
                if vn == "server":
                    continue
                ov = []
                for qi, q in enumerate(QUERIES):
                    a = set(torch.topk(V["server"] @ T[qi], kk).indices.tolist())
                    b = set(torch.topk(v @ T[qi], kk).indices.tolist())
                    ov.append(len(a & b) / kk)
                    res["queries"].setdefault(q, {})[f"{vn} overlap@{k}"] = round(len(a & b) / kk, 3)
                res[f"{vn} overlap@{k}"] = dict(mean=round(float(np.mean(ov)), 3), min=round(float(np.min(ov)), 3),
                                                n_queries=len(ov), k=kk)
        out["retrieval"][pname] = res
        print(pname, {k: v for k, v in res.items() if k != "queries"}, flush=True)


if __name__ == "__main__":
    main()
