"""Commercial-licence face recognizers vs buffalo_l (non-commercial), same protocol as eval_face_models.py.
One shared detector (buffalo_l SCRFD det_10g) + one 5-point alignment (insightface norm_crop, 112x112) per image,
then every recognizer embeds the SAME aligned crop, so only the recognizer differs.
Recognizers: buffalo_l w600k_r50 (reference, non-commercial), AuraFace glintr100 (Apache-2.0),
SFace 2021dec (OpenCV Zoo, Apache-2.0 file, training data undocumented), HyperFace-50k-StyleGAN and
HyperFace-10k-LDM (Idiap, MIT weights, IResNet50 trained only on synthetic HyperFace images).
Datasets: DigiFace-1M (300 rendered identities x 72, as eval_face_models.py; none of these models was trained on it)
and CelebA test split (real celebrities, 755 identities with >= 13 images; evaluation use only).
Caveat for CelebA: real-data-trained models (buffalo_l, maybe AuraFace/SFace) may have seen these celebrities.
Recall is compared at EQUAL wrong-match rates: buffalo_l's rate at its product threshold 0.40, and 10x lower.
Usage (fp env, GPU): python eval/eval_face_commercial.py [--limit_ids N]
"""
import argparse
import glob
import io
import json
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics"
DIGI = f"{ROOT}/data/public/raw/digiface/subjects_0-1999_72_imgs.zip"
CELEBA = sorted(glob.glob(f"{ROOT}/data/public/raw/celeba/img_align+identity+attr/test-*.parquet"))
IF = f"{ROOT}/models/insightface/models"
N_REF = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def canvas(im, resize=True):  # DigiFace: identical to eval_face_models.py (224 px face on a 448 gray canvas);
    c = Image.new("RGB", (448, 448), (127, 127, 127))  # CelebA (178x218): pasted at native size, no aspect distortion
    if resize:
        c.paste(im.resize((224, 224)), (112, 112))
    else:
        c.paste(im, ((448 - im.width) // 2, (448 - im.height) // 2))
    return c


def digiface_items(n_id):
    z = zipfile.ZipFile(DIGI)
    for i in range(n_id):
        for k in range(72):
            yield i, Image.open(io.BytesIO(z.read(f"{i}/{k}.png"))).convert("RGB")


def celeba_items(n_id):
    import pyarrow.parquet as pq
    rows = []
    for f in CELEBA:
        t = pq.read_table(f, columns=["image", "celeb_id"])
        rows += list(zip(t.column("celeb_id").to_pylist(), (d["bytes"] for d in t.column("image").to_pylist())))
    u, c = np.unique([r[0] for r in rows], return_counts=True)
    keep = set(u[c >= N_REF + 5][:n_id].tolist())
    for cid, b in rows:
        if cid in keep:
            yield int(cid), Image.open(io.BytesIO(b)).convert("RGB")


class Recognizers:
    def __init__(self, rgb_check=False):
        self.rgb_check = rgb_check  # also feed HyperFace models RGB (channel-order sanity check)
        import onnxruntime as ort
        try:  # same as findpics.models.FaceEncoder: CUDA 12 / cuDNN 9 from pip packages for ORT's CUDA provider
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
        except Exception as e:
            print("onnxruntime preload_dlls failed:", e)
        import insightface
        from insightface.app import FaceAnalysis
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.det = FaceAnalysis(name="buffalo_l", root=f"{ROOT}/models/insightface", providers=prov,
                                allowed_modules=["detection"])
        self.det.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        self.onnx = {"buffalo_l": insightface.model_zoo.get_model(f"{IF}/buffalo_l/w600k_r50.onnx", providers=prov),
                     "auraface": insightface.model_zoo.get_model(f"{IF}/auraface/glintr100.onnx", providers=prov)}
        for m in self.onnx.values():
            m.prepare(ctx_id=0)
        self.sface = cv2.FaceRecognizerSF.create(f"{ROOT}/models/sface/face_recognition_sface_2021dec.onnx", "")
        sys.path.insert(0, f"{ROOT}/models/hyperface")
        import net
        self.torch = {}
        for name, ck in [("hyperface_50k_stylegan", "HyperFace_50k_StyleGAN.ckpt"), ("hyperface_10k_ldm", "HyperFace_10k_LDM.ckpt")]:
            m = net.build_model("ir_50")
            sd = torch.load(f"{ROOT}/models/hyperface/{ck}", map_location="cpu", weights_only=False)["state_dict"]
            missing = m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
            print(name, "loaded", missing, flush=True)
            self.torch[name] = m.eval().to(DEV)
        self.names = list(self.onnx) + ["sface"] + list(self.torch) + ([n + "_rgb" for n in self.torch] if rgb_check else [])
        self.ms = {n: 0.0 for n in self.names}

    def align(self, pil):
        bgr = np.asarray(pil)[:, :, ::-1].copy()
        f = self.det.get(bgr)
        if not f:
            return None
        from insightface.utils.face_align import norm_crop
        return norm_crop(bgr, max(f, key=lambda d: d.det_score).kps, 112)

    @torch.no_grad()
    def embed(self, crops):  # crops: list of 112x112 BGR uint8
        out = {}
        for n, m in self.onnx.items():
            t = time.time(); out[n] = m.get_feat(crops); self.ms[n] += time.time() - t
        t = time.time(); out["sface"] = np.concatenate([self.sface.feature(c) for c in crops]); self.ms["sface"] += time.time() - t
        x = torch.from_numpy(((np.stack(crops).astype(np.float32) / 255.0) - 0.5) / 0.5).permute(0, 3, 1, 2).contiguous().to(DEV)  # BGR, as HyperFace inference.to_input
        for n, m in self.torch.items():
            t = time.time(); f, _ = m(x); out[n] = f.float().cpu().numpy(); self.ms[n] += time.time() - t
            if self.rgb_check:
                f, _ = m(x.flip(1).contiguous()); out[n + "_rgb"] = f.float().cpu().numpy()
        return {n: (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32) for n, v in out.items()}


def scores(E, who, seed=0):  # identical to eval_face_models.py
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for i in np.unique(who):
        mine = np.where(who == i)[0]
        if len(mine) < N_REF + 5:
            continue
        ref = rng.choice(mine, N_REF, replace=False); tgt = np.setdiff1d(mine, ref); others = np.where(who != i)[0]
        s = (E @ E[ref].T).max(1)
        pos.append(s[tgt]); neg.append(s[others])
    return np.concatenate(pos), np.concatenate(neg)


def run(R, ds, items, pad):
    embs = {n: [] for n in R.names}; who, miss, n_img, batch, bwho, sheet = [], 0, 0, [], [], []
    def flush():
        for n, v in R.embed(batch).items():
            embs[n].append(v)
        who.extend(bwho); batch.clear(); bwho.clear()
    for i, im in items:
        n_img += 1
        c = R.align(canvas(im, resize=(ds == "digiface")))
        if c is None:
            miss += 1; continue
        if len(sheet) < 48 and n_img % 97 == 0:
            sheet.append(c)
        batch.append(c); bwho.append(i)
        if len(batch) == 256:
            flush()
    if batch:
        flush()
    who = np.array(who)
    if len(sheet) >= 8:
        grid = np.concatenate([np.concatenate([cv2.resize(c, (224, 224)) for c in sheet[r * 8:(r + 1) * 8]], 1) for r in range(len(sheet) // 8)], 0)
        cv2.imwrite(f"{ROOT}/eval/audits/face_commercial_crops_{ds}.jpg", grid) if not R.rgb_check else None
    res = {n: scores(np.concatenate(v), who) for n, v in embs.items()}
    fmr0 = float((res["buffalo_l"][1] >= 0.40).mean())
    out = dict(images=n_img, not_detected=miss, identities=int(len(np.unique(who))), buffalo_rate_at_0_40=fmr0)
    for n, (pos, neg) in res.items():
        row = dict(n_pos=int(len(pos)), n_neg=int(len(neg)), ms_per_face=round(1000 * R.ms[n] / max(1, n_img - miss), 2))
        for lbl, fmr in [("at_buffalo_0.40_rate", fmr0), ("at_10x_lower_rate", fmr0 / 10)]:
            t = float(np.quantile(neg, 1 - fmr))
            row[lbl] = dict(threshold=round(t, 3), recall=round(float((pos >= t).mean()), 4),
                            hits=int((pos >= t).sum()), wrong_match_rate=round(float((neg >= t).mean()), 6))
        out[n] = row
        R.ms[n] = 0.0
    print(ds, json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit_ids", type=int, default=0); ap.add_argument("--rgb_check", action="store_true"); a = ap.parse_args()
    R = Recognizers(a.rgb_check)
    out = {"digiface": run(R, "digiface", digiface_items(a.limit_ids or 300), pad=True),
           "celeba": run(R, "celeba", celeba_items(a.limit_ids or 10**6), pad=True)}
    if not a.limit_ids:
        json.dump(out, open(f"{ROOT}/eval/results_face_commercial{'_rgbcheck' if a.rgb_check else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
