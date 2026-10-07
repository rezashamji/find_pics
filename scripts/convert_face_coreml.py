"""Face recognizer (ONNX) -> Core ML for the phone: ONNX -> PyTorch (onnx2torch) -> trace -> coremltools, fp16.
Input "face": 1x3x112x112 float, RGB, (x - 127.5) / 127.5 of the aligned face crop (insightface norm_crop + swapRB;
both packs are insightface ArcFaceONNX models with input_mean = input_std = 127.5, checked 10-07).
Output: 512-d embedding, L2-normalized. With flip (AuraFace, the shipped model): the graph embeds the crop AND its
horizontal mirror, L2-normalizes each, adds them and L2-normalizes again (flip test-time averaging, eval/face_free_deepdive.md:
CelebA 0.908 -> 0.927 single-image), so the app makes one prediction call per face and needs no flip code of its own.
Checks the traced model against onnxruntime (random input + real aligned faces from data/public/derived/face_check_crops.npz
when present, written by eval/face_fixtures.py). Core ML itself cannot run on Linux: the app's Self-check compares the
phone's output with the server's (SelfCheck/refs.json).
Packs: auraface (Apache-2.0, shipped), buffalo_l (NON-COMMERCIAL weights: dev / personal testing only, not shipped).
Usage (envs/coreml, which has onnx2torch; `module load gcc/13.2.0-fasrc01` first: libmodelpackage needs GLIBCXX_3.4.26): python scripts/convert_face_coreml.py [auraface|buffalo_l]"""
import sys
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
import onnx
from onnx2torch import convert

PACKS = {"auraface": ("models/insightface/models/auraface/glintr100.onnx", True),
         "buffalo_l": ("models/insightface/models/buffalo_l/w600k_r50.onnx", False)}
pack = sys.argv[1] if len(sys.argv) > 1 else "auraface"
src, flip = PACKS[pack]
net = convert(onnx.load(src)).eval()
nrm = torch.nn.functional.normalize


class Embed(torch.nn.Module):
    def __init__(self, m, flip): super().__init__(); self.m = m; self.flip = flip
    def forward(self, x):
        e = nrm(self.m(x), dim=-1)
        if self.flip:
            e = nrm(e + nrm(self.m(torch.flip(x, dims=[3])), dim=-1), dim=-1)
        return e


x = torch.randn(1, 3, 112, 112)
wrapped = Embed(net, flip).eval()
with torch.no_grad():
    tr = torch.jit.trace(wrapped, x)
ml = ct.convert(tr, inputs=[ct.TensorType(name="face", shape=x.shape)], outputs=[ct.TensorType(name="embedding")],
                convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
ml.short_description = f"{pack} face embedding" + (" (crop + mirror averaged)" if flip else "")
out = Path(f"models/coreml/face_{pack}.mlpackage"); ml.save(str(out))

import onnxruntime as ort
s = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
name = s.get_inputs()[0].name


def ort_embed(b):  # b: N x 3 x 112 x 112, the same normalized input the app feeds
    def one(z):
        r = s.run(None, {name: z.astype(np.float32)})[0]; return r / np.linalg.norm(r, axis=1, keepdims=True)
    e = one(b)
    if flip:
        e = e + one(b[:, :, :, ::-1].copy()); e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


worst = 1.0
batches = [x.numpy()]
ck = Path("data/public/derived/face_check_crops.npz")
if ck.exists():   # aligned RGB uint8 crops N x 112 x 112 x 3 of public faces
    crops = np.load(ck)["rgb"].astype(np.float32)
    batches += [((c - 127.5) / 127.5).transpose(2, 0, 1)[None] for c in crops]
for b in batches:
    with torch.no_grad():
        got = tr(torch.from_numpy(b.astype(np.float32))).numpy()
    worst = min(worst, float((ort_embed(b) * got).sum()))
print(f"traced torch vs onnxruntime: worst cosine {worst:.6f} over {len(batches)} inputs ({len(batches) - 1} real faces)")
assert worst > 0.9999, "conversion changed the embedding"
print("saved", out, round(sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6, 1), "MB")
print("FACE_COREML_OK")
