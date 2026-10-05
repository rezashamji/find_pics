"""InsightFace recognizer (ONNX) -> Core ML for the phone: ONNX -> PyTorch (onnx2torch) -> trace -> coremltools.
Input "face": 1x3x112x112 float, RGB, (x - 127.5) / 127.5 of the aligned face crop (insightface norm_crop + swapRB).
Output: 512-d embedding (L2-normalized here). Checks the converted model's output against onnxruntime on random input.
NOTE: buffalo_l weights are licensed for non-commercial research only (dev / personal testing); see JOURNAL 10-05.
Usage (envs/coreml + onnx2torch): python scripts/convert_face_coreml.py [pack]"""
import sys
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
import onnx
from onnx2torch import convert

pack = sys.argv[1] if len(sys.argv) > 1 else "buffalo_l"
src = {"buffalo_l": "models/insightface/models/buffalo_l/w600k_r50.onnx"}[pack]
net = convert(onnx.load(src)).eval()


class Normed(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.nn.functional.normalize(self.m(x), dim=-1)


x = torch.randn(1, 3, 112, 112)
with torch.no_grad():
    tr = torch.jit.trace(Normed(net), x)
ml = ct.convert(tr, inputs=[ct.TensorType(name="face", shape=x.shape)], convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
out = Path(f"models/coreml/face_{pack}.mlpackage"); ml.save(str(out))
import onnxruntime as ort
s = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
ref = s.run(None, {s.get_inputs()[0].name: x.numpy()})[0]; ref /= np.linalg.norm(ref, axis=1, keepdims=True)
with torch.no_grad():
    got = Normed(net)(x).numpy()
print("torch vs onnxruntime cosine:", float((ref * got).sum()))
print("saved", out, sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6, "MB")
print("FACE_COREML_OK")
