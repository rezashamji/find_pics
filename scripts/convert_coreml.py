"""Convert the image and text towers of PE-Core-L14-336 to Core ML (.mlpackage, fp16) for on-device indexing/search on
iPhone/Mac. Conversion runs on Linux (coremltools); PREDICTION needs macOS/iOS, so on Linux we can only check that
conversion succeeds and the PyTorch-traced output matches the original model.
Output: models/coreml/pe_core_image.mlpackage, models/coreml/pe_core_text.mlpackage
Linux needs a newer C++ runtime for coremltools' package writer (system libstdc++ lacks GLIBCXX_3.4.26):
  module load gcc/13.2.0-fasrc01
The text tower goes through torch.export (TorchScript tracing fails inside nn.MultiheadAttention's shape casts).
Usage (envs/coreml): module load gcc/13.2.0-fasrc01 && python scripts/convert_coreml.py
"""
from pathlib import Path

import coremltools as ct
import numpy as np
import open_clip
import torch

import sys
NAME = sys.argv[1] if len(sys.argv) > 1 else "PE-Core-L-14-336"     # e.g. PE-Core-B-16 (phone candidate)
SIZE = 336 if "336" in NAME else (384 if "384" in NAME else 224)
TAG = "" if NAME == "PE-Core-L-14-336" else "_" + NAME.replace("-", "_")
OUT = Path("models/coreml"); OUT.mkdir(parents=True, exist_ok=True)
model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:timm/{NAME}")
model.eval()
tok = open_clip.get_tokenizer(f"hf-hub:timm/{NAME}")


class ImageTower(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.nn.functional.normalize(self.m.encode_image(x), dim=-1)


class TextTower(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, t): return torch.nn.functional.normalize(self.m.encode_text(t), dim=-1)


x = torch.randn(1, 3, SIZE, SIZE)
t = tok(["a photo of bread"])
with torch.no_grad():
    img = torch.jit.trace(ImageTower(model), x); txt = torch.jit.trace(TextTower(model), t)
    ref_i, ref_t = ImageTower(model)(x), TextTower(model)(t)
    print("trace max abs diff image", float((img(x) - ref_i).abs().max()), "text", float((txt(t) - ref_t).abs().max()))


class TextTowerI32(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, t): return torch.nn.functional.normalize(self.m.encode_text(t.long()), dim=-1)


with torch.no_grad():
    txt_ep = torch.export.export(TextTowerI32(model), (t.to(torch.int32),)).run_decompositions({})
for name, prog, inp in [(f"pe_core_image{TAG}", img, [ct.TensorType(name="pixels", shape=x.shape)]),
                        (f"pe_core_text{TAG}", txt_ep, None)]:
    ml = ct.convert(prog, inputs=inp, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS17)
    p = OUT / f"{name}.mlpackage"; ml.save(str(p))
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    print(f"saved {p} ({size:.0f} MB)")
print("COREML_OK")
