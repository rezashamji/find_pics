"""Apply a PEFT LoRA onto a given base checkpoint's weights (W += scale * B @ A) and save a servable copy. Used to test
the distilled planner on the PHONE's 4-bit weights (models/qwen35_4b_mlx4sim, scripts/sim_mlx_quant.py) instead of the
16-bit base it was trained on. Usage: python scripts/merge_lora_into.py <base_dir> <peft_dir> <out_dir>"""
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

base, peft, out = map(Path, sys.argv[1:4])
cfg = json.load(open(peft / "adapter_config.json")); scale = cfg["lora_alpha"] / cfg["r"]
L = load_file(str(peft / "adapter_model.safetensors"))
delta = {}
for k, v in L.items():
    m = re.search(r"(language_model\.layers\.\d+\..+)\.lora_([AB])\.weight$", k)
    delta.setdefault("model." + m.group(1) + ".weight", {})[m.group(2)] = v.float()
out.mkdir(parents=True, exist_ok=True); done = 0
for f in sorted(base.iterdir()):
    if f.suffix == ".safetensors":
        t = load_file(str(f))
        for k in list(t):
            if k in delta:
                d = delta[k]; t[k] = (t[k].float() + scale * d["B"] @ d["A"]).to(t[k].dtype); done += 1
        save_file(t, str(out / f.name), metadata={"format": "pt"})
    elif f.is_file():
        shutil.copy(f, out / f.name)
print("MERGED", done, "of", len(delta), "LoRA targets", flush=True)
assert done == len(delta)
