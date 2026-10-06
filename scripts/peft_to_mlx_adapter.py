"""Convert the distilled planner's PEFT LoRA (models/planner_4b27ball_lora) into the MLX adapter layout (mlx-lm /
mlx-swift-lm: adapter_config.json + adapters.safetensors; lora_a is (in, r) and lora_b is (r, out), i.e. PEFT's
lora_A / lora_B transposed; scale = alpha / r). Keys follow the mlx-community Qwen3.5 module names
("language_model.model.layers.N...."). Output: models/planner_4b27ball_mlx_adapter/ (~60 MB), to ship next to the
base 4-bit model so the phone loads ONE base model for judge and planner.
Usage: python scripts/peft_to_mlx_adapter.py [peft_dir] [out_dir]"""
import json
import re
import sys
from pathlib import Path

from safetensors.numpy import load_file, save_file

src = Path(sys.argv[1] if len(sys.argv) > 1 else "models/planner_4b27ball_lora")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "models/planner_4b27ball_mlx_adapter")
cfg = json.load(open(src / "adapter_config.json"))
t = load_file(str(src / "adapter_model.safetensors"))
res, layers, keys = {}, set(), set()
for k, v in t.items():
    m = re.search(r"language_model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$", k)
    assert m, k
    layer, mod, ab = int(m.group(1)), m.group(2), m.group(3)
    layers.add(layer); keys.add(mod)
    res[f"language_model.model.layers.{layer}.{mod}.lora_{ab.lower()}"] = v.T.astype("float16").copy()   # (r,in)->(in,r); fp16 halves the size
out.mkdir(parents=True, exist_ok=True)
save_file(res, str(out / "adapters.safetensors"))
json.dump(dict(fine_tune_type="lora", num_layers=len(layers),
               lora_parameters=dict(rank=cfg["r"], scale=cfg["lora_alpha"] / cfg["r"], dropout=0.0, keys=sorted(keys)),
               base_model="mlx-community/Qwen3.5-4B-4bit", source=str(src)),
          open(out / "adapter_config.json", "w"), indent=1)
print("MLX_ADAPTER", len(res), "tensors,", len(layers), "layers, keys", sorted(keys))
