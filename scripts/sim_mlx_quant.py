"""Simulate the PHONE's 4-bit weights on the cluster: apply MLX's affine 4-bit quantization (group 64, round to nearest,
per-group scale = (max - min) / 15, bias = min; scale/bias stored in bf16) to exactly the tensors the mlx-community
4-bit release quantizes (read from its weight index: the language model's linears, embeddings and lm_head; the vision
tower stays full precision), then write the de-quantized weights as a normal HF checkpoint vLLM can serve.
So "how good is the judge on the phone" is measured with the phone's numbers, not the server's bf16 ones.
Usage (GPU node, main or vLLM env): python scripts/sim_mlx_quant.py Qwen/Qwen3.5-9B mlx-community/Qwen3.5-9B-4bit models/qwen35_9b_mlx4sim
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

GROUP, BITS = 64, 4


def fake_quant(w: torch.Tensor) -> torch.Tensor:
    shape, dt = w.shape, w.dtype
    g = w.to("cuda", torch.float32).reshape(-1, GROUP)
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    scale = ((hi - lo) / (2 ** BITS - 1)).clamp_min(1e-8).to(torch.bfloat16).float()
    bias = lo.to(torch.bfloat16).float()
    q = ((g - bias) / scale).round().clamp(0, 2 ** BITS - 1)
    return (q * scale + bias).reshape(shape).to(dt).cpu()


def hf_name(mlx: str) -> str:
    if mlx == "language_model.lm_head":
        return "lm_head"
    assert mlx.startswith("language_model.model."), mlx
    return "model.language_model." + mlx[len("language_model.model."):]


def main():
    base, mlx_repo, out = sys.argv[1:4]
    global BITS
    BITS = int(sys.argv[4]) if len(sys.argv) > 4 else 4    # 3: the 9B that fits the app memory budget
    idx = json.load(open(hf_hub_download(mlx_repo, "model.safetensors.index.json")))["weight_map"]
    want = {hf_name(k[:-len(".scales")]) + ".weight" for k in idx if k.endswith(".scales")}
    src = Path(snapshot_download(base))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in sorted(src.iterdir()):
        if f.suffix == ".safetensors":
            t = load_file(str(f))
            for k in list(t):
                if k in want:
                    assert t[k].shape[-1] % GROUP == 0, (k, t[k].shape)
                    t[k] = fake_quant(t[k]); done.add(k)
            save_file(t, str(out / f.name), metadata={"format": "pt"})
            print(f.name, len(done), flush=True)
        elif f.is_file():
            shutil.copy(f, out / f.name)
    missing = want - done - {"lm_head.weight"}    # tied embeddings (4B): lm_head IS embed_tokens, quantized above
    print("quantized", len(done), "of", len(want), "missing", sorted(want - done)[:5], flush=True)
    assert not missing


if __name__ == "__main__":
    main()
