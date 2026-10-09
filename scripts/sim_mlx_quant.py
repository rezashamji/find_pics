"""Simulate the PHONE's 4-bit weights on the cluster: apply MLX's affine 4-bit quantization (group 64, round to nearest,
per-group scale = (max - min) / 15, bias = min; scale/bias stored in bf16) to exactly the tensors the mlx-community
4-bit release quantizes (read from its weight index: the language model's linears, embeddings and lm_head; the vision
tower stays full precision), then write the de-quantized weights as a normal HF checkpoint vLLM can serve.
So "how good is the judge on the phone" is measured with the phone's numbers, not the server's bf16 ones.
Usage (GPU node, main or vLLM env): python scripts/sim_mlx_quant.py Qwen/Qwen3.5-9B mlx-community/Qwen3.5-9B-4bit models/qwen35_9b_mlx4sim

Compression variants (10-09, phone memory budget), all MLX affine round-to-nearest:
  4th positional arg / --lm-bits B   language-model linears + embeddings at B bits (default 4)
  --lm-group G                        their group size (default 64; MLX also takes 32 and 128)
  --embed-bits B                      embed_tokens (= tied lm_head) at B bits instead of --lm-bits
  --vision-bits B                     ALSO quantize the vision tower's Linear layers (attention qkv/proj, MLP fc1/fc2,
                                      merger + deepstack mergers) at B bits, group 64. patch_embed (a Conv3d) and
                                      pos_embed stay bf16: MLX's quantize() only converts Linear/Embedding, and we
                                      keep pos_embed exact (4.7 MB). Same layers mlx-swift's Qwen3VL declares Linear.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

VISION_LINEAR = (".attn.qkv.weight", ".attn.proj.weight", ".mlp.linear_fc1.weight", ".mlp.linear_fc2.weight",
                 ".linear_fc1.weight", ".linear_fc2.weight")


def fake_quant(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:     # min/max approximation of MLX
    shape, dt = w.shape, w.dtype
    g = w.to("cuda" if torch.cuda.is_available() else "cpu", torch.float32).reshape(-1, group)
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8).to(torch.bfloat16).float()
    bias = lo.to(torch.bfloat16).float()
    q = ((g - bias) / scale).round().clamp(0, 2 ** bits - 1)
    return (q * scale + bias).reshape(shape).to(dt).cpu()


def fake_quant_mlx(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """EXACT MLX affine_quantize + dequantize (mlx/ops.cpp, 10-2026): the larger-magnitude extreme is the anchor and the
    scale is nudged so 0.0 is representable (scale = edge / round(edge / scale)); ints from fp32 scales, de-quantized
    with the bf16-stored scales/biases. fake_quant above is the older min/max approximation (kept: the language model
    of every earlier phone-judge sim, so LM weights stay byte-identical to the baseline)."""
    shape, dt = w.shape, w.dtype
    g = w.to("cuda" if torch.cuda.is_available() else "cpu", torch.float32).reshape(-1, group)
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    n = float(2 ** bits - 1)
    mask = lo.abs() > hi.abs()
    scale = ((hi - lo) / n).clamp_min(1e-7)
    scale = torch.where(mask, scale, -scale)
    edge = torch.where(mask, lo, hi)
    q0 = (edge / scale).round()
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    q = ((g - bias) / scale).round().clamp(0, n)
    return (q * scale.to(torch.bfloat16).float() + bias.to(torch.bfloat16).float()).reshape(shape).to(dt).cpu()


def hf_name(mlx: str) -> str:
    if mlx == "language_model.lm_head":
        return "lm_head"
    assert mlx.startswith("language_model.model."), mlx
    return "model.language_model." + mlx[len("language_model.model."):]


def is_vision_linear(k: str, t: torch.Tensor) -> bool:
    return k.startswith("model.visual.") and k.endswith(VISION_LINEAR) and t.ndim == 2 and t.shape[-1] % 64 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("mlx_repo"); ap.add_argument("out")
    ap.add_argument("bits", nargs="?", type=int, default=None)          # old positional form (3: the 9B 3-bit sim)
    ap.add_argument("--lm-bits", type=int, default=4)
    ap.add_argument("--lm-group", type=int, default=64)
    ap.add_argument("--embed-bits", type=int, default=None)
    ap.add_argument("--vision-bits", type=int, default=None)
    ap.add_argument("--lm-exact", action="store_true", help="language model with the exact MLX rounding too")
    a = ap.parse_args()
    lm_bits = a.bits or a.lm_bits
    idx = json.load(open(hf_hub_download(a.mlx_repo, "model.safetensors.index.json")))["weight_map"]
    want = {hf_name(k[:-len(".scales")]) + ".weight" for k in idx if k.endswith(".scales")}
    all_language = not want     # index without .scales (e.g. mlx-community Qwen3-VL): MLX's default = every
    # language-model Linear / Embedding (and lm_head) whose input dim divides the group size; vision stays full
    src = Path(snapshot_download(a.base))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done, vis = set(), set()
    for f in sorted(src.iterdir()):
        if f.suffix == ".safetensors":
            t = load_file(str(f))
            for k in list(t):
                if all_language and (k.startswith("model.language_model.") or k == "lm_head.weight") and \
                        k.endswith(".weight") and t[k].ndim == 2 and t[k].shape[-1] % 64 == 0 and "norm" not in k:
                    want.add(k)
                if k in want:
                    assert t[k].shape[-1] % a.lm_group == 0, (k, t[k].shape)
                    b = a.embed_bits if (a.embed_bits and ("embed_tokens" in k or k == "lm_head.weight")) else lm_bits
                    exact = a.lm_exact or b != 4 or a.lm_group != 64     # new settings: exact MLX rounding
                    t[k] = (fake_quant_mlx if exact else fake_quant)(t[k], b, a.lm_group); done.add(k)
                elif a.vision_bits and is_vision_linear(k, t[k]):
                    t[k] = fake_quant_mlx(t[k], a.vision_bits, 64); vis.add(k)
            save_file(t, str(out / f.name), metadata={"format": "pt"})
            print(f.name, len(done), len(vis), flush=True)
        elif f.is_file():
            shutil.copy(f, out / f.name)
    missing = want - done - {"lm_head.weight"}    # tied embeddings (4B): lm_head IS embed_tokens, quantized above
    print("quantized", len(done), "of", len(want), "missing", sorted(want - done)[:5], "| vision linears", len(vis),
          f"| lm {lm_bits}b g{a.lm_group} embed {a.embed_bits or lm_bits}b vision {a.vision_bits}", flush=True)
    assert not missing
    json.dump({"lm_exact": a.lm_exact, "lm_bits": lm_bits, "lm_group": a.lm_group, "embed_bits": a.embed_bits or lm_bits,
               "vision_bits": a.vision_bits, "vision_tensors": sorted(vis)}, open(out / "sim_quant.json", "w"), indent=0)


if __name__ == "__main__":
    main()
