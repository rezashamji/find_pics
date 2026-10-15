"""RESULTS 61d (2): a facts-writer student -> the MLX checkpoint the app loads (mlx-swift-lm MLXVLM Qwen3VL, same layout
and key names as mlx-community/Qwen3-VL-2B-Instruct-4bit: one model.safetensors + its config / tokenizer files).

Why MERGED 4-bit weights and not a LoRA adapter: (a) the student has LoRA on the VISION tower too, and mlx-swift-lm's
Qwen3VL exposes only the language model's layers to LoRA (Libraries/MLXVLM/Models/Qwen3VL.swift `loraLayers` =
languageModel.model.layers, 3.32.3), so a vision adapter cannot be loaded; (b) the 2B base is not otherwise on the phone
(the judge is Qwen3-VL-4B), so an adapter saves nothing; (c) the cluster measured EXACTLY this form: merge_q4.py merges in
fp32 and re-quantizes every language Linear / Embedding with the exact MLX rounding (vision bf16). This script does the
same merge and writes the integer codes + bf16 scales / biases instead of their de-quantized product.

  python eval/facts_writer/to_mlx.py <base dir> <adapter dir | none> <out dir> [--ref=<mlx-community snapshot>] [--check=<q4 dir>]
--check: de-quantize what was written and compare with the vLLM phone-form checkpoint (merge_q4.py output).
--ref: every key / shape / dtype must match the real mlx-community file (also used for the patch-embed layout).
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def quant_mlx(w: torch.Tensor, bits=4, group=64):
    """MLX affine_quantize (same arithmetic as scripts/sim_mlx_quant.fake_quant_mlx) -> (packed uint32 codes,
    bf16 scales, bf16 biases), shapes [out, in*bits/32], [out, in/group], [out, in/group]."""
    out_, inn = w.shape
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
    q = ((g - bias) / scale).round().clamp(0, n).to(torch.int64).reshape(out_, inn // 8, 8)
    packed = torch.zeros(out_, inn // 8, dtype=torch.int64, device=q.device)
    for k in range(8):
        packed |= q[..., k] << (4 * k)
    packed = packed.cpu().to(torch.uint32)
    return (packed, scale.reshape(out_, inn // group).to(torch.bfloat16).cpu(),
            bias.reshape(out_, inn // group).to(torch.bfloat16).cpu())


def dequant(packed, s, b, group=64):
    wq = packed.view(torch.int32) if packed.dtype != torch.int32 else packed
    out_, words = wq.shape; inn = words * 8
    codes = torch.stack([(wq >> (4 * k)) & 15 for k in range(8)], -1).reshape(out_, inn).float()
    return (codes.reshape(out_, inn // group, group) * s.float()[..., None] + b.float()[..., None]).reshape(out_, inn)


def mlx_key(k: str) -> str:
    if k.startswith("model.language_model."):
        return "language_model.model." + k[len("model.language_model."):]
    if k.startswith("model.visual."):
        return "vision_tower." + k[len("model.visual."):]
    if k == "lm_head.weight":
        return "language_model.lm_head.weight"
    raise KeyError(k)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    arg = lambda k: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), None)
    base, ad, out = Path(pos[0]), pos[1], Path(pos[2])
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32)
    if ad != "none":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ad).merge_and_unload()
    sd = m.state_dict()
    tied = bool(getattr(m.config, "tie_word_embeddings", False) or getattr(m.config.get_text_config(), "tie_word_embeddings", False))
    ref = {}
    if arg("ref"):
        with safe_open(str(Path(arg("ref")) / "model.safetensors"), "pt") as f:
            ref = {k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()) for k in f.keys()}
    res = {}
    for k, v in sd.items():
        if k == "lm_head.weight" and tied:
            continue
        mk = mlx_key(k)
        if (k.startswith("model.language_model.") or k == "lm_head.weight") and k.endswith(".weight") and v.ndim == 2 \
                and v.shape[-1] % 64 == 0 and "norm" not in k:
            p, s, b = quant_mlx(v)
            stem = mk[:-len(".weight")]
            res[mk], res[stem + ".scales"], res[stem + ".biases"] = p, s, b
        else:
            t = v.to(torch.bfloat16)
            if "patch_embed.proj.weight" in k and t.ndim == 5:     # torch (out, in, t, h, w) -> MLX (out, t, h, w, in)
                t = t.permute(0, 2, 3, 4, 1).contiguous()
            res[mk] = t.contiguous()
    if ref:
        miss, extra = sorted(set(ref) - set(res)), sorted(set(res) - set(ref))
        bad = [(k, tuple(res[k].shape), ref[k][0]) for k in set(res) & set(ref) if tuple(res[k].shape) != ref[k][0]]
        dts = sorted({(str(res[k].dtype), ref[k][1]) for k in set(res) & set(ref)})
        print("REF missing", len(miss), miss[:5], "extra", len(extra), extra[:5], "shape mismatches", len(bad), bad[:5],
              "dtype pairs", dts, flush=True)
        assert not miss and not extra and not bad
    out.mkdir(parents=True, exist_ok=True)
    save_file(res, str(out / "model.safetensors"), metadata={"format": "mlx"})
    src_cfg = Path(arg("ref")) if arg("ref") else base
    for f in src_cfg.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    json.dump({"weight_map": {k: "model.safetensors" for k in sorted(res)}, "metadata": {}},
              open(out / "model.safetensors.index.json", "w"), indent=1)
    (out / "SOURCE.txt").write_text(f"base {base}\nadapter {ad}\nmerged fp32, language Linear/Embedding -> MLX affine 4-bit "
                                    f"g64 (eval/facts_writer/to_mlx.py), vision bf16\n")
    print("WROTE", out, len(res), "tensors", flush=True)
    if arg("check"):
        q4 = load_file(str(Path(arg("check")) / "model.safetensors")) if (Path(arg("check")) / "model.safetensors").exists() else {}
        if not q4:
            for fn in sorted(Path(arg("check")).glob("*.safetensors")):
                q4.update(load_file(str(fn)))
        n_eq = n_all = 0; worst = 0.0; nv = 0
        for k, v in q4.items():
            mk = mlx_key(k) if not (k == "lm_head.weight" and tied) else None
            if mk is None or mk not in res:
                continue
            stem = mk[:-len(".weight")]
            if stem + ".scales" in res:
                w = dequant(res[mk], res[stem + ".scales"], res[stem + ".biases"]).to(torch.bfloat16).float()
            else:
                w = res[mk].float()
                if w.ndim == 5:
                    w = w.permute(0, 4, 1, 2, 3)
                nv += 1
            e = (w - v.float()).abs()
            n_eq += int((e == 0).sum()); n_all += e.numel(); worst = max(worst, float(e.max()))
        print(f"CHECK vs {arg('check')}: {n_eq:,}/{n_all:,} weights identical after bf16 cast ({n_eq / max(n_all, 1):.6f}); "
              f"max abs diff {worst:.3g}; {nv} unquantized tensors compared", flush=True)


if __name__ == "__main__":
    main()
