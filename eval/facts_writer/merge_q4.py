"""RESULTS 61: a student adapter -> the PHONE FORM checkpoint vLLM can serve: LoRA merged into the 4-bit-sim base in
fp32, then every language-model Linear / Embedding (in-dim divisible by 64, not a norm; MLX's default set) re-quantized
to MLX affine 4-bit group 64 with the EXACT MLX rounding (scripts/sim_mlx_quant.fake_quant_mlx); vision tower bf16
(mlx-community keeps it unquantized). Prints how much of the LoRA delta survives the 4-bit rounding.
  python eval/facts_writer/merge_q4.py <base dir> <adapter dir> <out dir>
"""
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from sim_mlx_quant import fake_quant_mlx  # noqa: E402


def main():
    base, ad, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    m0 = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32)
    W0 = {k: v.detach().clone() for k, v in m0.state_dict().items()
          if (k.startswith("model.language_model.") or k == "lm_head.weight") and v.ndim == 2}
    m = PeftModel.from_pretrained(m0, ad).merge_and_unload()
    sd = m.state_dict()
    num = den = 0.0; n_q = 0
    with torch.no_grad():
        for k, v in sd.items():
            if (k.startswith("model.language_model.") or k == "lm_head.weight") and k.endswith(".weight") and \
                    v.ndim == 2 and v.shape[-1] % 64 == 0 and "norm" not in k:
                q = fake_quant_mlx(v, 4, 64)
                if k in W0:
                    d = v - W0[k]
                    if d.abs().max() > 0:
                        num += float(((q - W0[k]) * d).sum()); den += float((d * d).sum())
                v.copy_(q); n_q += 1
    print(f"re-quantized {n_q} language tensors to 4-bit; LoRA delta kept after rounding (projection) {num / max(den, 1e-12):.3f}",
          flush=True)
    m = m.to(torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out, safe_serialization=True)
    for f in base.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in ("model.safetensors.index.json", "config.json",
                                                                           "sim_quant.json"):
            shutil.copy(f, out / f.name)
    print("MERGED", out, flush=True)


if __name__ == "__main__":
    main()
