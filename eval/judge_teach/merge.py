"""RESULTS 58 step 3b: turn a trained adapter into the checkpoints the evaluation serves (all HF format for vLLM).

  <out>_v8      PHONE FORM. Vision: LoRA merged into the bf16 tower in fp32, then MLX affine 8-bit (exact rounding of
                scripts/sim_mlx_quant.fake_quant_mlx, group 64; the shipped side-loaded tower). Language: the phone's
                de-quantized 4-bit weights UNCHANGED; the language LoRA is saved separately as <out>_lm_lora (PEFT) and
                applied at serve time by vLLM's LoRA path = what the phone's runtime adapter computes (no re-rounding of
                W + BA into bf16 weights).
  <out>_v8m     same vision; language W + BA merged in fp32 and stored bf16 (reported: how much bf16 storage rounds).
  <out>_v8rq4   same vision; language W + BA re-quantized to MLX 4-bit (exact rounding) = "merge, then ship a new 4-bit
                checkpoint"; reported, not deciding.
Prints how much of the LoRA delta survives each rounding (|Q(W + d) - W| / |d| aggregated).
Usage (GPU): python eval/judge_teach/merge.py <adapter dir> <out prefix>   [--base=models/q3vl4b_real]
"""
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from sim_mlx_quant import fake_quant_mlx, is_vision_linear  # noqa: E402


def main():
    ad, out = Path(sys.argv[1]), sys.argv[2]
    base = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--base=")), str(ROOT / "models/q3vl4b_real")))
    cfg = json.load(open(ad / "adapter_config.json"))
    A = load_file(str(ad / "adapter_model.safetensors"))
    # PEFT keys: base_model.model.<module>.lora_A.weight / lora_B.weight
    mods = {}
    for k, v in A.items():
        m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
        mods.setdefault(m.group(1), {})[m.group(2)] = v
    rp, ap = cfg.get("rank_pattern") or {}, cfg.get("alpha_pattern") or {}

    def scale(name):
        r, a = cfg["r"], cfg["lora_alpha"]
        for pat, v in rp.items():
            if re.match(rf"(.*\.)?({pat})$", name):
                r = v
        for pat, v in ap.items():
            if re.match(rf"(.*\.)?({pat})$", name):
                a = v
        return a / r
    vis = {m: d for m, d in mods.items() if m.startswith("model.visual.")}
    lm = {m: d for m, d in mods.items() if not m.startswith("model.visual.")}
    print(f"adapter: {len(vis)} vision modules, {len(lm)} language modules", flush=True)
    outs = {s: Path(f"{out}_{s}") for s in ("v8", "v8m", "v8rq4")}
    for p in outs.values():
        p.mkdir(parents=True, exist_ok=True)
    stats = {s: [0.0, 0.0] for s in ("vision8", "lm_bf16", "lm_q4")}
    for f in sorted(base.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file() and f.name != "sim_quant.json":
                for p in outs.values():
                    shutil.copy(f, p / f.name)
            continue
        t = load_file(str(f))
        tv8, tm, trq = {}, {}, {}
        for k, w in t.items():
            mname = k[:-len(".weight")] if k.endswith(".weight") else None
            d = None
            if mname in mods:
                ab = mods[mname]
                d = (ab["B"].float().cuda() @ ab["A"].float().cuda()) * scale(mname)
            if k.startswith("model.visual."):
                wv = w.float().cuda() + (d if d is not None else 0)
                if is_vision_linear(k, w):
                    q = fake_quant_mlx(wv, 8, 64).float().cuda()
                    if d is not None:
                        stats["vision8"][0] += float((q - w.float().cuda()).flatten() @ d.flatten())
                        stats["vision8"][1] += float(d.flatten() @ d.flatten())
                    wv = q
                wv = wv.to(w.dtype).cpu()
                tv8[k] = tm[k] = trq[k] = wv
            elif d is not None:
                w32 = w.float().cuda()
                m_ = (w32 + d).to(torch.bfloat16)
                rq = fake_quant_mlx(w32 + d, 4, 64).to(torch.bfloat16)
                for s, x in (("lm_bf16", m_), ("lm_q4", rq)):
                    stats[s][0] += float((x.float().cuda() - w32).flatten() @ d.flatten())
                    stats[s][1] += float(d.flatten() @ d.flatten())
                tv8[k] = w; tm[k] = m_.cpu(); trq[k] = rq.cpu()
            else:
                tv8[k] = tm[k] = trq[k] = w
        for s, tt in (("v8", tv8), ("v8m", tm), ("v8rq4", trq)):
            save_file(tt, str(outs[s] / f.name), metadata={"format": "pt"})
        print(f.name, flush=True)
    # the language-only LoRA for vLLM's runtime path
    lo = Path(f"{out}_lm_lora"); lo.mkdir(parents=True, exist_ok=True)
    save_file({k: v for k, v in A.items() if ".visual." not in k}, str(lo / "adapter_model.safetensors"))
    c2 = dict(cfg); c2["rank_pattern"] = {}; c2["alpha_pattern"] = {}
    c2["target_modules"] = sorted({m.split(".")[-1] for m in lm})
    c2["base_model_name_or_path"] = str(outs["v8"])
    json.dump(c2, open(lo / "adapter_config.json", "w"), indent=1)
    # projection of the realised change onto the intended delta: 1.0 = delta kept exactly, 0 = wiped out
    rep = {s: (a / b if b else None) for s, (a, b) in stats.items()}
    json.dump({"adapter": str(ad), "delta_kept": rep}, open(f"{out}_merge.json", "w"), indent=1)
    print("DELTA KEPT (projection on the intended LoRA delta):", rep, flush=True)


if __name__ == "__main__":
    main()
