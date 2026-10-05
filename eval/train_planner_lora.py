"""Distill the 9B planner into Qwen3.5-4B with a LoRA adapter (small add-on weights; the base 4B stays the judge).
Input: data/public/distill/planner_part*.jsonl (eval/distill_planner_data.py), split == "train" only.
Loss only on the answer tokens (the plan JSON), prompt formatted with the same chat template vLLM uses
(enable_thinking=False). Output: models/planner_4b_lora/ (adapter) and models/planner_4b_merged/ (base + adapter merged,
for vLLM evaluation and for MLX conversion to the phone).
Usage (vLLM env, one A100): python eval/train_planner_lora.py [--epochs 2]
"""
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

BASE = "Qwen/Qwen3.5-4B"
OUT = Path("models/planner_4b_lora")
MERGED = Path("models/planner_4b_merged")


def load_rows():
    rows = [json.loads(l) for f in sorted(glob.glob("data/public/distill/planner_part*.jsonl")) for l in open(f)]
    return [r for r in rows if r["split"] == "train" and "{" in r["output"]]


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 2
    tok = AutoTokenizer.from_pretrained(BASE)
    rows = load_rows()
    print("train rows", len(rows), flush=True)
    data = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(r["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (pi + ai)[:6144]
        lab = ([-100] * len(pi) + ai)[:6144]
        data.append((ids, lab))
    # the full vision-language model (so the merged planner loads in vLLM like the base); LoRA only on the text layers
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|"
                                    r"out_proj)|mlp\.(gate|up|down)_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    accum = 8
    total = math.ceil(len(data) * epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(total, 1)))
    model.train(); step = 0; t0 = time.time()
    for ep in range(epochs):
        random.Random(ep).shuffle(data)
        run = 0.0
        for i, (ids, lab) in enumerate(data):
            x = torch.tensor([ids], device="cuda"); y = torch.tensor([lab], device="cuda")
            loss = model(input_ids=x, labels=y).loss / accum
            loss.backward(); run += loss.item()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"epoch {ep} step {step}/{total} loss {run / 20:.4f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    merged = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED); tok.save_pretrained(MERGED)
    # keep the base model's processor / chat template files so vLLM serves the merged model the same way
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    for f in src.iterdir():
        if f.suffix in (".json", ".jinja", ".txt") and not (MERGED / f.name).exists():
            (MERGED / f.name).write_bytes(f.read_bytes())
    print("SAVED", OUT, MERGED, flush=True)


if __name__ == "__main__":
    main()
