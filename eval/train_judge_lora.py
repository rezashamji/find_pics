"""Distill the 9B JUDGE into Qwen3.5-4B with a LoRA adapter: the 4B's P(yes) (yes/no next-token probabilities, the same
prompt and image preprocessing as VLLMJudge.p_yes: 896-px thumbnail, JPEG q88, "<question> Answer with one word: yes
or no.") is trained toward the 9B's P(yes) with a soft binary cross-entropy. Text layers only; the vision tower is
frozen. 10% of QUESTIONS are held out (whole questions, not photos) for the agreement test.
Input: data/public/judge_distill/part*.jsonl (eval/judge_distill_data.py). Output: models/judge_4b_lora/ and
models/judge_4b_merged/ (servable by vLLM, convertible to MLX).
Usage (vLLM env, one GPU): python eval/train_judge_lora.py [--epochs 1]
"""
import glob
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

BASE = "Qwen/Qwen3.5-4B"
OUT, MERGED = Path("models/judge_4b_lora"), Path("models/judge_4b_merged")


def split_of(q: str) -> str:
    return "test" if int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 == 0 else "train"


def judge_image(path: str) -> Image.Image:
    from findpics.media import load_image
    im = load_image(path).copy(); im.thumbnail((896, 896))      # == vlm._data_url
    b = io.BytesIO(); im.convert("RGB").save(b, format="JPEG", quality=88)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 1
    rows = [json.loads(l) for f in sorted(glob.glob("data/public/judge_distill/part*.jsonl")) for l in open(f)]
    rows = [r for r in rows if split_of(r["question"]) == "train"]
    # ~2 s per pair on the reference kernels: keep a few pairs per question (3 best-match + 1 random), many questions
    per_q = int(sys.argv[sys.argv.index("--per-question") + 1]) if "--per-question" in sys.argv else 4
    rnd = random.Random(0); by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], {"top": [], "rand": []})[r["kind"]].append(r)
    rows = [x for d in by_q.values() for x in rnd.sample(d["top"], min(per_q - 1, len(d["top"]))) +
            rnd.sample(d["rand"], min(1, len(d["rand"])))]
    if "--max" in sys.argv:
        rows = rows[:int(sys.argv[sys.argv.index("--max") + 1])]
    path = dict(pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
                .astype({"item_id": str}).itertuples(index=False, name=None))
    print("train pairs", len(rows), "questions", len({r["question"] for r in rows}), flush=True)
    proc = AutoProcessor.from_pretrained(BASE)
    tok = proc.tokenizer
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")})
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|"
                                    r"in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")
    model = get_peft_model(model, cfg); model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    accum = 16; total = math.ceil(len(rows) * epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(total, 1)))
    model.train(); step = 0; t0 = time.time(); run = 0.0
    for ep in range(epochs):
        random.Random(ep).shuffle(rows)
        for i, r in enumerate(rows):
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": r["question"] + " Answer with one word: yes or no."}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inp = proc(text=[text], images=[judge_image(path[r["item_id"]])], return_tensors="pt").to("cuda")
            logits = model(**inp).logits[0, -1].float()
            lse_y, lse_n = torch.logsumexp(logits[yes], 0), torch.logsumexp(logits[no], 0)
            logp_yes = lse_y - torch.logaddexp(lse_y, lse_n)          # log P(yes | yes or no)
            logp_no = lse_n - torch.logaddexp(lse_y, lse_n)
            t = float(r["p9"])
            loss = -(t * logp_yes + (1 - t) * logp_no) / accum
            loss.backward(); run += loss.item()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 25 == 0:
                    print(f"epoch {ep} step {step}/{total} loss {run / 25:.4f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    OUT.mkdir(parents=True, exist_ok=True); model.save_pretrained(OUT)
    merged = model.merge_and_unload(); MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED); proc.save_pretrained(MERGED)
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    for f in src.iterdir():     # base processor / chat template files, never the base shard index
        if f.suffix in (".json", ".jinja", ".txt") and not (MERGED / f.name).exists() and not f.name.endswith(".index.json"):
            (MERGED / f.name).write_bytes(f.read_bytes())
    print("SAVED", OUT, MERGED, flush=True)


if __name__ == "__main__":
    main()
