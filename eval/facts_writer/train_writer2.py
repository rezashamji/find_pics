"""RESULTS 61b: the short-facts student trained on TRAIN folds (cross-fit half) + the extra public photos labelled by the
27B (eval/facts_writer/extra_labels), several epochs, multi-GPU (torchrun DDP). Same model, LoRA, input and target as
train_writer.py (RESULTS 61); adds: --extra (use the extra photos), checkpoints at fixed PHOTOS-SEEN milestones for the
learning curve, validation loss on 200 held-out TRAIN photos AND 300 held-out extra photos.
  torchrun --nproc_per_node=G eval/facts_writer/train_writer2.py --base=<dir> --folds=0,1 --out=<dir> [--extra=1]
      [--epochs=2] [--bs=8] [--accum=1] [--lr=1e-4] [--save_at=100000,200000,400000]
  -> <out>/seen_<n>/ (PEFT adapter at n photos seen), <out>/final, train.log
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from short_format import STUDENT_PROMPT  # noqa: E402
from train_writer import LM_RE, VIS_RE, folds, load_targets, phone_rendition  # noqa: E402


def arg(k, d=None, t=str):
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    return None if v is None else t(v)


def rows(want_folds, extra):
    users = [u for u, k in folds().items() if k in want_folds]
    T = load_targets(users)
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(it.item_id.astype(str), it.path))
    T = T[T.item_id.isin(path)]
    T = pd.DataFrame(dict(src="train", key=T.item_id, path=T.item_id.map(path), target=T.target))
    if not extra:
        return T, None
    E = []
    for f in sorted(Path("eval/facts_writer/extra_labels").glob("part*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("short"):
                E.append((r["src"], r["id"], r["path"], r["short"]))
    E = pd.DataFrame(E, columns=["src", "key", "path", "target"])
    return T, E


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    lr_ = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(lr_)
    base, out = arg("base"), Path(arg("out"))
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a") if rank == 0 else None

    def say(*a):
        if rank == 0:
            s = " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()
    T, E = rows({int(x) for x in arg("folds").split(",")}, arg("extra", "1") == "1")
    rng = np.random.RandomState(0)
    vT = T.iloc[rng.choice(len(T), 200, replace=False)]
    T = T.drop(index=vT.index)
    vE = None
    if E is not None:
        vE = E.iloc[rng.choice(len(E), 300, replace=False)]
        E = E.drop(index=vE.index)
    tr = pd.concat([T] + ([E] if E is not None else []), ignore_index=True)
    epochs, bs, accum, lr = arg("epochs", 2.0, float), arg("bs", 8, int), arg("accum", 1, int), arg("lr", 1e-4, float)
    per_step = bs * accum * world
    steps_total = int(math.ceil(len(tr) * epochs / per_step))
    save_at = sorted({int(x) for x in arg("save_at", "100000,200000,400000").split(",") if x})
    say("config", dict(base=base, folds=arg("folds"), world=world, train_rows=len(tr), train_disbench=len(T),
                       extra=0 if E is None else len(E), val_train=len(vT), val_extra=0 if vE is None else len(vE),
                       epochs=epochs, photos_per_step=per_step, steps=steps_total, lr=lr, save_at=save_at))
    proc = AutoProcessor.from_pretrained(base)
    proc.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    r, vr = arg("r", 16, int), arg("vr", 16, int)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules="(" + LM_RE + "|" + VIS_RE + ")",
                     rank_pattern={r"model\.visual\..*": vr}, alpha_pattern={r"model\.visual\..*": 2 * vr})
    model = get_peft_model(model, cfg)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_]) if ddp else model
    end = "<|im_end|>"

    def texts(target):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": STUDENT_PROMPT}]}]
        p = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return p, p + target + end + "\n"

    def batch(g):
        ims = [phone_rendition(p) for p in g.path]
        pr, full = zip(*[texts(t) for t in g.target])
        enc = proc(text=list(full), images=ims, return_tensors="pt", padding=True)
        plen = proc(text=list(pr), images=ims, return_tensors="pt", padding=True)["attention_mask"].sum(1)
        lab = enc["input_ids"].clone()
        for j in range(len(g)):
            lab[j, :int(plen[j])] = -100
        lab[enc["attention_mask"] == 0] = -100
        enc["labels"] = lab
        return {k: v.cuda() for k, v in enc.items()}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(10, steps_total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(steps_total, 1)))

    def validate(V):
        model.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, len(V), 8):
                g = V.iloc[s:s + 8]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tot += float(model(**batch(g)).loss) * len(g)
        model.train()
        return tot / len(V)

    def val_line(step, seen):
        if rank != 0:
            return
        a = validate(vT); b = validate(vE) if vE is not None else float("nan")
        say(f"VAL step {step} seen {seen}: train-photos {a:.4f} extra-photos {b:.4f}")
    val_line(0, 0)
    order = []
    ep = 0
    while len(order) < steps_total * per_step + per_step:
        o = list(range(len(tr))); random.Random(ep).shuffle(o); order += o; ep += 1
    t0 = time.time(); run = 0.0; step = 0; nxt = 0
    model.train()
    while step < steps_total:
        base_i = step * per_step
        for a in range(accum):
            i0 = base_i + (a * world + rank) * bs
            g = tr.iloc[order[i0:i0 + bs]]
            sync = (a == accum - 1) or not ddp
            ctx = net.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net(**batch(g)).loss / accum
                loss.backward()
            run += float(loss) * accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        seen = step * per_step
        if step % 50 == 0:
            say(f"step {step}/{steps_total} seen {seen} loss {run / (50 * accum):.4f} {time.time() - t0:.0f}s")
            run = 0.0
        if nxt < len(save_at) and seen >= save_at[nxt]:
            if rank == 0:
                model.save_pretrained(out / f"seen_{save_at[nxt]}")
            val_line(step, seen); nxt += 1
            if ddp:
                dist.barrier()
    if rank == 0:
        model.save_pretrained(out / "final")
    val_line(step, step * per_step)
    say(f"TRAIN_DONE steps {step} seen {step * per_step} {time.time() - t0:.0f}s")
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
