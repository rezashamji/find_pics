"""RESULTS 58 step 3: train the PHONE JUDGE. LoRA on BOTH the vision tower and the language model of Qwen3-VL-4B.

Base = models/q3vl4b_real: the phone's own language weights (the real mlx-community 4-bit tensors de-quantized) + the
bf16 vision tower, so the adapter learns a correction ON the weights the phone runs. Inputs = the exact judge images
(eval/judge_teach/label.py pack: 896-px thumbnail, JPEG q88; person rows with the red box) and the product prompt
"<question> Answer with one word: yes or no." through the model's chat template; read-out = next-token logits at the
answer position, log P(yes | yes or no) with yes = {yes, Yes, " yes", " Yes"}, no likewise (= VLLMJudge's P).
Loss: soft binary cross-entropy against the teacher's P(yes) (+ optional --gap-w x Huber on the logit gap vs the
teacher's clipped logit, to carry the far tail the 0.999 / 0.9999 cuts read). Pairs where the phone judge and the
teacher disagree at 0.5 are repeated --hard times (hard negatives = phone yes / teacher no; hard positives the reverse).
LoRA: language q/k/v/o/gate/up/down of all 36 layers (rank --r), vision blocks qkv/proj/fc1/fc2 + merger + deepstack
mergers (rank --vr). Validation = 5% of QUESTIONS held out (pairs.split == 'val'), against the TEACHER only (eye labels
are never seen here); the best checkpoint by val loss is kept.
Usage: torchrun --nproc_per_node=G eval/judge_teach/train.py --teacher=<labels name> --out=models/<dir> [...]
"""
import io
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/public/judge_teach"
LM_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIS_RE = (r"model\.visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))|merger\.linear_fc(1|2)|"
          r"deepstack_merger_list\.\d+\.linear_fc(1|2))")


def arg(k, d=None, t=str):
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    return None if v is None else t(v)


class Packs:
    def __init__(self, parts):
        self.ix, self.buf = {}, {}
        for k in parts:
            t = pd.read_parquet(D / "img" / f"part{k}.parquet")
            self.buf[k] = (D / "img" / f"part{k}.bin").read_bytes()
            for key, o, n in zip(t.img_key, t.off, t.len):
                self.ix[key] = (k, int(o), int(n))

    def get(self, key):
        k, o, n = self.ix[key]
        return Image.open(io.BytesIO(self.buf[k][o:o + n])).convert("RGB") if n else None


def load_rows(teacher, phone, K):
    P = pd.read_parquet(D / "pairs.parquet")
    P["pair"] = np.arange(len(P))
    t = pd.concat([pd.read_parquet(f) for f in sorted((D / "labels" / teacher).glob("part*.parquet"))]).set_index("pair").p
    P["pt"] = t.reindex(P.pair).to_numpy()
    if phone and (D / "labels" / phone).exists():
        ph = pd.concat([pd.read_parquet(f) for f in sorted((D / "labels" / phone).glob("part*.parquet"))]).set_index("pair").p
        P["pp"] = ph.reindex(P.pair).to_numpy()
    else:
        P["pp"] = np.nan
    P["part"] = P.img_key.map(lambda s: zlib.crc32(s.encode()) % K)
    return P[~P.pt.isna()].reset_index(drop=True)


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    lr_ = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr_)
    base = arg("base", str(ROOT / "models/q3vl4b_real"))
    out = Path(arg("out"))
    K = len(list((D / "img").glob("part*.parquet")))
    epochs = arg("epochs", 1.0, float)
    bs, accum = arg("bs", 8, int), arg("accum", 2, int)
    hard = arg("hard", 3, int)
    gap_w = arg("gap-w", 0.0, float)
    lr = arg("lr", 1e-4, float)
    P = load_rows(arg("teacher"), arg("phone", "phone_v8"), K)
    mine = sorted(k for k in range(K) if k % world == rank)
    P = P[P.part.isin(mine)].reset_index(drop=True)
    packs = Packs(mine)
    P = P[P.img_key.map(lambda k: k in packs.ix and packs.ix[k][2] > 0)].reset_index(drop=True)
    tr = P[P.split == "train"]
    va = P[P.split == "val"].sample(n=min(arg("nval", 1500, int), (P.split == "val").sum()), random_state=0)
    dis = (tr.pp >= 0.5) != (tr.pt >= 0.5)
    dis &= ~tr.pp.isna()
    tr = pd.concat([tr] + [tr[dis]] * (hard - 1), ignore_index=True)
    n_local = torch.tensor([len(tr)], device="cuda")
    if ddp:
        dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
    n_per_epoch = int(n_local.item()) // bs * bs
    steps_total = int(math.ceil(n_per_epoch * epochs / bs / accum))
    if rank == 0:
        print(f"world {world}, rows/rank {len(tr)} (disagreements x{hard}: {int(dis.sum())} base rows), "
              f"val {len(va)}, opt steps {steps_total}", flush=True)
    proc = AutoProcessor.from_pretrained(base)
    proc.tokenizer.padding_side = "left"
    tok = proc.tokenizer
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    r, vr = arg("r", 16, int), arg("vr", 16, int)
    targets = []
    if r > 0:
        targets.append(LM_RE)
    if vr > 0:
        targets.append(VIS_RE)
    pattern = "(" + "|".join(targets) + ")"
    rank_pattern = {r"model\.visual\..*": vr} if vr > 0 else {}
    alpha_pattern = {r"model\.visual\..*": 2 * vr} if vr > 0 else {}
    cfg = LoraConfig(r=r if r > 0 else vr, lora_alpha=2 * (r if r > 0 else vr), lora_dropout=0.05, target_modules=pattern,
                     rank_pattern=rank_pattern, alpha_pattern=alpha_pattern)
    model = get_peft_model(model, cfg)
    if rank == 0:
        model.print_trainable_parameters()
        n_vis = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ".visual." in n)
        print("trainable vision params", n_vis, flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_]) if ddp else model
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(10, steps_total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(steps_total, 1)))

    def batch_inputs(g):
        ims = [packs.get(k) for k in g.img_key]
        texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {
            "type": "text", "text": q + " Answer with one word: yes or no."}]}], tokenize=False, add_generation_prompt=True)
            for q in g.q]
        return proc(text=texts, images=ims, return_tensors="pt", padding=True).to("cuda")

    def forward(m, g):
        inp = batch_inputs(g)
        logits = m(**inp, logits_to_keep=1).logits[:, -1].float()
        return torch.logsumexp(logits[:, yes], 1) - torch.logsumexp(logits[:, no], 1)     # logit gap

    def loss_of(gap, t):
        l = -(t * F.logsigmoid(gap) + (1 - t) * F.logsigmoid(-gap))
        if gap_w > 0:
            tt = torch.logit(t.clamp(1e-6, 1 - 1e-6)).clamp(-12, 12)
            l = l + gap_w * F.huber_loss(gap.clamp(-14, 14), tt, reduction="none", delta=1.0)
        return l.mean()

    def validate():
        model.eval(); ls, gs = [], []
        with torch.no_grad():
            for s in range(0, len(va), 16):
                g = va.iloc[s:s + 16]
                gap = forward(model, g)
                t = torch.tensor(g.pt.to_numpy(), device="cuda", dtype=torch.float32)
                ls.append(float(loss_of(gap, t)) * len(g)); gs.append(gap.cpu().numpy())
        model.train()
        gap = np.concatenate(gs); pt = va.pt.to_numpy()
        agree = float(((gap > 0) == (pt >= 0.5)).mean())
        return sum(ls) / len(va), agree, gap

    step, t0, run, best = 0, time.time(), 0.0, (1e9, -1)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        v, a, gap0 = validate()
        print(f"val before training: loss {v:.4f} agree@0.5 {a:.3f}", flush=True)
        np.save(out / "val_gap_step0.npy", gap0)
    model.train()
    val_every = arg("val-every", 250, int)
    done = False
    ep = 0
    while not done:
        order = tr.sample(frac=1.0, random_state=1000 * ep + rank).iloc[:n_per_epoch]
        for i in range(0, n_per_epoch, bs):
            g = order.iloc[i:i + bs]
            gap = forward(net, g)
            t = torch.tensor(g.pt.to_numpy(), device="cuda", dtype=torch.float32)
            loss = loss_of(gap, t) / accum
            loss.backward(); run += float(loss) * accum
            if (i // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if rank == 0 and step % 25 == 0:
                    print(f"ep {ep} step {step}/{steps_total} loss {run / (25 * accum):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True); run = 0.0
                if step % val_every == 0 or step >= steps_total:
                    if rank == 0:
                        v, a, _ = validate()
                        tag = ""
                        if v < best[0]:
                            best = (v, step); model.save_pretrained(out / "best"); tag = " (best, saved)"
                        model.save_pretrained(out / "last")
                        print(f"VAL step {step}: loss {v:.4f} agree@0.5 {a:.3f}{tag}", flush=True)
                        json.dump({"best_step": best[1], "best_val": best[0], "step": step}, open(out / "state.json", "w"))
                    if ddp:
                        dist.barrier()
                if step >= steps_total:
                    done = True; break
        ep += 1
    if rank == 0:
        print("TRAIN_DONE best", best, f"{time.time() - t0:.0f}s", flush=True)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
