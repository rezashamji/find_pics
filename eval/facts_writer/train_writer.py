"""RESULTS 61 step 2: distil the 27B's SHORT facts line into a phone-size VLM. LoRA on the language model (attention /
linear-attention / MLP projections) AND the vision tower (blocks + mergers), on the phone's 4-bit-sim weights.
Input = the photo as the phone renders it (longest side 448 px, Lanczos, JPEG q80) + STUDENT_PROMPT; target = the 27B
short line (eval/facts_writer/short27b) + end of turn; loss = token cross-entropy on the target only.
TRAIN libraries of the given folds only (fold k = sorted TRAIN users [k::4]); 200 of their photos held out for val loss.
  python eval/facts_writer/train_writer.py --base=models/qwen3vl_2b_mlx4sim --folds=0,1 --out=eval/facts_writer/models/<name>
      [--epochs=1] [--bs=8] [--accum=2] [--lr=1e-4] [--r=16] [--vr=16] [--max_steps=0]
  -> <out>/best, <out>/last (PEFT adapters), train.log
"""
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from short_format import STUDENT_PROMPT  # noqa: E402

LM_RE = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj\w*|out_proj)|"
         r"mlp\.(gate|up|down)_proj)")
VIS_RE = (r"model\.visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc(1|2)|fc(1|2)))|merger\.(linear_fc(1|2)|mlp\.\d+)|"
          r"deepstack_merger_list\.\d+\.linear_fc(1|2))")


def arg(k, d=None, t=str):
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    return None if v is None else t(v)


def phone_rendition(path, max_side=448, q=80):
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def folds():
    tr = sorted(json.loads(Path("eval/complex_bench/splits.json").read_text())["train"])
    return {u: k for k in range(4) for u in tr[k::4]}


def load_targets(users, tdir=Path("eval/facts_writer/short27b")):
    rows = []
    for u in users:
        if not (tdir / f"{u}.jsonl").exists():
            print("MISSING targets", u, flush=True)
            continue
        for line in open(tdir / f"{u}.jsonl"):
            r = json.loads(line)
            t = (r.get("facts") or {}).get("caption", "")
            if t:
                rows.append((u, str(r["item_id"]), t))
    return pd.DataFrame(rows, columns=["user", "item_id", "target"])


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    base, out = arg("base"), Path(arg("out"))
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(*a):
        s = " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()
    want = {int(x) for x in arg("folds").split(",")}
    users = [u for u, k in folds().items() if k in want]
    T = load_targets(users)
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(it.item_id.astype(str), it.path))
    T = T[T.item_id.isin(path)].reset_index(drop=True)
    rng = np.random.RandomState(0)
    va_ix = rng.choice(len(T), 200, replace=False)
    va, tr = T.iloc[va_ix], T.drop(index=va_ix).reset_index(drop=True)
    epochs, bs, accum, lr = arg("epochs", 1.0, float), arg("bs", 8, int), arg("accum", 2, int), arg("lr", 1e-4, float)
    steps_total = int(math.ceil(len(tr) * epochs / bs / accum))
    if arg("max_steps", 0, int):
        steps_total = min(steps_total, arg("max_steps", 0, int))
    say("config", dict(base=base, folds=sorted(want), users=len(users), train=len(tr), val=len(va), epochs=epochs, bs=bs,
                       accum=accum, lr=lr, steps=steps_total))
    proc = AutoProcessor.from_pretrained(base)
    tok = proc.tokenizer
    tok.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    r, vr = arg("r", 16, int), arg("vr", 16, int)
    pattern = "(" + "|".join([LM_RE] + ([VIS_RE] if vr > 0 else [])) + ")"
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=pattern,
                     rank_pattern={r"model\.visual\..*": vr} if vr > 0 else {},
                     alpha_pattern={r"model\.visual\..*": 2 * vr} if vr > 0 else {})
    model = get_peft_model(model, cfg)
    n_lm = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and ".visual." not in n)
    n_vis = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and ".visual." in n)
    say(f"LoRA modules: language {n_lm}, vision {n_vis}")
    model.print_trainable_parameters()
    end = "<|im_end|>"

    def texts(target=None):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": STUDENT_PROMPT}]}]
        p = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return p if target is None else (p, p + target + end + "\n")

    def batch(g):
        ims = [phone_rendition(path[i]) for i in g.item_id]
        pr, full = zip(*[texts(t) for t in g.target])
        enc = proc(text=list(full), images=ims, return_tensors="pt", padding=True)
        enc_p = proc(text=list(pr), images=ims, return_tensors="pt", padding=True)
        plen = enc_p["attention_mask"].sum(1)
        labels = enc["input_ids"].clone()
        for j in range(len(g)):
            labels[j, :int(plen[j])] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return {k: v.cuda() for k, v in enc.items()}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(10, steps_total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(steps_total, 1)))

    def validate():
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), 8):
                g = va.iloc[s:s + 8]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l = model(**batch(g)).loss
                tot += float(l) * len(g); n += len(g)
        model.train()
        return tot / n
    t0 = time.time(); step = 0; run = 0.0; best = 1e9
    say(f"val before: {validate():.4f}")
    model.train()
    order = list(range(len(tr))) * int(math.ceil(epochs))
    random.Random(0).shuffle(order)
    i = 0
    while step < steps_total:
        if i + bs > len(order):
            more = list(range(len(tr))); random.Random(i).shuffle(more); order += more
        g = tr.iloc[order[i:i + bs]]; i += bs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch(g)).loss / accum
        loss.backward(); run += float(loss) * accum
        if (i // bs) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                say(f"step {step}/{steps_total} loss {run / (25 * accum):.4f} {time.time() - t0:.0f}s"); run = 0.0
            if step % 400 == 0 or step >= steps_total:
                v = validate()
                tag = ""
                if v < best:
                    best = v; model.save_pretrained(out / "best"); tag = " (best)"
                model.save_pretrained(out / "last")
                say(f"VAL step {step}: {v:.4f}{tag}")
    say(f"TRAIN_DONE best {best:.4f} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
