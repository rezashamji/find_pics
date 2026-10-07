"""Follow-up to eval/coreml_preproc_parity.py: per-channel int8 image weights alone cost ~0.995 mean cosine and
overlap@600 ~0.96. Would a finer int8 (coremltools granularity="per_block", block_size B along the input axis; needs
iOS 18) keep the ~90 MB size without that loss? Simulated in PyTorch, server preprocessing throughout.
Usage (GPU job): python eval/coreml_int8_blocks.py -> eval/coreml_int8_blocks.json"""
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "eval")
from coreml_preproc_parity import QUERIES, dev, enc, int8_copy, load_image, model, pre, selfcheck_images, tok  # noqa


def int8_block(m, B):
    q = copy.deepcopy(m)
    with torch.no_grad():
        for _, p in q.visual.named_parameters():
            if p.dim() >= 2 and p.numel() > 2048:
                w = p.data.float(); co, ci = w.shape[0], w.shape[1]
                if ci % B:
                    s = w.abs().flatten(1).amax(1).clamp_min(1e-12).view(-1, *[1] * (w.dim() - 1)) / 127
                    p.data.copy_((w / s).round().clamp(-127, 127) * s); continue
                wb = w.reshape(co, ci // B, B, *w.shape[2:])
                s = wb.abs().flatten(2).amax(2).clamp_min(1e-12).view(co, ci // B, 1, *[1] * (w.dim() - 2)) / 127
                p.data.copy_(((wb / s).round().clamp(-127, 127) * s).reshape(w.shape))
    return q


sc, ref = selfcheck_images()
paths = random.Random(1).sample(sorted(Path("data/public/raw/openimages/images").glob("*.jpg")), 2000)
X = [pre(load_image(str(p))) for p in paths]
Xs = [pre(sc[n]) for n in sc]
with torch.no_grad():
    T = F.normalize(model.encode_text(tok(QUERIES).to(dev)).float(), dim=-1).cpu()
base, bs = enc(model, X), enc(model, Xs)
out = {}
for name, m in [("per_channel", int8_copy(model)), ("per_block32", int8_block(model, 32)), ("per_block64", int8_block(model, 64)),
                ("per_block128", int8_block(model, 128))]:
    V, Vs = enc(m, X), enc(m, Xs)
    c = (V * base).sum(1).numpy()
    row = dict(selfcheck={n: round(float(Vs[i] @ bs[i]), 4) for i, n in enumerate(sc)},
               openimages2000_cos=dict(mean=round(float(c.mean()), 4), p10=round(float(np.percentile(c, 10)), 4), min=round(float(c.min()), 4)))
    for k in (600, 100, 20):
        ov = [len(set(torch.topk(base @ T[i], k).indices.tolist()) & set(torch.topk(V @ T[i], k).indices.tolist())) / k for i in range(len(QUERIES))]
        row[f"overlap@{k}"] = dict(mean=round(float(np.mean(ov)), 3), min=round(float(np.min(ov)), 3), n_queries=len(ov))
    out[name] = row; print(name, row, flush=True)
json.dump(out, open("eval/coreml_int8_blocks.json", "w"), indent=1)
print("INT8_BLOCKS_OK", flush=True)
