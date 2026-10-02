"""Query by example vs by text (fast stage only, no judge). For each Open Images concept: pick 3 verified positives as
'examples', remove them from the evaluation, and measure recall of the remaining verified positives in the top-K, for
(a) text "a photo of <c>", (b) mean similarity to the 3 example images, (c) both combined.
Usage: python eval/eval_by_example.py   (CPU is fine: only a few text encodings)"""
import json
import numpy as np
from findpics import store
from findpics.models import ImageTextEncoder
from findpics.store import per_item_max

idx = store.load("data/public/index_testlib")
gt = json.load(open("data/public/testlib/ground_truth.json"))
row = {u: i for i, u in enumerate(idx.items.item_id)}
enc = ImageTextEncoder(idx.clip_model, device="cpu")
C = idx.clip.astype(np.float32); ur = idx.units.item_row.to_numpy()
item_vec = np.zeros((idx.n_items, C.shape[1]), np.float32); np.add.at(item_vec, ur, C)
item_vec /= np.linalg.norm(item_vec, axis=1, keepdims=True) + 1e-9
rng = np.random.default_rng(0); out = {}
print(f"{'concept':16s} {'pos':>4s} {'K':>5s}  text  example  both")
for key, v in gt.items():
    if not key.startswith("concept:"):
        continue
    c = key.split(":", 1)[1]; pos = np.array([row[u] for u in v["pos"] if u in row])
    if len(pos) < 15:
        continue
    ex = rng.choice(pos, 3, replace=False); tgt = np.setdiff1d(pos, ex); K = 2 * len(tgt)
    t = per_item_max((C @ enc.texts([f"a photo of {c.lower()}"]).T)[:, 0], ur, idx.n_items)
    e = item_vec @ item_vec[ex].mean(0)
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    res = {}
    for name, s in (("text", t), ("example", e), ("both", z(t) + z(e))):
        s = s.copy(); s[ex] = -np.inf
        top = np.argsort(-s)[:K]; res[name] = float(np.isin(tgt, top).mean())
    out[c] = dict(n_targets=int(len(tgt)), K=int(K), **res)
    print(f"{c:16s} {len(tgt):4d} {K:5d}  {res['text']:.2f}  {res['example']:.2f}     {res['both']:.2f}")
m = {k: float(np.mean([r[k] for r in out.values()])) for k in ("text", "example", "both")}
print("mean recall@2x:", {k: round(v, 3) for k, v in m.items()})
json.dump(dict(per_concept=out, mean=m), open("eval/results_by_example.json", "w"), indent=1)
