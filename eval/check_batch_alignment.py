"""Is P(yes) attached to the right image? Re-judge suspicious oracle 'yes' cases (cars judged 'bicycle' at p~0.99)
one image per call vs in a batch, plus true bicycles as controls. If single-image p differs from the stored batch p,
batching misaligns answers (a bug affecting every judged result)."""
import numpy as np, pandas as pd
from findpics import store
from findpics.engine import _frame_for, _judge_rows
from findpics.vlm import VLLMJudge
idx = store.load("data/public/index_testlib")
d = pd.read_parquet("eval/oracle/bicycle.parquet"); P = d.p.to_numpy(); L = d.look.to_numpy()
rank = np.empty(len(L), int); rank[np.argsort(-L)] = np.arange(len(L))
sus = [i for i in np.argsort(rank) if P[i] >= 0.9 and rank[i] > 1200][:12]     # low cheap rank, high judge yes
ctrl = [i for i in np.argsort(rank) if P[i] >= 0.9][:6]                         # top-ranked yes (true bikes)
neg = [i for i in np.argsort(-rank) if P[i] < 0.1][:6]
rows = np.array(sus + ctrl + neg)
J = VLLMJudge(gpu_mem=0.7); q = "Is there a bicycle visible in this image?"
single = [J.p_yes([_frame_for(idx, int(r), -1)], q)[0] for r in rows]
batch = _judge_rows(idx, J, rows, np.full(len(rows), -1), q, batch=96)
rng = np.random.default_rng(0); perm = rng.permutation(len(rows))
batch_shuf = np.empty(len(rows)); batch_shuf[perm] = _judge_rows(idx, J, rows[perm], np.full(len(rows), -1), q, batch=96)
for r, s, b, bs in zip(rows, single, batch, batch_shuf):
    print(f"row {r:6d} rank {rank[r]:6d} stored {P[r]:.2f} single {s:.2f} batch {b:.2f} batch_shuffled {bs:.2f} {idx.items.item_id.iloc[r]}")
