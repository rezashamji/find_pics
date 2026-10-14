import sys; sys.path.insert(0, "eval/person_first"); sys.path.insert(0, "eval")
import numpy as np, pandas as pd
import r62, tune_stop_time as TT
from findpics import person_first as PF
R = pd.read_parquet("eval/person_first/facts_runs.parquet")
x = R[R.over > 0]
print("runs with any overclaim:", len(x), "of", len(R), "| by order", x.groupby("order").size().to_dict(), "| by set", x.groupby("set").size().to_dict())
print("runs per order:", R.groupby("order").size().to_dict())
D = r62.people()
S = {(a, b, c): g for a, b, c, g in r62.fact_searches(D)}
_, r = TT.parse(r62.SHIPPED)
rows = []
for t in x.itertuples():
    g = S[(t.set, t.pool, t.cond)].reset_index(drop=True)
    s = g.s.to_numpy(); h = g.h.to_numpy()
    o = PF.fact_order(s) if t.order == "COS" else PF.fact_order(s, h) if t.order == "BLEND" else np.random.default_rng(1000).permutation(len(g))
    y = (g.p.to_numpy() >= 0.7)[o]; Y = int(y.sum())
    _, rx = TT.sim_timed(y, r, t.seed, exhaustive=True)
    K = len(rx)
    for (nc, cert, k, last) in rx:
        miss = Y - cert["found"]
        if miss > cert["missed_upper"] + 1e-9:
            rows.append(dict(set=t.set, cond=t.cond, order=t.order, seed=t.seed, n=len(y), Y=Y, k=k, calls=nc, n_tail=cert["n_tail"],
                             sampled=cert["tail_sampled"], hits=cert["tail_hits"], upper=round(cert["missed_upper"], 1), missed=miss,
                             alpha=TT.alpha_k(k, TT.max_rounds(len(y), r), r["a1"])))
O = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(O.to_string())
print(O.groupby("k").size())
