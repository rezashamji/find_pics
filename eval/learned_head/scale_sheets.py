"""M48-scale contact sheets at NATIVE pixels (DISBench ~500 px = what the judge sees). Public data; *.jpg git-ignored.
  boat_head / boat_look : EV16 "photos with a boat", one library: top 9 by the N10000 head alone vs by the phone's look
  car_blend_gain        : A4 "photos with a car": photos inside the judge's count by blend (N10000b) but NOT by look,
                          with their eye label (what the blend adds)
  teacher_q{k}          : 3 random generated questions: 9 photos each (from the 250 most similar), with the phone
                          judge's P, to check the teacher labels by eye
  python eval/learned_head/scale_sheets.py -> eval/learned_head/scale/sheets/*.jpg + index.tsv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/learned_head"); sys.path.insert(0, "eval")
import sheets as SH  # noqa: E402

S = Path("eval/learned_head/scale")
SH.OUT = S


def main():
    it = pd.read_parquet("data/public/index_disbench/items.parquet")
    path = dict(zip(it.item_id.astype(str), it.path))
    idx = []
    X = pd.read_parquet(S / "scores_N10000.parquet")
    ev = X[(X.src == "EV16") & (X["query"] == "photos with a boat")]
    lib = ev.groupby("lib").y.sum().sort_values().index[-1]          # the library with the most judge-yes boats
    g = ev[ev.lib == lib]
    for col, name in (("mlp", "boat_head"), ("look", "boat_look")):
        rows = [dict(path=path[str(r["item"])], caption=f"{col} {r[col]:.3f} | judge {'YES' if r.y else 'no'}")
                for _, r in g.sort_values(col, ascending=False).head(9).iterrows()]
        SH.render(rows, name); idx.append((name, lib, sum('YES' in r['caption'] for r in rows)))
    B = pd.read_parquet(S / "scores_N10000b.parquet")
    a4 = B[(B.src == "A4") & (B["query"] == "photos with a car")].copy()
    n = int(a4.y.sum())
    kb = set(a4.sort_values("mlp", ascending=False).item.head(n)); kl = set(a4.sort_values("look", ascending=False).item.head(n))
    add = a4[a4.item.isin(kb - kl) & a4.eye.isin(["match", "no_match"])].sample(frac=1, random_state=0).head(9)
    rows = [dict(path=path[str(r["item"])], caption=f"eye {r.eye} | judge {'YES' if r.y else 'no'} | head {r['head']:.2f}")
            for _, r in add.iterrows()]
    SH.render(rows, "car_blend_gain"); idx.append(("car_blend_gain", f"{len(kb - kl)} added, eye-labelled shown {len(rows)}",
                                                    sum('eye match' in r['caption'] for r in rows)))
    Q = json.load(open(S / "questions.json"))
    L = pd.concat([pd.read_parquet(f) for f in sorted((S / "labels").glob("part*.parquet"))[:8]], ignore_index=True)
    rng = np.random.default_rng(3)
    for k, q in enumerate(rng.choice(L.qid.unique(), 3, replace=False)):
        d = L[(L.qid == q) & (L.how == "top")].sample(9, random_state=k)
        rows = [dict(path=path[str(r.item_id)], caption=f"P(yes) {r.p:.3f}  {Q[q]['q'][:52]}") for _, r in d.iterrows()]
        SH.render(rows, f"teacher_q{k}"); idx.append((f"teacher_q{k}", Q[q]["q"], int((d.p >= 0.7).sum())))
    pd.DataFrame(idx, columns=["sheet", "what", "count"]).to_csv(S / "sheets" / "index.tsv", sep="\t", index=False)
    print(pd.DataFrame(idx).to_string(), flush=True)


if __name__ == "__main__":
    main()
