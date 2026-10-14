"""RESULTS 59: contact sheets (native DISBench pixels, <= 500 px = what the judge sees) of per-photo disagreements
between a scorer and the 9B judge on DEV truth questions, for eye review. Sheets in .cache/tmp/r59_sheets (public
DISBench renders: never committed); the facts text + scores of every tile -> <out>/<name>.txt.
  python eval/facts_scorer/sheets.py <eval_photo parquet> <scorer column> [--n=12] [--seed=0] [--facts=27b]
Kinds: miss = oracle match, 9B >= .7, scorer < .3 ; fp = oracle non-match, 9B < .3, scorer >= .7
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "eval/complex_bench")
from audit import sheet, tile  # noqa: E402

OUT = Path(".cache/tmp/r59_sheets")


def main():
    T = pd.read_parquet(sys.argv[1]); col = sys.argv[2]
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    n, seed = int(arg("n", "12")), int(arg("seed", "0"))
    tag = arg("facts", "27b")
    FT = pd.read_parquet(f"eval/facts_scorer/data/facts_{tag}.parquet")
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(it.item_id.astype(str), it.path))
    OUT.mkdir(parents=True, exist_ok=True)
    txt = []
    for kind, m in (("miss", (T.y == 1) & (T.p9b >= .7) & (T[col] < .3)),
                    ("fp", (T.y == 0) & (T.p9b < .3) & (T[col] >= .7))):
        X = T[m]
        txt.append(f"== {kind}: {len(X)} pairs of {len(T)}; by kind: " + ", ".join(f"{k} {v}" for k, v in X.kind.value_counts().items()))
        S = X.sample(min(n, len(X)), random_state=seed)
        tiles = []
        for k, r in enumerate(S.itertuples(), 1):
            tiles.append(tile(path[r.item_id], str(k), f"{r.q.split('::')[-1]}"))
            txt.append(f"{kind} {k} item {r.item_id} [{r.kind}] Q: {r.q} | oracle {r.p:.2f} 9B {r.p9b:.2f} {col} {getattr(r, col):.2f}\n"
                       f"   FACTS: {facts.get(r.item_id, '')}")
        for s in range(0, len(tiles), 6):
            sheet(tiles[s:s + 6]).save(OUT / f"{col}_{kind}_{s // 6}.jpg", quality=92)
    (OUT / f"{col}.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
