"""RESULTS 59d raw-trace review: contact sheets (native DISBench pixels) of photos one variant returns and the other does
not, final turn, union of albums, on requests whose final albums differ; label = in the benchmark truth (any album).
Sheets -> .cache/tmp/r59d_sheets (public renders, never committed), tile list -> eval/facts_scorer/r59d/sheet_<name>.txt
  python eval/facts_scorer/r59d/sheets.py <split> <run dir A> <run dir B> <name> [--n=12] [--seed=0]"""
import glob
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "eval/complex_bench")
from audit import sheet, tile  # noqa: E402

split, A, B, name = sys.argv[1:5]
arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)  # noqa: E731
n, seed = int(arg("n", "12")), int(arg("seed", "0"))
load = lambda d: {r["qid"]: r for f in glob.glob(f"{d}/part*.json") for r in json.loads(Path(f).read_text())}  # noqa: E731
RA, RB = load(A), load(B)
Q = {q["qid"]: q for q in json.loads(Path(f"eval/complex_bench/queries/{split}.json").read_text())["queries"]}
it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
path = dict(zip(it.item_id.astype(str), it.path))
cand = []
for qid in sorted(set(RA) & set(RB)):
    ta, tb = RA[qid]["turns"], RB[qid]["turns"]
    if not ta or not tb or not ta[-1]["final"] or not tb[-1]["final"]:
        continue
    a = set().union(*map(set, ta[-1]["final"]["albums"])); b = set().union(*map(set, tb[-1]["final"]["albums"]))
    ans = eval(Q[qid]["answers"]) if isinstance(Q[qid]["answers"], str) else Q[qid]["answers"]
    gt = set().union(*[set(map(str, x)) for x in ans[-1]]) if ans and ans[-1] else set()
    for side, s in (("A only", a - b), ("B only", b - a)):
        for i in s:
            cand.append((side, qid, i, i in gt, Q[qid]["turns"]))
random.Random(seed).shuffle(cand)
lines, out = [], Path(".cache/tmp/r59d_sheets"); out.mkdir(parents=True, exist_ok=True)
for side in ("A only", "B only"):
    C = [c for c in cand if c[0] == side]
    lines.append(f"== {side} ({A if side == 'A only' else B}): {len(C)} photos, in truth {sum(c[3] for c in C)}")
    tiles = []
    for k, (s, qid, i, t, words) in enumerate(C[:n], 1):
        tiles.append(tile(path[i], str(k), f"{'TRUTH' if t else 'not truth'} {qid}"))
        lines.append(f"{side} {k} {qid} item {i} truth={t} :: {words}")
    for s in range(0, len(tiles), 6):
        sheet(tiles[s:s + 6]).save(out / f"{name}_{side[0]}_{s // 6}.jpg", quality=92)
Path(f"eval/facts_scorer/r59d/sheet_{name}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
