"""RESULTS 59d: dump TRAIN requests of one category with all-judge vs R59c (4B facts) F1 and every judge question the
all-judge stack asked, with its R59c route. TRAIN only (the design set for rule (c)).
  python eval/facts_scorer/r59d/dump_h.py [cat=h] [split=train]"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "eval/facts_scorer")
from route import TRIP, _TAIL, place_candidates, route  # noqa: E402

cat = sys.argv[1] if len(sys.argv) > 1 else "h"
split = sys.argv[2] if len(sys.argv) > 2 else "train"
if split == "train":
    REF, PT, R = "eval/facts_scorer/replay/train_alljudge", "eval/facts_scorer/replay/train_passthrough", "eval/facts_scorer/replay/c59c_train_4b"
else:
    raise SystemExit("TRAIN only")
A = {r["qid"]: r for r in json.loads(Path(REF, "scored.json").read_text())}
C = {r["qid"]: r for r in json.loads(Path(R, "r59c/scored.json").read_text())}
K = {r["qid"]: r for r in json.loads(Path(R, "k0/scored.json").read_text())}
L = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{PT}/pairs*.parquet")])
nq = L.groupby(["qid", "question", "crop"]).size()
bench = {q["qid"]: q for q in json.loads(Path("eval/complex_bench/queries/train.json").read_text())["queries"]}
plans = json.loads(Path("eval/facts_scorer/replay/train_plans/plans_r60.json").read_text())["plans"]
rows = sorted([q for q in bench if bench[q]["cat"] == cat and q in A], key=lambda q: C[q]["final"]["f1"] - A[q]["final"]["f1"])
for q in rows:
    a, c, k = A[q]["final"]["f1"], C[q]["final"]["f1"], K[q]["final"]["f1"]
    print(f"\n### {q}  alljudge {a:.2f}  r59c {c:.2f}  k0 {k:.2f}  (n_gt {A[q]['final']['n_gt']}, got aj {A[q]['final']['n_got']} r59c {C[q]['final']['n_got']})")
    for i, t in enumerate(bench[q]["turns"]):
        print(f"  turn {i}: {t}")
    sp = plans.get(q)
    if sp:
        for i, t in enumerate(sp["turns"]):
            for al in t["plan"]["albums"]:
                print(f"  plan {i}: " + json.dumps({k_: v for k_, v in al.items() if v not in (None, [], '', False)})[:400])
    if q in nq.index.get_level_values(0):
        for (qq, cr), n in nq.loc[q].items():
            r = route(qq, cr)
            trip = bool(TRIP.search(_TAIL.sub("", qq)))
            print(f"    Q[{n}] crop={cr} route={r or 'scorer'}{' TRIP' + str(place_candidates(qq)) if trip else ''}: {qq}")
