#!/bin/bash
# RESULTS 54: score the four runs, paired comparison S1 - S0 per split, error counts (CPU; main env).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
D=eval/complex_bench/runs/m54
for sp in test dev; do
  for s in s0 s1; do python eval/complex_bench/score.py $sp $D/m54_${sp}_$s > /dev/null; done
  python eval/complex_bench/compare.py S0=$D/m54_${sp}_s0 S1=$D/m54_${sp}_s1 --out=$D/compare_$sp.txt > /dev/null
done
python - <<'PY'
import json
D = "eval/complex_bench/runs/m54"
for sp in ("test", "dev"):
    for s in ("s0", "s1"):
        P = json.load(open(f"{D}/{sp}_{s}_plan/plans.json"))["plans"]
        S = json.load(open(f"{D}/m54_{sp}_{s}/scored.json"))
        pe = sum(bool(r["error"]) for r in P.values())
        se = sum(bool(r["error"]) for r in S)
        none = sum(r["final"]["n_got"] == 0 for r in S)
        kinds = {}
        for r in S:
            if r["error"]:
                k = r["error"].split(":")[0]
                kinds[k] = kinds.get(k, 0) + 1
        print(f"{sp} {s}: requests scored {len(S)}/{len(P)}; plan errors {pe}; requests with an error {se} {kinds}; nothing returned {none}")
PY
