"""RESULTS 57: the B3 training file = the 4,426 old 27B distillation TRAIN rows (planner27ball_clean.jsonl, the RESULTS 31
adapter's data) + every passing B3 row x3 (B2's mix: data/public/distill/planner_cbmix2.jsonl = 4,426 + 465 x 3),
shuffled with a fixed seed. Usage: python eval/complex_bench/b57/make_mix.py <B3 rows.jsonl> <out .jsonl>"""
import json
import random
import sys

old = [json.loads(l) for l in open("data/public/distill/planner27ball_clean.jsonl")]
old = [r for r in old if r["split"] == "train"]
new = [json.loads(l) for l in open(sys.argv[1])]
assert all(r["split"] == "train" and r["source"] == "complex_bench_train" for r in new)
rows = old + new * 3
random.Random(0).shuffle(rows)
with open(sys.argv[2], "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"old {len(old)} + new {len(new)} x3 = {len(rows)} rows -> {sys.argv[2]}")
