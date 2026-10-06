"""Score a heavier-vs-fit demo run against Reza's own labels (data/private/audits/reza_labels_demo6.json: file name ->
H / F / other). Prints counts only (no file names). Usage: python scripts/score_demo.py <run dir> [<run dir> ...]"""
import collections
import json
import re
import sys
from pathlib import Path

lab = json.load(open("data/private/audits/reza_labels_demo6.json"))
tot = collections.Counter(lab.values())
for run in sys.argv[1:]:
    t = Path(run) / "turn_1"
    out = {}
    for d in sorted(p for p in t.iterdir() if p.is_dir() and p.name != "thumbs"):
        names = [re.sub(r"^[0-9a-f]{8}_", "", f.name) for f in d.iterdir() if f.is_file()]
        out[d.name.split()[-1]] = collections.Counter(lab.get(n, "unlabeled") for n in names)
    print(Path(run).name, {k: dict(v) for k, v in out.items()}, "| labeled totals", dict(tot))
