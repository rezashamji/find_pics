"""RESULTS 61d: remove requests that errored (infrastructure: lazily loaded 9B engine init failure) from cascade outputs
and print the shards to re-run (`<replay dir> <k>` per line). Usage: python eval/facts_writer/r61d_fix_errors.py <replay dir> ..."""
import glob
import json
import re
import sys
from pathlib import Path

for D in sys.argv[1:]:
    for f in glob.glob(f"{D}/*/part*.json"):
        R = json.load(open(f)); bad = {r["qid"] for r in R if r.get("error")}
        if not bad:
            continue
        k = re.search(r"part(\d+)", f).group(1)
        Path(f).write_text(json.dumps([r for r in R if r["qid"] not in bad]))
        st = Path(f).parent / f"stats{k}.json"
        if st.exists():
            S = json.loads(st.read_text()); st.write_text(json.dumps({q: v for q, v in S.items() if q not in bad}))
        print(D, k, len(bad), file=sys.stderr)
        print(D, k)
