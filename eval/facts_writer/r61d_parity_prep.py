"""RESULTS 61d (2) parity, step 1 (main env): pick 50 public DEV photos (seed 0), write each one's phone rendition (the
EXACT bytes write_facts.py sends to vLLM: 448 px Lanczos, JPEG q80) to .cache/r61d/parity/img/<item>.jpg, and the vLLM
phone-form reference line for each from eval/facts_writer/facts/<tag> (written by that checkpoint's q4 form).
  python eval/facts_writer/r61d_parity_prep.py <tag, e.g. t2b_final_xfit> [--n=50]
"""
import base64
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from write_facts import data_url  # noqa: E402


def main():
    tag = sys.argv[1]
    n = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--n=")), 50))
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    it["item_id"] = it.item_id.astype(str); it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    ref = {}
    for u in sp["dev"]:
        for line in open(Path("eval/facts_writer/facts") / tag / f"{u}.jsonl"):
            r = json.loads(line); ref[str(r["item_id"])] = r
    dev = it[it.user.isin(sp["dev"]) & it.item_id.isin(ref)].sort_values("item_id")
    pick = dev.iloc[sorted(random.Random(0).sample(range(len(dev)), n))]
    out = Path(".cache/r61d/parity"); (out / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in zip(pick.item_id, pick.path):
        jpg = out / "img" / f"{i}.jpg"
        jpg.write_bytes(base64.b64decode(data_url(p).split(",", 1)[1]))
        rows.append(dict(item_id=i, image=str(jpg.resolve()), vllm=ref[i]["facts"]["caption"], vllm_n_out=ref[i]["n_out"],
                         vllm_n_prompt=ref[i]["n_prompt"]))
    (out / f"photos_{tag}.json").write_text(json.dumps(rows, indent=1))
    print("PARITY_PREP", len(rows), "photos ->", out / f"photos_{tag}.json")


if __name__ == "__main__":
    main()
