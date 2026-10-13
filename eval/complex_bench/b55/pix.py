"""RESULTS 55: pixel statistics (dump.pixel_stats) for every photo of some libraries, for fitting look-from-pixel thresholds
on TRAIN libraries. Usage: python eval/complex_bench/b55/pix.py split=train k=0 K=4 -> eval/complex_bench/b55/pix/pix_<user>.parquet"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, load_items  # noqa: E402
from dump import pixel_stats  # noqa: E402


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"{k}=")), d)
    split, k, K = arg("split", "train"), int(arg("k", 0)), int(arg("K", 1))
    users = sorted(json.loads((ROOT / "splits.json").read_text())[split])[k::K]
    from findpics.media import load_image
    it = load_items()
    out = ROOT / "b55" / "pix"; out.mkdir(parents=True, exist_ok=True)
    for u in users:
        f = out / f"pix_{u}.parquet"
        if f.exists():
            continue
        rows = []
        for iid, p in zip(it[it.user == u].item_id, it[it.user == u].path):
            try:
                rows.append(dict(item_id=iid, **pixel_stats(load_image(p))))
            except Exception as e:
                print(u, iid, e, flush=True)
        pd.DataFrame(rows).to_parquet(f)
        print(u, len(rows), flush=True)


if __name__ == "__main__":
    main()
