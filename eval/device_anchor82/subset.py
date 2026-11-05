"""M82: the fixed photo subsets (chosen before any judge or writer output was looked at).
DEV: the 24,598 complex_bench DEV photos sorted by (library, item_id), permuted by numpy default_rng(82); the first
5,000 = the AUC set (dev5k); its first 500 = the MLX-vs-vLLM subset (dev500); the rest, in the same order, = the
extension pool used only if a question has < 50 judge-yes (4B) in dev5k (devext_<n> = the next n photos).
COMMONS (parity pool): the first 1,000 photos of the new Commons set, in commons_new.py's seeded order.
  python eval/device_anchor82/subset.py [--ext=N]   -> data/public/device_anchor82/manifest_{dev5k,dev500,commons1k,pool6k}
                                                       .{parquet,json} (pool6k = dev5k + commons1k)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/public/device_anchor82")


def save(M, name):
    M = M.reset_index(drop=True)
    M.to_parquet(D / f"manifest_{name}.parquet")
    (D / f"manifest_{name}.json").write_text(json.dumps(M[["key", "r448"]].to_dict("records")))
    print(name, len(M), flush=True)


def main():
    M = pd.read_parquet(D / "manifest_dev.parquet")
    M = M.sort_values(["user", "key"]).reset_index(drop=True)
    o = M.iloc[np.random.default_rng(82).permutation(len(M))].reset_index(drop=True)
    o["order82"] = np.arange(len(o))
    ext = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ext=")), 0))
    if ext:
        save(o.iloc[5000:5000 + ext], f"devext_{ext}")
        return
    save(o.iloc[:5000], "dev5k")
    save(o.iloc[:500], "dev500")
    if (D / "manifest_commons.parquet").exists():
        c1 = pd.read_parquet(D / "manifest_commons.parquet").iloc[:1000]
        save(c1, "commons1k")
        save(pd.concat([o.iloc[:5000], c1], ignore_index=True), "pool6k")     # one PE job + one vLLM base-writer job


if __name__ == "__main__":
    main()
