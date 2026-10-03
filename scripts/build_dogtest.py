"""Small library for an end-to-end 'find my dog Max' test of the product path (public data only).
data/public/dogtest/library: 300 random test-library photos + 50 photos of OTHER DogFaceNet dogs + 3 photos of 'Max';
data/public/dogtest/refs: 3 other photos of Max (the --ref photos). Writes truth.json (Max's library file names)."""
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

R = Path("data/public/dogtest"); shutil.rmtree(R, ignore_errors=True)
(R / "library").mkdir(parents=True); (R / "refs").mkdir()
rng = np.random.default_rng(7)
df = pd.concat([pq.read_table(f).to_pandas() for f in sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))], ignore_index=True)
col = [c for c in df.columns if c != "label"][0]; raw = lambda v: v["bytes"] if isinstance(v, dict) else v
df["h"] = [hashlib.md5(raw(v)).hexdigest() for v in df[col]]
multi = df.groupby("h").label.nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h")
cnt = df.label.value_counts(); max_id = rng.choice(cnt[cnt >= 6].index)
mine = df[df.label == max_id].sample(6, random_state=7)
others = df[df.label != max_id].sample(50, random_state=7)
save = lambda v, p: Image.open(io.BytesIO(raw(v))).convert("RGB").save(p, quality=92)
for i, v in enumerate(mine[col].iloc[:3]):
    save(v, R / "refs" / f"max_ref{i}.jpg")
truth = []
for i, v in enumerate(mine[col].iloc[3:]):
    save(v, R / "library" / f"maxdog_{i}.jpg"); truth.append(f"maxdog_{i}.jpg")
for i, v in enumerate(others[col]):
    save(v, R / "library" / f"otherdog_{i}.jpg")
items = pd.read_parquet("data/public/index_testlib/items.parquet")
ph = items[items.media == "photo"].sample(300, random_state=7)
for p in ph.path:
    shutil.copy(p, R / "library" / Path(p).name)
json.dump(dict(max_label=str(max_id), truth=truth), open(R / "truth.json", "w"), indent=1)
print("dogtest:", len(list((R / "library").iterdir())), "library photos; Max truth", truth)
