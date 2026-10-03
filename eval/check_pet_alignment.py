import hashlib, io
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
from PIL import Image
d = pd.read_parquet("eval/results_pet_judge_pairs.parquet")
fs = sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))
df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
col = [c for c in df.columns if c != "label"][0]; raw = lambda v: v["bytes"] if isinstance(v, dict) else v
df["h"] = [hashlib.md5(raw(v)).hexdigest() for v in df[col]]
multi = df.groupby("h").label.nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h").reset_index(drop=True)
for mode in ("verify", "convert"):
    ok = []
    for v in df[col]:
        try:
            im = Image.open(io.BytesIO(raw(v)))
            im.verify() if mode == "verify" else im.convert("RGB")
            ok.append(True)
        except Exception:
            ok.append(False)
    lab = df[ok].reset_index(drop=True).label.to_numpy()
    s = d[d.same]; match = (lab[s.a.to_numpy()] == lab[s.b.to_numpy()]).mean()
    print(mode, "kept", sum(ok), "same-dog pairs with equal labels:", round(float(match), 3))
