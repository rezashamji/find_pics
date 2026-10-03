"""Render the different-dog pairs the strict side-by-side judge called 'same' (eval/results_pet_judge_pairs.parquet).
Rebuilds eval_pet_judge.py's photo list (dedupe + drop undecodable) without decoding every photo into memory."""
import hashlib
import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

d = pd.read_parquet("eval/results_pet_judge_pairs.parquet")
fs = sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))
df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
col = [c for c in df.columns if c != "label"][0]
raw = lambda v: v["bytes"] if isinstance(v, dict) else v
df["h"] = [hashlib.md5(raw(v)).hexdigest() for v in df[col]]
multi = df.groupby("h").label.nunique()
df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h").reset_index(drop=True)
ok = []
for v in df[col]:
    try:
        Image.open(io.BytesIO(raw(v))).convert("RGB"); ok.append(True)   # same test as eval_pet_judge (verify() keeps 4 more -> misaligned)
    except Exception:
        ok.append(False)
df = df[ok].reset_index(drop=True)
img = lambda i: Image.open(io.BytesIO(raw(df[col].iloc[int(i)]))).convert("RGB").resize((450, 450))
fp = d[(~d.same) & (d.p_strict >= 0.476)].sort_values("p_strict", ascending=False)
print("different-dog pairs called same at the best cut:", len(fp), "of", int((~d.same).sum()))
sheet = Image.new("RGB", (2 * 910 + 20, 2 * 460), "white")
for k, r in enumerate(fp.head(4).itertuples()):
    x, y = (k % 2) * 930, (k // 2) * 460
    sheet.paste(img(r.a), (x, y)); sheet.paste(img(r.b), (x + 455, y))
    print(k, int(r.a), int(r.b), round(float(r.p_strict), 2), "labels", df.label.iloc[int(r.a)], df.label.iloc[int(r.b)])
sheet.save("eval/pet_false_same.jpg", quality=88)
