"""Sheets of REAL photos (eye label match/right) that a candidate rule drops but q0>=0.7 keeps (look before claiming).
Writes eval/scene_fix/sheets/drop_<n>.jpg (native pixels, 6 per sheet) + eval/scene_fix/dropped.json. Public DISBench."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "eval")
from eval_question_variants import labels
from eval_recall import load_scores, _labels_eye, sheets
OUT = Path("eval/scene_fix")
new = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("part*.parquet"))], ignore_index=True)
W = new.pivot_table(index=["query", "item_id"], columns="qname", values="p").reset_index()
sc = load_scores()[["query", "item_id", "p", "path", "user", "stratum"]].rename(columns={"p": "q0r"})
W = W.merge(sc, on=["query", "item_id"], how="left")
W["q0"] = W.q0.fillna(W.q0r)
paths = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
pmap = dict(zip(paths.item_id.astype(str), paths.path))
W["path"] = W.item_id.map(pmap)
lab = labels(); eye = _labels_eye()
W["lab"] = [eye.get((q, i)) or {"right": "match", "wrong": "no_match", "unsure": "unsure"}.get(lab.get((q, i)))
            for q, i in zip(W["query"], W.item_id)]
RULES = {"strict>=0.7": lambda d: d.strict >= 0.7, "desc>=0.5": lambda d: (d.q0 >= 0.7) & (d.desc >= 0.5),
         "main>=0.5": lambda d: (d.q0 >= 0.7) & (d.main >= 0.5), "q0>=0.95": lambda d: d.q0 >= 0.95,
         "q0>=0.99": lambda d: d.q0 >= 0.99}
rows = []
for r, fn in RULES.items():
    k = fn(W).fillna(False)
    m = W[(W.q0 >= 0.7) & ~k & (W.lab == "match") & (W.strict.notna() if r.startswith("strict") else True)]
    for x in m.itertuples():
        rows.append(dict(rule=r, query=x.query, item_id=x.item_id, path=x.path, q0=x.q0, strict=x.strict, main=x.main,
                         desc=x.desc, held_out=pd.isna(x.user)))
D = pd.DataFrame(rows)
print(D.groupby(["query", "rule"]).size().to_string())
U = D.drop_duplicates("item_id").reset_index(drop=True)
U["user"] = ""; U["stratum"] = ""; U["p"] = U.q0
from PIL import ImageFont
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:
    font = ImageFont.load_default()
import eval_recall
eval_recall.OUT = OUT
key = sheets(list(U.itertuples()), "drop", font)
for k_ in key:
    k_["rules"] = sorted(D[D.item_id == k_["item_id"]].rule.unique().tolist())
    r = U[U.item_id == k_["item_id"]].iloc[0]
    k_.update(q0=round(r.q0, 3), strict=None if pd.isna(r.strict) else round(r.strict, 3), main=round(r.main, 3),
              desc=round(r.desc, 3), held_out=bool(r.held_out))
json.dump(key, open(OUT / "dropped.json", "w"), indent=1)
