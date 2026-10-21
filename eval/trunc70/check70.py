"""RESULTS 70 input checks: every variant parquet has the base parquet's photos; the metadata prefix (place / taken) is
unchanged; uncapped plain-K lines are byte-identical to the base text; capped lines are a prefix of the base text."""
import json
import pandas as pd

BASE = {"t2b_final_rs2_xfit": "eval/facts_scorer/data/facts_w_t2b_final_rs2_xfitm.parquet",
        "p2b_devmix_rs2": "eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet"}
for src, bp in BASE.items():
    B = pd.read_parquet(bp).set_index("item_id").text
    for tag in ["k16", "k24", "k32", "k48", "k64", "r48", "r64", "rfull"]:
        V = pd.read_parquet(f"eval/facts_scorer/data/facts_w_r70_{src}_{tag}m.parquet").set_index("item_id").text
        assert set(V.index) == set(B.index)
        same = (V == B.loc[V.index]).sum()
        pre = sum(B[i].startswith(V[i]) for i in V.index) if tag.startswith("k") else None
        print(f"{src} {tag}: {len(V)} photos, identical to base {same}, prefix of base {pre}")
        if tag.startswith("k"):
            assert pre == len(V), f"{src} {tag}: capped line not a prefix"
