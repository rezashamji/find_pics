"""RESULTS 59, eye-labelled everyday searches (public DISBench; all on TRAIN libraries -> scored by fold models that did
not train on the photo's library):
  E1 = RESULTS 45 (eval/vector_enough: 746 blind labels; census of first screens + stratified ev16 draws; NOT uniform)
  E2 = RESULTS 34 (eval/recall_audit: 1,950 blind labels, stratified by the phone judge's P; weight = stratum size /
       labelled photos of that stratum (sample_counts.csv), so weighted sums estimate the 4-library population)
Question per search = the product's wording (stored in recall_audit for its 6 searches; ev16's by the same templates).
  python eval/facts_scorer/eye_sets.py -> eval/facts_scorer/data/eye_sets.parquet
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

OBJ = "Is there a real {} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
QUESTION = {
    "photos with a dog": OBJ.format("dog"), "photos with a car": OBJ.format("car"),
    "photos with a bicycle": OBJ.format("bicycle"), "photos with a boat": OBJ.format("boat"),
    "photos of a cat": "Is this a photo of a cat?", "photos of a church": "Is this a photo of a church?",
    "photos of flowers": "Is this a photo of flowers?", "sunset photos": "Is this a photo of a sunset?",
    "beach photos": "Is this a photo of a beach?", "food photos": "Is this a photo of food?",
    "photos taken at night": "Was this photo taken at night?",
}


def labels(files):
    out = {}
    for f in files:
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            sheet, num, lab = line.split()[:3]
            out[(sheet.replace(".jpg", ""), int(num))] = lab
    return out


def main():
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    user_of = dict(zip(it.item_id.astype(str), it.path.str.extract(r"/images/images/([^/]+)/")[0]))
    rows = []
    L1 = labels(sorted(Path("eval/vector_enough").glob("labels_*.txt")))
    for k in json.load(open("eval/vector_enough/key.json")):
        lab = L1.get((k["sheet"].replace(".jpg", ""), int(k["number"])))
        if lab is None:
            continue
        iid = str(k["item_id"])
        rows.append(dict(set="E1", src=k["source"], user=user_of[iid], item_id=iid, query=k["query"],
                         question=QUESTION[k["query"]], eye=lab, stratum="", weight=1.0, p_phone=float("nan")))
    L2 = labels(["eval/recall_audit/labels.txt"])
    K2 = json.load(open("eval/recall_audit/key.json")) + json.load(open("eval/recall_audit/key2.json"))
    cnt = pd.read_csv("eval/recall_audit/sample_counts.csv").set_index("query")
    nlab = Counter((k["query"], k["stratum"]) for k in K2 if (k["sheet"].replace(".jpg", ""), int(k["number"])) in L2)
    for k in K2:
        lab = L2.get((k["sheet"].replace(".jpg", ""), int(k["number"])))
        if lab is None:
            continue
        N = float(cnt.loc[k["query"], f"N_{k['stratum']}"])
        rows.append(dict(set="E2", src="audit4", user=k["user"], item_id=str(k["item_id"]), query=k["query"],
                         question=QUESTION[k["query"]], eye=lab, stratum=k["stratum"],
                         weight=N / nlab[(k["query"], k["stratum"])], p_phone=float(k["p"])))
    T = pd.DataFrame(rows)
    T.to_parquet("eval/facts_scorer/data/eye_sets.parquet")
    print(T.groupby(["set", "eye"]).size().to_string())
    print(T.groupby("set").user.nunique().to_string())


if __name__ == "__main__":
    main()
