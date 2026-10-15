"""RESULTS 59d: golden fixture for Swift QuestionTree (FindPicsCore) from findpics.compose - every distinct judge question
the TRAIN + dev all-judge stacks asked that compose could touch (trip words, " or ", "and also", two sentences) + 400
plain ones, each with its library's distinct place texts; expected = compose.tree_json(decompose(...)) (masks as indices
into the library's place list), plus split_parts / remainder / PlaceIndex.mask on hand cases.
  python eval/facts_scorer/r59d/compose_fixture.py -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/compose.json"""
import glob
import json
import random
import re

import pandas as pd

from findpics import compose as C

I = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["path", "place"])
I["user"] = I.path.str.extract(r"/images/images/([^/]+)/")[0]
L = pd.concat([pd.read_parquet(f, columns=["qid", "question"]) for d in ("train_passthrough", "passthrough_r60")
               for f in glob.glob(f"eval/facts_scorer/replay/{d}/pairs*.parquet")]).drop_duplicates()
L["user"] = L.qid.str.split(":").str[0]
L = L[["user", "question"]].drop_duplicates()
touch = L.question.map(lambda q: bool(C.TRIP.search(C._TAIL.sub("", q)) or re.search(r"(?i)\bor\b|\band also\b|\?\s+[A-Z]", C._TAIL.sub("", q))))
plain = L[~touch].sample(400, random_state=0)
Q = pd.concat([L[touch], plain])
libs = {}
for u in sorted(Q.user.unique()):
    libs[u] = sorted(set(I[I.user == u].place.fillna("")))
cases = []
for u, g in Q.groupby("user"):
    PI = C.PlaceIndex(libs[u])
    for q in sorted(g.question):
        cases.append({"lib": u, "q": q, "tree": C.tree_json(C.decompose(q, PI))})
hand = ["Leon", "Machu Picchu", "Pagsanjan Falls", "London Olympics", "Pompeii", "Puerto Rico", "Madrid or León",
        "New York City", "Galápagos Islands", "Zürich"]
places = ["Viñales, Pinar del Río, Cuba", "Pinagsanjan, Laguna, Philippines", "Machupicchu, Urubamba, Cusco, Peru",
          "León, Castilla y León, España", "Madrid, España", "Olympic Park, Stratford, London, United Kingdom",
          "Soho, London, United Kingdom", "San Juan, Puerto Rico", "Pompei, Napoli, Italia", "",
          "New York County, New York, United States", "Isla Santa Cruz, Galápagos, Ecuador", "Zurich, Switzerland"]
PI = C.PlaceIndex(places)
masks = [{"name": n, "rows": [int(i) for i in PI.mask(n).nonzero()[0]]} for n in hand]
folds = [{"s": s, "fold": C.fold(s)} for s in places + ["Ærø Ø ß İstanbul Œuvre"]]
hq = ["Is this building in real life or not?", "Is this a real dog or a toy dog?", "Is there a cat or anything else in this photo?",
      "Is there a dog and also a cat in this photo?", "Is the bottle on a kitchen counter or a table?",
      "Is this a photo of a pink or yellow flower?", "Is there a red car, or are there people in this photo?"]
splits = [{"q": q, "split": (lambda r: None if r is None else [r[0], r[1]])(C.split_parts(q))} for q in hq]
json.dump({"libs": libs, "cases": cases, "hand_places": places, "masks": masks, "folds": folds, "splits": splits},
          open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/compose.json", "w"), ensure_ascii=False)
print(len(cases), "cases", sum(c["tree"].get("leaf") != c["q"] for c in cases), "composed;", len(libs), "libraries")
