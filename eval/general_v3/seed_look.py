import json, sys
import pandas as pd
from PIL import Image
sys.path.insert(0, "src")
from findpics.media import load_image
items = pd.read_parquet("data/public/index_testlib/items.parquet"); Q = {q["k"]: q for q in json.load(open("eval/general/queries.json"))}
ims = []
for k in (19, 43, 67):
    q = Q[k]; im = load_image(items.path.iloc[int(q["seed_row"])]); im.thumbnail((900, 900)); ims.append(im)
    print(k, "|", q.get("caption", "")[:400])
W = sum(i.width for i in ims) + 20; H = max(i.height for i in ims); s = Image.new("RGB", (W, H), "white"); x = 0
for i in ims:
    s.paste(i, (x, 0)); x += i.width + 10
s.save("eval/general_v3/seed_attr.jpg", quality=85)
