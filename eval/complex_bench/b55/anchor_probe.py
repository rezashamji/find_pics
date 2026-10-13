"""RESULTS 55: the 9B's cached P for the plan's ANCHOR question on the truth day of 'smooth water from that river trip'
(70408381@N00:b6), to see why the same_event window missed the Mostar trip. Cache lookup only (no model)."""
import glob, hashlib, json, sys
import pandas as pd
from PIL import Image, ImageOps
sys.path.insert(0, "src")
from findpics.media import load_image
U, Q = "70408381@N00", "Is this a photo from a river trip?"
it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path", "taken"])
it = it[it.path.str.contains(f"/{U}/")]
cache = {}
for f in glob.glob(f"eval/complex_bench/runs/cmp/jcache/*/{U}.json"):
    cache.update(json.load(open(f)))
rows = []
for iid, p, t in zip(it.item_id, it.path, it.taken):
    im = load_image(p)
    k = hashlib.md5(Q.encode() + im.tobytes() + str(im.size).encode()).hexdigest()
    rows.append(dict(item_id=str(iid), day=str(t)[:10], p=cache.get(k)))
d = pd.DataFrame(rows)
print("photos with a cached anchor answer:", d.p.notna().sum(), "of", len(d))
print("anchor P >= 0.7 by day:\n", d[d.p >= 0.7].groupby("day").size().sort_values(ascending=False).head(15).to_string())
m = d[d.day == "2010-01-09"]
print("Mostar day 2010-01-09:", len(m), "photos; anchor P max %.3f, median %.3f, >=0.7: %d" % (m.p.max(), m.p.median(), (m.p >= 0.7).sum()))
