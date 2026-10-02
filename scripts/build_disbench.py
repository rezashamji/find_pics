"""DISBench (57 real users' Flickr libraries, 109,467 photos, real taken_time / GPS / reverse-geocoded address) ->
an osxphotos-shaped library_metadata.json so it goes through the product's normal ingest path (uuid = photo_id)."""
import json
from pathlib import Path

D = Path("data/public/raw/disbench")
recs, n_gps = [], 0
for f in sorted((D / "metadata").glob("*.jsonl")):
    user = f.stem
    for line in open(f):
        r = json.loads(line); m = r.get("metadata", {})
        rec = dict(uuid=str(r["photo_id"]), date=(m.get("taken_time") or "").replace(" ", "T") or None,
                   persons=[], labels=[], ismovie=False, user=user)
        if m.get("latitude") is not None and m.get("longitude") is not None:
            rec["latitude"], rec["longitude"] = m["latitude"], m["longitude"]; n_gps += 1
        if m.get("address"):
            rec["place"] = {"address_str": m["address"]}
        recs.append(rec)
json.dump(recs, open(D / "library_metadata.json", "w"))
print(f"{len(recs)} records, {n_gps} with GPS")
