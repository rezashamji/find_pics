"""Plumbing check for Reza's library (the holdout): did everything load, and is the data good enough -- WITHOUT running
any search or looking at any result (Reza, 10-03: his library is the final exam).
Reads only the index metadata (items/units/faces parquet + DONE markers) and image headers (size), never pixels' content.
Prints: counts by media, decode errors, date source mix (exif / sidecar / takeout / mtime), GPS + place coverage, faces,
video frames, image long-side distribution (how many below the judge's ~896 px), and per-year counts.
Output stays in data/private (never committed). Usage: python scripts/plumbing_report.py <index_dir> [--sizes N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main():
    root = Path(sys.argv[1]); n_sizes = int(sys.argv[sys.argv.index("--sizes") + 1]) if "--sizes" in sys.argv else 2000
    it = pd.read_parquet(root / "items.parquet")
    shards = sorted((root / "shards").glob("*"))
    done = [json.loads((s / "stats.json").read_text()) for s in shards if (s / "DONE").exists() and (s / "stats.json").exists()]
    rep = dict(items=len(it), by_media=it.media.value_counts().to_dict(),
               shards_done=f"{len(done)}/{len(shards)}",
               units=int(sum(d.get("units", 0) for d in done)), faces=int(sum(d.get("faces", 0) for d in done)),
               decode_errors=int(sum(d.get("errors", 0) for d in done)))
    src_col = next((c for c in ("taken_source", "taken_src", "date_source") if c in it), None)
    if src_col:   # where each date came from: exif / sidecar / takeout / metadata_json / mtime (mtime = no real date)
        rep["date_source"] = it[src_col].value_counts().to_dict()
    t = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")
    rep["years"] = t.dt.year.value_counts().sort_index().astype(int).to_dict()
    if "lat" in it:
        rep["with_gps"] = int(it.lat.notna().sum())
    if "place" in it:
        rep["with_place_name"] = int((it.place.fillna("") != "").sum())
    photos = it[it.media == "photo"]
    sample = photos.sample(min(n_sizes, len(photos)), random_state=0)
    sides = []
    for p in sample.path:
        try:
            with Image.open(p) as im:
                sides.append(max(im.size))
        except Exception:
            sides.append(-1)
    sides = np.array(sides)
    rep["photo_long_side_sample"] = dict(n=int(len(sides)), unreadable=int((sides < 0).sum()),
                                         below_896=int(((sides >= 0) & (sides < 896)).sum()),
                                         median=int(np.median(sides[sides >= 0])) if (sides >= 0).any() else None)
    print(json.dumps(rep, indent=1, default=str))
    out = Path("data/private/audits"); out.mkdir(parents=True, exist_ok=True)
    (out / "plumbing_report.json").write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
