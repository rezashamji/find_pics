"""A fake Apple 'copy of your data' export from PUBLIC data, to dry-run scripts/run_private_copy.sh end to end:
2 zips, each: iCloud Photos/Photos/<files> + Photo Details.csv; 200 test-library photos + 20 Pexels videos; a few rows marked
deleted and a Recently Deleted folder (must be skipped). -> data/public/fakecopy_zips/"""
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

Z = Path("data/public/fakecopy_zips"); Z.mkdir(parents=True, exist_ok=True)
for f in Z.glob("*.zip"):
    f.unlink()
items = pd.read_parquet("data/public/index_testlib/items.parquet")
photos = list(items[items.media == "photo"].sample(200, random_state=3).path)
videos = sorted(Path("data/public/raw/pexels_videos").glob("*.mp4"))[:20]
files = photos + [str(v) for v in videos]
for part in range(2):
    sel = files[part::2]
    rows = io.StringIO(); w = csv.writer(rows)
    w.writerow(["imgName", "fileChecksum", "favorite", "hidden", "deleted", "originalCreationDate", "viewCount", "importDate"])
    with zipfile.ZipFile(Z / f"iCloud Photos Part {part + 1} of 2.zip", "w", zipfile.ZIP_STORED) as z:
        for i, p in enumerate(sel):
            name = Path(p).name
            z.write(p, f"iCloud Photos/Photos/{name}")
            w.writerow([name, "x", "no", "no", "yes" if i % 50 == 7 else "no", "Tuesday October 1,2019 5:20 PM GMT", "1", ""])
        z.write(sel[0], "iCloud Photos/Recently Deleted/" + Path(sel[0]).name)
        z.writestr("iCloud Photos/Photo Details.csv", rows.getvalue())
    print(Z / f"iCloud Photos Part {part + 1} of 2.zip", len(sel), "files")
