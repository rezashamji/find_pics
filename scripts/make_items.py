"""ingest.scan -> <index_dir>/items.parquet. Usage: python scripts/make_items.py <library_root> <index_dir> [metadata.json]"""
import sys
from pathlib import Path
from findpics.ingest import scan, to_frame

lib, out = sys.argv[1], Path(sys.argv[2])
meta = sys.argv[3] if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)
df = to_frame(scan(lib, meta))
df.to_parquet(out / "items.parquet")
print(df["media"].value_counts().to_dict(), df["date_source"].value_counts().to_dict(), "->", out / "items.parquet")
