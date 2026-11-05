"""M82 cluster anchor, step 1 (CPU, main env): photo manifests + the writer's 448 px rendition files.

  python eval/device_anchor82/prep.py dev        all complex_bench DEV photos (13 libraries; media = photo)
  python eval/device_anchor82/prep.py commons    the new Commons iPhone set (commons_new.py; downloaded files only)
  -> data/public/device_anchor82/manifest_<set>.parquet: key, set, src (judge input = the stored photo), r448 (writer and
     PE input), prefix (factsPrefix built as the PRODUCT builds it: place = FindPicsCore Geocoder.placeText(lat, lon) =
     findpics.ingest.nearest_place; taken = capture month in UTC), prefix_nom (DEV only: the benchmark's own Nominatim
     address string instead, = what towervec was trained with, eval/facts_scorer/data.py --meta=1)
     data/public/device_anchor82/r448/<key>.jpg = the EXACT bytes eval/facts_writer/write_facts.py sends to vLLM (long
     side 448, Lanczos, JPEG q80): the phone's ~448 px local rendition as simulated everywhere since RESULTS 61.
"""
import base64
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, "eval/facts_writer")
sys.path.insert(0, "src")
OUT = Path("data/public/device_anchor82")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]


def facts_prefix(place, taken):
    """FindPicsCore.factsPrefix(place:taken:) - address parts without digits, de-duplicated in order; month in UTC."""
    parts = []
    for p in (place if isinstance(place, str) else "").split(","):
        t = p.strip()
        if not t or re.search(r"\d", t) or t in parts:
            continue
        parts.append(t)
    s = ("place: " + ", ".join(parts) + ". ") if parts else ""
    if taken is not None and not pd.isna(taken):
        s += f"taken: {MONTHS[taken.month - 1]} {taken.year}. "
    return s


def rendition(args):
    src, dst = args
    if Path(dst).exists():
        return True
    from write_facts import data_url
    try:
        b = base64.b64decode(data_url(src).split(",", 1)[1])
        Path(dst + ".part").write_bytes(b); Path(dst + ".part").rename(dst)
        return True
    except Exception as e:
        print("rendition failed", src, e, flush=True)
        return False


def place_of(lat, lon):
    from findpics.ingest import nearest_place
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    lat, lon = float(lat), float(lon)
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        return None
    return nearest_place(lat, lon)


def main():
    which = sys.argv[1]
    (OUT / "r448").mkdir(parents=True, exist_ok=True)
    if which == "dev":
        it = pd.read_parquet("data/public/index_disbench/items.parquet")
        it["item_id"] = it.item_id.astype(str)
        it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
        sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
        d = it[it.user.isin(sp["dev"])].copy()
        print("dev items", len(d), d.media.value_counts().to_dict(), flush=True)
        d = d[d.media == "photo"].sort_values(["user", "item_id"]).reset_index(drop=True)
        taken = pd.to_datetime(d.taken, utc=True, errors="coerce", format="ISO8601")
        M = pd.DataFrame(dict(key=d.item_id, set="dev", user=d.user, src=d.path,
                              r448=[str((OUT / "r448" / f"{k}.jpg").resolve()) for k in d.item_id]))
        M["place"] = [place_of(a, b) for a, b in zip(d.lat, d.lon)]
        M["prefix"] = [facts_prefix(p, t) for p, t in zip(M.place, taken)]
        M["prefix_nom"] = [facts_prefix(p, t) for p, t in zip(d.place, taken)]
    else:
        C = pd.read_parquet(OUT / "commons_manifest.parquet")
        C = C[C.ok].reset_index(drop=True)
        ym = C.taken_exif.fillna("").str.extract(r"^(\d{4})[:-](\d\d)")   # EXIF local time: the month as written
        taken = [pd.Timestamp(int(y), int(m), 1) if isinstance(y, str) and 1 <= int(m) <= 12 else None
                 for y, m in zip(ym[0], ym[1])]
        M = pd.DataFrame(dict(key=C.key, set="commons", user=C.user, src=C.path,
                              r448=[str((OUT / "r448" / f"{k}.jpg").resolve()) for k in C.key]))
        M["place"] = [place_of(a, b) for a, b in zip(pd.to_numeric(C.gps_lat, errors="coerce"),
                                                     pd.to_numeric(C.gps_lon, errors="coerce"))]
        M["prefix"] = [facts_prefix(p, t) for p, t in zip(M.place, taken)]
        M["prefix_nom"] = M.prefix
    with ProcessPoolExecutor(8) as ex:
        ok = list(ex.map(rendition, zip(M.src, M.r448), chunksize=64))
    M = M[ok].reset_index(drop=True)
    M.to_parquet(OUT / f"manifest_{which}.parquet")
    (OUT / f"manifest_{which}.json").write_text(json.dumps(M[["key", "r448"]].to_dict("records")))
    print(f"PREP_DONE {which}: {len(M)} photos; with place {int(M.place.notna().sum())}; prefix examples:",
          M.prefix.head(3).tolist(), flush=True)


if __name__ == "__main__":
    main()
