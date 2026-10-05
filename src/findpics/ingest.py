"""Walk a photo library (plain folder, Google Takeout, or osxphotos export) and build an item table.

Read-only: files are opened for reading only. Nothing here writes next to the user's photos.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

try:   # HEIC (iPhone default): without this, dates/GPS of AirDropped photos fell back to file mtime (sample 10-04: 549/598)
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".3gp", ".webm"}


@dataclass
class Item:
    item_id: str          # stable id: Apple uuid when available, else path relative to root
    path: str             # absolute path
    media: str            # "photo" | "video"
    taken: str | None     # ISO-8601 capture time if known
    date_source: str      # "sidecar" | "metadata_json" | "exif" | "takeout" | "mtime"
    apple_persons: list[str]
    apple_labels: list[str]
    width: int | None = None
    height: int | None = None
    lat: float | None = None
    lon: float | None = None
    place: str = ""          # human-readable place text (Apple place names, Takeout/EXIF GPS -> offline reverse geocode)
    taken_local: str | None = None   # wall-clock time where it was taken ("YYYY-MM-DDTHH:MM:SS"), for time-of-day filters;
                                     # None when only a UTC instant is known (Apple's CSV, Takeout timestamps, mtime)
    camera: str = ""                 # "front" | "back" from EXIF LensModel ("iPhone 13 Pro front camera 2.71mm f/2.2"); "" unknown


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _exif_date(path: Path) -> datetime | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            # 36867 DateTimeOriginal lives in the Exif sub-IFD (0x8769); 306 DateTime in IFD0
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            raw = sub.get(36867) or exif.get(36867) or exif.get(306)
        if raw:
            return datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def _wall_clock(s, need_offset: bool = False) -> str | None:
    """Local wall-clock time from a timestamp STRING: naive EXIF/Flickr times are already local; a string with an offset
    ("2019-07-04T21:03:11-04:00") shows local time before the offset. Epoch numbers are UTC instants -> None."""
    if not s or isinstance(s, (int, float)):
        return None
    m = re.match(r"(\d{4})[-:](\d\d)[-:](\d\d)[ T](\d\d):(\d\d):(\d\d)(.*)$", str(s).strip())
    if not m:
        return None
    rest = m.group(7).strip()
    if rest in ("Z", "UTC", "GMT", "+00:00", "+0000") and need_offset:
        return None
    if need_offset and not re.match(r"[+-]\d\d", rest):
        return None
    if rest in ("Z", "UTC", "GMT"):
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}"


def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    s = str(s).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S%z", "%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:25] if "%z" in fmt else s[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _sidecar_candidates(path: Path) -> list[Path]:
    """osxphotos `--sidecar json`: <file>.json. Google Takeout: <file>.json (old), <file>.supplemental-metadata.json
    (since late 2024), and truncated forms like <file>.supplemental-metadat.json / .supplementa.json because Takeout
    clips sidecar names at 46 characters."""
    import glob as _glob
    c = [Path(str(path) + ".json"), Path(str(path) + ".supplemental-metadata.json"), path.with_suffix(".json")]
    c += sorted(path.parent.glob(_glob.escape(path.name) + ".supp*.json"))
    if len(path.name) > 30:  # whole sidecar name clipped at 46 chars
        c += sorted(q for q in path.parent.glob(_glob.escape(path.name[:30]) + "*.json") if q.name != path.name)
    return c


def _sidecar(path: Path) -> dict | None:
    for cand in _sidecar_candidates(path):
        if cand.exists():
            try:
                data = json.loads(cand.read_text())
                return data[0] if isinstance(data, list) and data else data
            except Exception:
                return None
    return None


def _strings(x) -> list[str]:
    """All strings inside a nested dict/list (osxphotos `place` is nested: name, names{city:[...], country:[...]}, address_str)."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, (list, tuple)):
        return [s for v in x for s in _strings(v)]
    return []


def _video_meta(path: Path):
    """(taken UTC, local wall-clock string, (lat, lon)) from a video's container metadata. iPhone .MOV/.MP4 keep
    com.apple.quicktime.creationdate WITH its offset (local time) and location.ISO6709; creation_time is UTC."""
    try:
        import av
        with av.open(str(path)) as c:
            meta = {**dict(c.metadata), **(dict(c.streams.video[0].metadata) if c.streams.video else {})}
    except Exception:
        return None, None, None
    qc = meta.get("com.apple.quicktime.creationdate") or ""
    ct = qc or meta.get("creation_time") or ""
    taken = local = None
    if ct:
        try:
            taken = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            taken = None
    if re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d", qc):
        local = qc[:19]
    m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", meta.get("com.apple.quicktime.location.ISO6709") or meta.get("location") or "")
    gps = (float(m.group(1)), float(m.group(2))) if m else None
    return taken, local, gps


def _exif_camera(path: Path) -> str:
    """'front' / 'back' from EXIF LensModel (0xA434, Exif sub-IFD). Phones name the selfie camera ("front camera");
    files without a lens tag (screenshots, saved or received images, many older cameras) are '' (unknown)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            lm = str(im.getexif().get_ifd(0x8769).get(0xA434) or "").lower()
        return "" if not lm else ("front" if "front" in lm else "back")
    except Exception:
        return ""


def _exif_gps(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            g = im.getexif().get_ifd(0x8825)
        if not g or 2 not in g or 4 not in g:
            return None
        dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat = dms(g[2]) * (-1 if g.get(1) == "S" else 1); lon = dms(g[4]) * (-1 if g.get(3) == "W" else 1)
        return (lat, lon) if (lat, lon) != (0.0, 0.0) else None
    except Exception:
        return None


import numpy as np  # noqa: E402  (geocoding only)

_GEO = None


def _geo():
    """Nearest-city table: reverse_geocoder's GeoNames cities1000 (CC-BY 4.0) on the unit sphere + country names.
    reverse_geocoder itself compares raw latitude/longitude degrees as if flat (longitude not shrunk by cos(lat)), which
    picked a farther town for 274 of 2,000 test points (10-05); the phone app (FindPicsCore.Geocoder) uses this too."""
    global _GEO
    if _GEO is None:
        import csv
        import reverse_geocoder
        from scipy.spatial import cKDTree
        rows = list(csv.DictReader(open(Path(reverse_geocoder.__file__).parent / "rg_cities1000.csv", encoding="utf-8")))
        lat = np.radians([float(r["lat"]) for r in rows]); lon = np.radians([float(r["lon"]) for r in rows])
        xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        cpath = Path(__file__).resolve().parents[2] / "ios/FindPicsCore/Sources/FindPicsCore/Resources/countries.tsv"
        names = dict(l.split("\t")[:2] for l in cpath.read_text().splitlines() if "\t" in l) if cpath.exists() else {}
        _GEO = (cKDTree(xyz), rows, names)
    return _GEO


def nearest_place(lat: float, lon: float) -> str:
    tree, rows, names = _geo()
    a, b = np.radians(lat), np.radians(lon)
    r = rows[int(tree.query([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])[1])]
    return ", ".join(x for x in (r["name"], r["admin2"], r["admin1"], r["cc"], names.get(r["cc"], "")) if x)


def reverse_geocode(items: list) -> None:
    """Offline: fill `place` for items with GPS but no place text ("Paris, Paris, Ile-de-France, FR, France")."""
    todo = [i for i in items if i.lat is not None and i.lon is not None and not i.place]
    if not todo:
        return
    try:
        _geo()
    except ImportError:
        return
    for it in todo:
        it.place = nearest_place(it.lat, it.lon)


def load_osxphotos_metadata(json_path: Path) -> dict[str, dict]:
    """`osxphotos query --json` output -> {uuid: record}."""
    data = json.loads(Path(json_path).read_text())
    return {r["uuid"]: r for r in data if isinstance(r, dict) and "uuid" in r}


def scan(root: str | os.PathLike, metadata_json: str | None = None) -> list[Item]:
    root = Path(root).resolve()
    meta = load_osxphotos_metadata(Path(metadata_json)) if metadata_json else {}
    items: list[Item] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in IMAGE_EXT:
                media = "photo"
            elif ext in VIDEO_EXT:
                media = "video"
            else:
                continue
            stem = p.stem.removesuffix("_preview")
            rec = meta.get(stem)
            item_id = stem if rec else str(p.relative_to(root))
            persons, labels = [], []
            taken, src, local = None, "mtime", None
            lat = lon = None; place = ""
            if rec:
                taken, src = _parse_dt(rec.get("date")), "metadata_json"
                local = _wall_clock(rec.get("date"))
                if rec.get("latitude") is not None and rec.get("longitude") is not None:
                    lat, lon = float(rec["latitude"]), float(rec["longitude"])
                ps = list(dict.fromkeys(_strings(rec.get("place") or {})))
                place = ", ".join(ps) if ps else str(rec.get("address") or "")
                persons = [x for x in rec.get("persons", []) if x and x != "_UNKNOWN_"]
                labels = list(rec.get("labels", []) or [])
                if rec.get("ismovie"):
                    media = "video"
            if taken is None:
                sc = _sidecar(p)
                if sc:
                    local = sc.get("localTime") or _wall_clock(sc.get("EXIF:DateTimeOriginal")) or \
                        _wall_clock(sc.get("QuickTime:CreationDate"), need_offset=True)
                    for key in ("EXIF:DateTimeOriginal", "QuickTime:CreationDate", "XMP:DateCreated"):
                        if sc.get(key):
                            taken, src = _parse_dt(sc[key]), "sidecar"
                            break
                    if taken is None and isinstance(sc.get("photoTakenTime"), dict):  # Google Takeout
                        taken, src = _parse_dt(int(sc["photoTakenTime"].get("timestamp", 0)) or None), "takeout"
                    geo = sc.get("geoData") or sc.get("geoDataExif") or {}
                    if lat is None and geo.get("latitude") not in (None, 0, 0.0):
                        lat, lon = float(geo["latitude"]), float(geo["longitude"])
                    if not persons:
                        pii = sc.get("XMP:PersonInImage") or []
                        persons = [pii] if isinstance(pii, str) else list(pii)
            if (taken is None or local is None) and media == "photo":
                d = _exif_date(p)            # camera clock = local wall-clock time
                if d:
                    local = local or d.strftime("%Y-%m-%dT%H:%M:%S")
                    if taken is None:
                        taken, src = d, "exif"
            if (taken is None or local is None or lat is None) and media == "video":
                vt, vl, vg = _video_meta(p)
                if taken is None and vt is not None:
                    taken, src = vt, "video_meta"
                local = local or vl
                if lat is None and vg:
                    lat, lon = vg
            if taken is None:
                taken, src = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc), "mtime"
            if lat is None and media == "photo":
                g = _exif_gps(p)
                if g:
                    lat, lon = g
            items.append(Item(item_id, str(p), media, _iso(taken), src, persons, labels, lat=lat, lon=lon, place=place,
                              taken_local=local, camera=_exif_camera(p) if media == "photo" else ""))
    reverse_geocode(items)
    return items


def to_frame(items: list[Item]):
    import pandas as pd
    return pd.DataFrame([asdict(i) for i in items])
