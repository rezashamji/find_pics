"""Walk a photo library (plain folder, Google Takeout, or osxphotos export) and build an item table.

Read-only: files are opened for reading only. Nothing here writes next to the user's photos.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

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
            taken, src = None, "mtime"
            if rec:
                taken, src = _parse_dt(rec.get("date")), "metadata_json"
                persons = [x for x in rec.get("persons", []) if x and x != "_UNKNOWN_"]
                labels = list(rec.get("labels", []) or [])
                if rec.get("ismovie"):
                    media = "video"
            if taken is None:
                sc = _sidecar(p)
                if sc:
                    for key in ("EXIF:DateTimeOriginal", "QuickTime:CreationDate", "XMP:DateCreated"):
                        if sc.get(key):
                            taken, src = _parse_dt(sc[key]), "sidecar"
                            break
                    if taken is None and isinstance(sc.get("photoTakenTime"), dict):  # Google Takeout
                        taken, src = _parse_dt(int(sc["photoTakenTime"].get("timestamp", 0)) or None), "takeout"
                    if not persons:
                        pii = sc.get("XMP:PersonInImage") or []
                        persons = [pii] if isinstance(pii, str) else list(pii)
            if taken is None and media == "photo":
                d = _exif_date(p)
                if d:
                    taken, src = d, "exif"
            if taken is None:
                taken, src = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc), "mtime"
            items.append(Item(item_id, str(p), media, _iso(taken), src, persons, labels))
    return items


def to_frame(items: list[Item]):
    import pandas as pd
    return pd.DataFrame([asdict(i) for i in items])
