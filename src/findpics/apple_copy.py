"""Ingest Apple's "Request a copy of your data" iCloud Photos export (privacy.apple.com) without unpacking 2 TB.

The export is a set of zips ("iCloud Photos Part k of N.zip"). Inside: an iCloud Photos folder of ORIGINAL files, a
"Recently Deleted" folder, album / Memories CSVs, "Photo Details*.csv" (imgName, fileChecksum, favorite, hidden, deleted,
originalCreationDate, viewCount, importDate) and a "Shared Albums" zip (other people's photos).
Per zip, streamed member by member (never fully extracted):
  photo  -> decoded (HEIC too), long side shrunk to MAX_SIDE, JPEG with the original EXIF kept (dates, GPS)
  video  -> re-encoded to <= 720p H.264 (the search samples one frame every 2 s at ~1,000 px)
  every file gets a Takeout-style sidecar <file>.json (photoTakenTime, geoData) so `findpics scan` reads it unchanged
Skipped on purpose: Recently Deleted, rows marked deleted, Shared Albums (not the owner's library).
A zip is marked done (<out>/.done/<zip name>) only after all its members are written, so a crash just redoes that zip.
Nothing is ever deleted by this module (product code never deletes; tests/test_safety.py): removing the processed zips
is a manual decision for Reza (they are copies, but they are his photos).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .ingest import IMAGE_EXT, VIDEO_EXT

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

MAX_SIDE = 1600
SKIP_DIRS = re.compile(r"(?i)(^|/)(recently deleted|shared albums?)(/|$)")


def _parse_apple_date(s: str | None) -> datetime | None:
    """Photo Details dates look like 'Tuesday October 1,2019 5:20 PM GMT' (format not documented; try several)."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%A %B %d,%Y %I:%M %p %Z", "%A %B %d, %Y %I:%M %p %Z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        from dateutil import parser
        d = parser.parse(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def read_details(z: zipfile.ZipFile) -> dict[str, dict]:
    """imgName -> row, from every 'Photo Details*.csv' in the zip."""
    rows = {}
    for n in z.namelist():
        if re.search(r"(?i)photo details.*\.csv$", n):
            with z.open(n) as fh:
                for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace")):
                    if r.get("imgName"):
                        rows[r["imgName"]] = r
    return rows


def _out_name(member: str, ext: str) -> str:
    return hashlib.sha1(member.encode()).hexdigest()[:16] + "_" + Path(member).stem[:40] + ext


def _sidecar(dst: Path, taken: datetime | None, gps=None, local: str | None = None):
    sc = {}
    if taken:
        sc["photoTakenTime"] = {"timestamp": str(int(taken.timestamp()))}
    if local:   # wall-clock time where it was taken (for "between 8pm and 11pm"); photos keep it in their EXIF instead
        sc["localTime"] = local
    if gps:
        sc["geoData"] = {"latitude": gps[0], "longitude": gps[1]}
    dst.with_name(dst.name + ".json").write_text(json.dumps(sc))


def _photo(data: bytes, dst: Path) -> None:
    im = Image.open(io.BytesIO(data))
    exif = im.info.get("exif") or (im.getexif().tobytes() if hasattr(im, "getexif") else None)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
    kw = {"exif": exif} if exif else {}
    try:
        im.save(dst, "JPEG", quality=85, **kw)
    except Exception:                       # malformed EXIF block: keep the pixels, lose the EXIF
        im.save(dst, "JPEG", quality=85)


def _iso6709(s: str):
    m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", s or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def _video(src: Path, dst: Path, max_h: int = 720, bitrate: int = 1_500_000):
    """Re-encode to <= max_h lines, H.264, no audio. Returns (creation_time, gps) from the container metadata."""
    import av
    with av.open(str(src)) as inp:
        meta = dict(inp.metadata)
        vs = inp.streams.video[0]
        w, h = vs.codec_context.width, vs.codec_context.height
        scale = min(1.0, max_h / max(1, min(w, h)))
        W, H = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
        rate = vs.average_rate or 30
        with av.open(str(dst), "w") as out:
            os_ = out.add_stream("libx264", rate=rate)
            os_.width, os_.height, os_.pix_fmt, os_.bit_rate = W, H, "yuv420p", bitrate
            for frame in inp.decode(vs):
                for pkt in os_.encode(frame.reformat(width=W, height=H, format="yuv420p")):
                    out.mux(pkt)
            for pkt in os_.encode():
                out.mux(pkt)
    ct = meta.get("creation_time") or meta.get("com.apple.quicktime.creationdate")
    # Apple's creationdate is local time WITH its offset ("2019-07-04T21:03:11-0400"): the wall clock is the first 19 chars
    qc = meta.get("com.apple.quicktime.creationdate") or ""
    local = qc[:19] if re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d", qc) else None
    taken = None
    if ct:
        try:
            taken = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            pass
    gps = _iso6709(meta.get("com.apple.quicktime.location.ISO6709") or meta.get("location") or "")
    return taken, gps, local


def ingest_zip(zpath: Path, out: Path, tmp: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True); tmp.mkdir(parents=True, exist_ok=True)
    n = dict(photos=0, videos=0, skipped_deleted=0, skipped_other=0, errors=0)
    with zipfile.ZipFile(zpath) as z:
        details = read_details(z)
        for info in z.infolist():
            name = info.filename
            if info.is_dir():
                continue
            ext = Path(name).suffix.lower()
            if SKIP_DIRS.search(name) or ext == ".zip":
                n["skipped_deleted" if "deleted" in name.lower() else "skipped_other"] += 1
                continue
            if ext not in IMAGE_EXT and ext not in VIDEO_EXT:
                continue
            row = details.get(Path(name).name, {})
            if str(row.get("deleted", "")).lower() in ("yes", "true", "1"):
                n["skipped_deleted"] += 1
                continue
            taken = _parse_apple_date(row.get("originalCreationDate"))
            try:
                if ext in IMAGE_EXT:
                    dst = out / _out_name(name, ".jpg")
                    if not dst.exists():
                        _photo(z.read(info), dst)
                        _sidecar(dst, taken)
                    n["photos"] += 1
                else:
                    dst = out / _out_name(name, ".mp4")
                    if not dst.exists():
                        t = tmp / ("current_video" + Path(name).suffix)   # overwritten by the next video, never deleted
                        with z.open(info) as src, open(t, "wb") as fh:
                            while chunk := src.read(1 << 24):
                                fh.write(chunk)
                        vt, gps, local = _video(t, dst)
                        _sidecar(dst, taken or vt, gps, local)
                    n["videos"] += 1
            except Exception as e:
                n["errors"] += 1
                (out / "_errors.log").open("a").write(f"{zpath.name}\t{name}\t{type(e).__name__}: {e}\n")
    return n


def ingest_all(zips_dir, out_dir, shard: int = 0, n_shards: int = 1) -> list[dict]:
    """shard/n_shards: zip k goes to shard k % n_shards, so parallel jobs never touch the same zip."""
    zips_dir, out = Path(zips_dir), Path(out_dir)
    done = out / ".done"; done.mkdir(parents=True, exist_ok=True)
    report = []
    for zp in sorted(zips_dir.glob("*.zip"))[shard::n_shards]:
        if (done / zp.name).exists():
            continue
        n = ingest_zip(zp, out / "library", out / f".tmp{shard}")
        (done / zp.name).write_text(json.dumps(n))
        report.append(dict(zip=zp.name, **n))
        print(zp.name, n, flush=True)
    return report
