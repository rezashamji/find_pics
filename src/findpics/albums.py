"""Write results as albums. Two backends:

1. Folder album (any OS): <out>/<album name>/ with symlinks to the originals + manifest.json. Never copies or moves
   originals; deleting the folder later only deletes links.
2. Apple Photos (macOS only): creates a NEW album and adds items by Photos uuid using osxphotos' PhotosAlbum
   (AppleScript under the hood). Additive only. Dry-run unless apply=True.

Safety: this module contains no delete/remove/move calls on library items, and a test greps for them.
"""
from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path

import pandas as pd


def _safe(name: str) -> str:
    return re.sub(r"[^\w\- .]+", "_", name).strip() or "album"


def write_folder_album(name: str, rows: pd.DataFrame, out_dir: str | Path, report: str = "") -> Path:
    d = Path(out_dir) / _safe(name)
    d.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, r in enumerate(rows.itertuples(index=False)):
        src = Path(r.path)
        link = d / f"{i:05d}_{src.name}"
        if not link.exists():
            os.symlink(src, link)
        manifest.append({k: (v.item() if hasattr(v, "item") else v) for k, v in r._asdict().items()})
    (d / "manifest.json").write_text(json.dumps(dict(album=name, report=report, items=manifest), indent=1, default=str))
    return d


def write_apple_album(name: str, uuids: list[str], apply: bool = False) -> str:
    """Create album `name` in Apple Photos and add `uuids`. Returns a description of what was (or would be) done."""
    plan = f"Apple Photos: create album '{name}' and add {len(uuids)} items (no other changes)."
    if not apply:
        return "[dry-run] " + plan + " Re-run with --apply to do it."
    if platform.system() != "Darwin":
        raise RuntimeError("Apple Photos albums can only be written on macOS.")
    import osxphotos
    from osxphotos.photosalbum import PhotosAlbum
    db = osxphotos.PhotosDB()
    photos = db.photos(uuid=list(uuids))
    album = PhotosAlbum(name)  # creates the album if it doesn't exist
    album.extend(photos)       # add only
    return plan + f" Done ({len(photos)} found in library)."


def remove_album_link(link: Path) -> None:
    """Remove one LINK from a find_pics album folder. Refuses anything that is not a symlink inside a folder holding a
    find_pics manifest.json, so it cannot delete a photo even if called with the wrong path."""
    link = Path(link)
    if not link.is_symlink():
        raise ValueError(f"refusing: {link} is not a symlink (find_pics never deletes real files)")
    if not (link.parent / "manifest.json").exists():
        raise ValueError(f"refusing: {link.parent} is not a find_pics album folder")
    os.unlink(link)  # SAFE-LINK-ONLY: symlink inside an album folder, checked above
