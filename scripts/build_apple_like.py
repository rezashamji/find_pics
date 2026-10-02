"""Synthetic export shaped exactly like `docs/MAC_EXPORT.md` produces, built from PUBLIC photos, to test the full Apple
path end to end (scan -> GPU index -> ask) including the formats an iPhone library really has:
  B_all_photos/<year>/<uuid>.jpeg | <uuid>.heic | <uuid>_preview.jpeg   (HEIC + iCloud-preview cases)
  C_videos/<year>/<uuid>.mov                                             (HEVC video, like iPhone)
  library_metadata.json                                                  (osxphotos `query --json` shape)
Person = Kevin Bacon (IMDB, 336 photos). ~Half his photos are "Apple-tagged"; videos are never tagged (Apple's People
album often misses videos), so finding his videos must come from face matching inside sampled frames.
ground_truth.json: {"bacon_photos": [...uuids], "bacon_videos": [...], "other_videos": [...]}.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import av
import numpy as np
import pillow_heif
from PIL import Image

ROOT = Path(os.environ["FP_ROOT"])
TL = ROOT / "data/public/testlib"
OUT = ROOT / "data/public/apple_like_export"
rng = random.Random(7)


def letterbox(im: Image.Image, W=1280, H=720) -> Image.Image:
    im = im.convert("RGB"); im.thumbnail((W, H))
    c = Image.new("RGB", (W, H), (0, 0, 0)); c.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return c


def write_hevc(path: Path, frames: list[Image.Image], sec_per_frame: float = 2.0, fps: int = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mov") as c:
        s = c.add_stream("libx265", rate=fps)
        s.width, s.height, s.pix_fmt = 1280, 720, "yuv420p"
        s.options = {"x265-params": "log-level=error"}
        for im in frames:
            f = av.VideoFrame.from_image(letterbox(im))
            for _ in range(int(sec_per_frame * fps)):
                for pkt in s.encode(f):
                    c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def main():
    meta_tl = {m["uuid"]: m for m in json.load(open(TL / "library_metadata.json"))}
    gt = json.load(open(TL / "ground_truth.json"))
    paths = {p.stem: p for p in (TL / "library").rglob("*") if p.suffix.lower() in (".jpg", ".mp4")}
    bacon = gt["person:Kevin Bacon"]
    others_imdb = [u for u in meta_tl if u.startswith("imdb_") and u not in set(bacon)]
    oi = [u for u in meta_tl if u.startswith("oi_")]
    rng.shuffle(others_imdb); rng.shuffle(oi)
    tagged = {u for u in bacon if meta_tl[u]["persons"]}
    # hold out some untagged Bacon photos to build videos from (so videos contain faces the index must match)
    untagged = [u for u in bacon if u not in tagged]; rng.shuffle(untagged)
    vid_src, photo_bacon = untagged[:12], [u for u in bacon if u not in set(untagged[:12])]
    records, gtruth = [], {"bacon_photos": [], "bacon_videos": [], "other_videos": []}

    def put_photo(u, src_uuid, persons):
        y = meta_tl[src_uuid]["date"][:4]
        d = OUT / "B_all_photos" / y; d.mkdir(parents=True, exist_ok=True)
        im = Image.open(paths[src_uuid]).convert("RGB")
        kind = rng.random()
        if kind < 0.33:   # HEIC (if the user skipped --convert-to-jpeg)
            pillow_heif.from_pillow(im).save(d / f"{u}.heic", quality=85)
        elif kind < 0.5:  # iCloud-only original -> Photos' preview
            im.thumbnail((1024, 1024)); im.save(d / f"{u}_preview.jpeg", quality=85)
        else:
            im.save(d / f"{u}.jpeg", quality=88)
        records.append(dict(uuid=u, date=meta_tl[src_uuid]["date"].replace("T12:00:00", "T12:00:00.000000-04:00"),
                            persons=persons, labels=[], ismovie=False, isphoto=True))

    for i, s in enumerate(photo_bacon):
        u = f"B{i:04d}-BACON"; put_photo(u, s, ["Kevin Bacon"] if s in tagged else ["_UNKNOWN_"] * (i % 3 == 0))
        gtruth["bacon_photos"].append(u)
    for i, s in enumerate(others_imdb[:400] + oi[:1500]):
        put_photo(f"P{i:05d}-OTHER", s, [])
    # videos: 4 with Bacon (3 of his photos each, between scene frames), 6 without
    for k in range(4):
        fr = []
        for s in vid_src[3 * k:3 * k + 3]:
            fr += [Image.open(paths[oi[2000 + rng.randrange(500)]]), Image.open(paths[s])]
        u = f"V{k:03d}-BACONVID"; y = str(2008 + k)
        write_hevc(OUT / "C_videos" / y / f"{u}.mov", fr)
        records.append(dict(uuid=u, date=f"{y}-06-15T18:00:00.000000-04:00", persons=[], labels=[], ismovie=True, isphoto=False))
        gtruth["bacon_videos"].append(u)
    for k in range(6):
        fr = [Image.open(paths[x]) for x in others_imdb[400 + 4 * k:400 + 4 * k + 2] + oi[2600 + 3 * k:2600 + 3 * k + 2]]
        u = f"V{k:03d}-OTHERVID"; y = str(2009 + k)
        write_hevc(OUT / "C_videos" / y / f"{u}.mov", fr)
        records.append(dict(uuid=u, date=f"{y}-07-04T12:00:00.000000-04:00", persons=[], labels=[], ismovie=True, isphoto=False))
        gtruth["other_videos"].append(u)
    (OUT / "library_metadata.json").write_text(json.dumps(records))
    (OUT / "ground_truth.json").write_text(json.dumps(gtruth, indent=1))
    kinds = {}
    for p in OUT.rglob("*"):
        if p.is_file():
            k = "preview" if "_preview" in p.name else p.suffix
            kinds[k] = kinds.get(k, 0) + 1
    print("files by kind:", kinds, "| tagged Bacon photos:", sum("Kevin Bacon" in r["persons"] for r in records),
          "| Bacon photos total:", len(gtruth["bacon_photos"]), "| videos:", 10)


if __name__ == "__main__":
    main()
