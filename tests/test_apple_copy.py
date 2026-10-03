"""Apple data-copy ingester on a synthetic export zip (no network, no real photos)."""
import csv
import io
import json
import zipfile

import numpy as np
from PIL import Image

from findpics.apple_copy import _parse_apple_date, ingest_all
from findpics.ingest import scan


def _jpeg(color, date="2019:10:01 17:20:00", gps=True, size=(3000, 2000)):
    im = Image.new("RGB", size, color)
    ex = im.getexif(); ex[0x0132] = date
    exif_ifd = ex.get_ifd(0x8769); exif_ifd[0x9003] = date
    if gps:
        g = ex.get_ifd(0x8825)
        g.update({1: "N", 2: (42.0, 21.0, 0.0), 3: "W", 4: (71.0, 6.0, 0.0)})
    b = io.BytesIO(); im.save(b, "JPEG", exif=ex.tobytes()); return b.getvalue()


def _video_bytes(tmp_path):
    import av
    p = tmp_path / "src.mov"
    with av.open(str(p), "w") as c:
        s = c.add_stream("libx264", rate=10); s.width, s.height, s.pix_fmt = 1920, 1080, "yuv420p"
        for i in range(10):
            f = av.VideoFrame.from_ndarray(np.full((1080, 1920, 3), i * 20, np.uint8), format="rgb24")
            for pkt in s.encode(f.reformat(format="yuv420p")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return p.read_bytes()


def test_ingest_export_zip(tmp_path):
    zdir = tmp_path / "zips"; zdir.mkdir()
    rows = io.StringIO(); w = csv.writer(rows)
    w.writerow(["imgName", "fileChecksum", "favorite", "hidden", "deleted", "originalCreationDate", "viewCount", "importDate"])
    w.writerow(["IMG_0001.JPG", "x", "no", "no", "no", "Tuesday October 1,2019 5:20 PM GMT", "1", ""])
    w.writerow(["IMG_0002.PNG", "y", "no", "no", "yes", "Tuesday October 1,2019 5:21 PM GMT", "1", ""])
    with zipfile.ZipFile(zdir / "iCloud Photos Part 1 of 1.zip", "w") as z:
        z.writestr("iCloud Photos/Photos/IMG_0001.JPG", _jpeg("red"))
        z.writestr("iCloud Photos/Photos/IMG_0002.PNG", _jpeg("blue", gps=False))              # marked deleted in CSV
        z.writestr("iCloud Photos/Photos/IMG_0003.MOV", _video_bytes(tmp_path))
        z.writestr("iCloud Photos/Recently Deleted/IMG_0009.JPG", _jpeg("green"))
        z.writestr("iCloud Photos/Shared Albums.zip", b"PK")
        z.writestr("iCloud Photos/Photo Details.csv", rows.getvalue())
    out = tmp_path / "out"
    rep = ingest_all(zdir, out)
    assert rep[0]["photos"] == 1 and rep[0]["videos"] == 1 and rep[0]["skipped_deleted"] == 2 and rep[0]["errors"] == 0
    lib = out / "library"
    jpg = next(lib.glob("*_IMG_0001.jpg")); im = Image.open(jpg)
    assert max(im.size) == 1600                                    # shrunk, aspect kept
    vid = next(lib.glob("*_IMG_0003.mp4"))
    import av
    with av.open(str(vid)) as c:
        assert min(c.streams.video[0].codec_context.height, c.streams.video[0].codec_context.width) == 720
    items = {i.media: i for i in scan(lib)}
    assert items["photo"].taken.startswith("2019-10-01T17:20") and items["photo"].lat is not None   # CSV date + EXIF GPS kept
    assert ingest_all(zdir, out) == []                               # done markers: a re-run does nothing
    assert (zdir / "iCloud Photos Part 1 of 1.zip").exists()        # never deleted


def test_apple_date_formats():
    assert _parse_apple_date("Tuesday October 1,2019 5:20 PM GMT").year == 2019
    assert _parse_apple_date("") is None
