"""Numbered review sheets of person albums, for the owner to mark wrong items by eye.

Each tile shows the frame the search decided on (videos: the matched face's frame, not the middle one) with a red box
on the matched face, so "is the boxed person you?" is unambiguous in group photos. Numbers come from key.json when it
exists, so re-drawing never renumbers what the owner may already be marking.

  python scripts/review_sheets.py --index data/private/index_sample --out data/private/review \
      heavier="data/private/sample_runs/demo6/turn_1/me looking heavier" fit="data/private/sample_runs/demo6/turn_1/me looking fit"

Outputs stay in the --out folder (private data: never commit)."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from findpics.media import load_image, video_frame_at
from findpics.store import load
from findpics.vlm import draw_box

TILE, COLS, ROWS = 380, 5, 4
FONT = "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"


def tile(idx, it):
    video = str(it.get("media", "")) == "video" or str(it["path"]).lower().endswith((".mov", ".mp4", ".m4v"))
    fr = int(it.get("face_row", -1) if it.get("face_row") is not None else -1)
    t = it.get("frame_t")
    if video and (t is None or t != t) and fr >= 0:
        t = float(idx.faces.iloc[fr]["frame_t"])
    im = video_frame_at(it["path"], t) if video else load_image(it["path"])
    if im is None:
        return Image.new("RGB", (TILE, TILE), (80, 0, 0)), video
    if fr >= 0:
        f = idx.faces.iloc[fr]
        sx, sy = im.width / float(f["img_w"]), im.height / float(f["img_h"])
        im = draw_box(im, (f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy), width_frac=0.012)
    im.thumbnail((TILE, TILE))
    return im, video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("albums", nargs="+", help="label=album_folder (folder holding manifest.json)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = load(a.index)
    keyp = out / "key.json"
    old = json.loads(keyp.read_text()) if keyp.exists() else {}
    font = ImageFont.truetype(FONT, 30); small = ImageFont.truetype(FONT, 18)
    key = {}
    for spec in a.albums:
        lab, folder = spec.split("=", 1)
        items = json.loads((Path(folder) / "manifest.json").read_text())["items"]
        by_id = {str(i["item_id"]): i for i in items}
        order = [k["item_id"] for k in old.get(lab, []) if k["item_id"] in by_id]
        order += [i for i in by_id if i not in set(order)]
        key[lab] = [dict(n=n + 1, item_id=i, path=by_id[i]["path"]) for n, i in enumerate(order)]
        with ThreadPoolExecutor(8) as ex:
            tiles = list(ex.map(lambda i: tile(idx, by_id[i]), order))
        per = COLS * ROWS
        for s in range(0, len(order), per):
            sheet = Image.new("RGB", (COLS * (TILE + 8), ROWS * (TILE + 8)), (30, 30, 30))
            d = ImageDraw.Draw(sheet)
            for j, (im, video) in enumerate(tiles[s:s + per]):
                x, y = (j % COLS) * (TILE + 8), (j // COLS) * (TILE + 8)
                sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
                n = s + j + 1
                d.rectangle([x, y, x + 64, y + 40], fill=(0, 0, 0)); d.text((x + 6, y + 3), str(n), fill=(255, 230, 0), font=font)
                if video:
                    d.rectangle([x, y + TILE - 28, x + 80, y + TILE], fill=(0, 0, 0))
                    d.text((x + 6, y + TILE - 26), "VIDEO", fill=(255, 255, 255), font=small)
            sheet.save(out / f"{lab}_{s // per + 1:02d}.jpg", quality=88)
    keyp.write_text(json.dumps(key, indent=1))
    print({k: len(v) for k, v in key.items()})


if __name__ == "__main__":
    main()
