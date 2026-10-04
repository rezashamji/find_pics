"""Contact sheets: a grid of numbered thumbnails, so a human (or Claude) can actually look at results."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .media import load_image, sample_video_frames, video_frame_at


def thumb(path: str, size: int = 256, t=None) -> Image.Image:
    """t: for a video, the second the search matched (frame_t in results/manifest); None -> middle frame."""
    p = str(path)
    if p.lower().endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp")):
        if t is None:
            fr = sample_video_frames(p, max_frames=3)
            im = fr[len(fr) // 2][1] if fr else None
        else:
            im = video_frame_at(p, t)
        im = im if im is not None else Image.new("RGB", (size, size), (80, 0, 0))
    else:
        im = load_image(p, max_side=size * 2)
    im = im.copy(); im.thumbnail((size, size))
    return im


def sheet(paths, out: str | Path, labels=None, cols: int = 6, size: int = 256, title: str = "") -> Path:
    labels = labels or [str(i) for i in range(len(paths))]
    rows = (len(paths) + cols - 1) // cols
    top = 28 if title else 0
    W, H = cols * (size + 6), rows * (size + 24) + top
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((6, 6), title, fill=(0, 0, 0))
    for i, (p, lab) in enumerate(zip(paths, labels)):
        x, y = (i % cols) * (size + 6), top + (i // cols) * (size + 24)
        try:
            t = thumb(p, size)
        except Exception as e:
            t = Image.new("RGB", (size, size), (200, 200, 200)); lab = f"{lab} ERR"
        canvas.paste(t, (x + (size - t.width) // 2, y + (size - t.height) // 2))
        d.rectangle([x, y + size + 2, x + size, y + size + 22], fill=(255, 255, 255))
        d.text((x + 3, y + size + 5), str(lab)[:40], fill=(0, 0, 0))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=85)
    return out
