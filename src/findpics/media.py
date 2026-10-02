"""Decode photos (incl. HEIC) and sample frames from videos. Read-only."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # HEIC support optional
    pass

MAX_SIDE = 1600  # downscale huge originals before models see them


def load_image(path: str, max_side: int = MAX_SIDE) -> Image.Image:
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))  # JPEG: decode directly at reduced scale (>= max_side), several x faster
        im = ImageOps.exif_transpose(im)  # respect camera rotation
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.BICUBIC)
        return im.copy()


def sample_video_frames(path: str, every_s: float = 2.0, max_frames: int = 40, max_side: int = MAX_SIDE):
    """Return [(t_seconds, PIL.Image)] sampled every `every_s` seconds (evenly thinned to max_frames)."""
    import av
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else 0.0)
        if dur <= 0:
            dur = 1.0
        n = max(1, min(max_frames, int(dur // every_s) + 1))
        targets = list(np.linspace(0, max(dur - 0.05, 0), n))
        ti = 0
        for frame in c.decode(s):
            if frame.time is None:
                continue
            if frame.time + 1e-3 >= targets[ti]:
                im = frame.to_image()
                if max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.BICUBIC)
                out.append((float(frame.time), im))
                ti += 1
                if ti >= len(targets):
                    break
    return out
