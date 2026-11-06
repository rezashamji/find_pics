"""M82 (B') PARITY-EXACT writer inputs: rebuild the 20 PNGs from their Commons URLs and print their hashes.

Why: the phone's #writerparity read the 500/330/250 px Commons thumbnails and resampled them itself (CoreImage: the
448 best-fit, then the Qwen3-VL processor's smart_resize to multiples of 32), while the cluster resampled with Pillow
twice. Here every photo is resized ONCE, with Pillow BICUBIC, straight to the exact (W, H) the Qwen3-VL processor
would produce, so neither side resamples again: the phone's 448x448 best-fit scale is then 1 (W or H is 448 or less
and the other side fits) and its smart_resize maps (H, W) to itself. The PNG carries no metadata: no ICC profile
(15 of the 20 thumbnails embed Display P3 / Apple Wide Color profiles, which CoreImage would colour-manage while Pillow
reads the raw values), no gamma, no dpi, no EXIF - plain 8-bit RGB.

  python3 parity_exact_recipe.py [--json=writer_parity_exact.json] [--out=DIR]
Needs Python 3 + Pillow; nothing else (downloads with urllib). Prints per photo: thumbnail sha256 (must match),
(W, H), sha256 of the RGB pixel bytes (the real check: identical pixels), sha256 of the PNG file (zlib can differ
between builds, so a PNG-file mismatch with a pixel match is harmless).
The cluster built its PNGs with build_png() below (Pillow version in writer_parity_exact.json "recipe").
"""
import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

UA = "find_pics-research/0.1 (device anchor eval; rezamshamji@gmail.com) python-urllib"


def fetch(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network blip / 429
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)


def build_png(thumb_bytes, W, H):
    """Thumbnail JPEG bytes -> (png bytes, sha256 of the RGB pixel bytes). One resize, Pillow BICUBIC, no metadata."""
    from PIL import Image
    im = Image.open(io.BytesIO(thumb_bytes))
    im = im.convert("RGB")                                   # raw decoded values; the ICC profile is NOT applied
    if im.size != (W, H):
        im = im.resize((W, H), resample=Image.BICUBIC)
    clean = Image.frombytes("RGB", im.size, im.tobytes())   # a fresh image: drops info (icc_profile, dpi, exif)
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return buf.getvalue(), hashlib.sha256(clean.tobytes()).hexdigest()


def main():
    import PIL
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    J = json.loads(Path(arg("json", str(Path(__file__).with_name("writer_parity_exact.json")))).read_text())
    out = Path(arg("out", "parity_exact_png")); out.mkdir(parents=True, exist_ok=True)
    print(f"Pillow {PIL.__version__} (cluster: {J['recipe']['pillow']})")
    n_thumb = n_pix = n_png = 0
    for i, p in enumerate(J["photos"]):
        b = fetch(p["url"])
        t_ok = hashlib.sha256(b).hexdigest() == p["thumb_sha256"]
        png, pix = build_png(b, p["W"], p["H"])
        (out / f"{i:02d}.png").write_bytes(png)
        png_ok, pix_ok = hashlib.sha256(png).hexdigest() == p["png_sha256"], pix == p["pixel_sha256"]
        n_thumb += t_ok; n_pix += pix_ok; n_png += png_ok
        print(f"{i:02d} {p['W']}x{p['H']} thumb {'OK' if t_ok else 'MISMATCH'} pixels {pix[:16]} "
              f"{'OK' if pix_ok else 'MISMATCH'} png {hashlib.sha256(png).hexdigest()[:16]} {'OK' if png_ok else 'differs'}")
        time.sleep(0.5)
    print(f"thumbnails {n_thumb}/{len(J['photos'])}, pixels {n_pix}/{len(J['photos'])}, png files {n_png}/{len(J['photos'])}")


if __name__ == "__main__":
    main()
