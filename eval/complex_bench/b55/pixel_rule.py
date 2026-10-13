"""RESULTS 55: the pixel veto TESTED on dev and NOT adopted (dev b -.001 [-.005, +.003]); kept here, out of the product,
so veto_eval.py stays reproducible. Thresholds fitted on TRAIN libraries only (fit.py -> thresholds.json) by the rule
pre-registered in JOURNAL 10-13 ~05:15."""
import re

import numpy as np

# RESULTS 55: looks of the WHOLE PHOTO that its pixels state, checked from the pixels as a veto on the judge's yes. On dev
# category b the 9B judge said "blurry" to fog and dusk (eye 0/6 of the extra photos), "overexposed" to ordinary snow
# scenes (0/6), "black and white" to toned and colour photos. Only when the album's question names the look of the photo
# itself ("a black and white photo", "in black and white", "overexposed"), never a thing's colours ("a black and white
# cat" is a cat). Thresholds fitted on TRAIN libraries only (eval/complex_bench/b55/fit.py -> thresholds.json): each keeps
# >= 90-97.5% of the photos the 27B calls so. (Not ported to Swift: not adopted.)
PHOTO_LOOKS = {
    "black_and_white": re.compile(
        r"(?i)\bin black[- ]and[- ]white\b|\bblack[- ]and[- ]white (?:photo|photograph|picture|pic|image|shot|style|film|"
        r"version|print)s?\b|\b(?:photo|photograph|picture|image|shot)s? (?:is |are )?(?:in )?black[- ]and[- ]white\b|"
        r"\bmonochrome\b|\bgr[ae]y ?scale\b"),
    "overexposed": re.compile(r"(?i)\bover-?exposed\b|\bblown[- ]out\b"),
    "blurry": re.compile(r"(?i)\bblurr(?:y|ed)\b|\bout of focus\b|\bmotion blur"),
}
PHOTO_LOOK_NOT = {"black_and_white": re.compile(r"(?i)\bsepia\b")}   # sepia is toned: it HAS colour
BW_MAX_COLOUR = 0.3213     # black and white: at most this fraction of pixels with chroma (max-min channel) > 30 / 255
OVER_MIN_CLIPPED = 0.0037  # overexposed: at least this fraction of pixels with luminance > 250 / 255
BLUR_MAX_GRAD = 13.377     # blurry: mean absolute luminance step between neighbouring pixels at most this (0-255 scale)


def photo_looks(q: str | None) -> set:
    """Which photo-level looks a judge question asks for (RESULTS 55): subset of PHOTO_LOOKS' keys."""
    if not q:
        return set()
    return {k for k, rx in PHOTO_LOOKS.items() if rx.search(q) and not (k in PHOTO_LOOK_NOT and PHOTO_LOOK_NOT[k].search(q))}


def pixel_stats(im) -> dict:
    """Integer-exact pixel statistics of an RGB image (FindPicsCore PixelStats): fraction of pixels with chroma
    max(R,G,B) - min(R,G,B) > 30, and fraction with luminance 0.299 R + 0.587 G + 0.114 B > 250 (of 255)."""
    a = np.asarray(im.convert("RGB") if hasattr(im, "convert") else im, np.int32)
    n = a.shape[0] * a.shape[1]
    chroma = a.max(2) - a.min(2)
    y = 299 * a[..., 0] + 587 * a[..., 1] + 114 * a[..., 2]          # luminance x 1000
    gx = np.abs(np.diff(y, axis=1)).sum() / max(1, y.shape[0] * (y.shape[1] - 1))
    gy = np.abs(np.diff(y, axis=0)).sum() / max(1, (y.shape[0] - 1) * y.shape[1])
    return dict(frac_chroma_gt30=float((chroma > 30).sum()) / n, frac_clip_hi=float((y > 250_000).sum()) / n,
                grad_mean=float(gx + gy) / 2000.0)


def pixel_veto(looks, stats: dict) -> bool:
    """True = the pixels contradict a photo-level look the question asks for, so the photo is not a match."""
    if "black_and_white" in looks and stats["frac_chroma_gt30"] > BW_MAX_COLOUR:
        return True
    if "overexposed" in looks and stats["frac_clip_hi"] < OVER_MIN_CLIPPED:
        return True
    if "blurry" in looks and stats["grad_mean"] > BLUR_MAX_GRAD:
        return True
    return False


