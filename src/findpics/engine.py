"""Execute one album spec: filter -> fast scores for ALL in-scope items -> judge the head -> certify the tail.

Fast scores:
  person_score[i] = max cosine between any face in item i and the person's reference faces (-1 if no face found)
  look_score[i]   = max over the item's units of (mean cos to `looks` texts - mean cos to `avoid` texts)
Judge (VLM, P(yes)):
  attribute question on the photo with a red box on the person's best-matching face, and
  identity question on a side-by-side [reference face | photo] image when identity is uncertain.
Certificate: see audit.py. Head = top `head_size` items by fast score, all judged. Tail = the rest, uniformly sampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

from . import audit as A
from .media import load_image, sample_video_frames
from .people import item_person_scores
from .planner import AlbumSpec
from .store import Index, per_item_max
from .vlm import draw_box

IDENTITY_Q = "The left panel shows one person's face. Is that same person visible in the right panel photo?"


@dataclass
class Thresholds:
    person_sure: float = 0.45   # ArcFace cosine: identity accepted without asking the judge (tuned on eval)
    person_maybe: float = 0.20  # below: face treated as someone else (tail items are still identity-judged)
    judge_accept: float = 0.5
    head_size: int = 2000       # judge every item in the head
    tail_budget: int = 1000     # random tail judge calls
    alpha: float = 0.05


@dataclass
class AlbumResult:
    spec: AlbumSpec
    n_scanned: int
    n_in_scope: int
    returned: pd.DataFrame
    cert: dict | None
    report: str
    judged: pd.DataFrame = field(default_factory=pd.DataFrame)


def scope_mask(idx: Index, spec: AlbumSpec) -> np.ndarray:
    it = idx.items
    m = np.ones(len(it), bool)
    if spec.media in ("photo", "video"):
        m &= (it["media"] == spec.media).to_numpy()
    t = pd.to_datetime(it["taken"], utc=True, errors="coerce", format="ISO8601")
    if spec.date_from:
        m &= (t >= pd.Timestamp(spec.date_from, tz="UTC")).to_numpy()
    if spec.date_to:
        m &= (t < pd.Timestamp(spec.date_to, tz="UTC")).to_numpy()
    return m


def look_scores(idx: Index, enc, looks: list[str], avoid: list[str]) -> np.ndarray:
    if not looks and not avoid:
        return np.zeros(idx.n_items, np.float32)
    C = idx.clip
    s = np.zeros(len(C), np.float32)
    if looks:
        s += (C @ enc.texts(looks).T.astype(np.float16)).astype(np.float32).mean(1)
    if avoid:
        s -= (C @ enc.texts(avoid).T.astype(np.float16)).astype(np.float32).mean(1)
    return per_item_max(s, idx.units["item_row"].to_numpy(), idx.n_items)


def _frame_for(idx: Index, item_row: int, face_row: int) -> Image.Image | None:
    r = idx.items.iloc[item_row]
    if r["media"] != "video":
        return load_image(r["path"])
    frames = sample_video_frames(r["path"])
    if not frames:
        return None
    if face_row >= 0:
        t = float(idx.faces.iloc[face_row]["frame_t"])
        return min(frames, key=lambda x: abs(x[0] - t))[1]
    return frames[len(frames) // 2][1]


def _boxed(idx: Index, im: Image.Image, face_row: int) -> Image.Image:
    if face_row < 0:
        return im
    f = idx.faces.iloc[face_row]
    sx, sy = im.width / float(f["img_w"]), im.height / float(f["img_h"])
    return draw_box(im, (f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy))


def reference_crop(idx: Index, face_row: int, pad: float = 0.5) -> Image.Image:
    f = idx.faces.iloc[face_row]
    im = _frame_for(idx, int(f["item_row"]), face_row)
    sx, sy = im.width / float(f["img_w"]), im.height / float(f["img_h"])
    x1, y1, x2, y2 = f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy
    w, h = x2 - x1, y2 - y1
    return im.crop((max(0, x1 - pad * w), max(0, y1 - pad * h), min(im.width, x2 + pad * w), min(im.height, y2 + pad * h)))


def side_by_side(ref: Image.Image, im: Image.Image, H: int = 640) -> Image.Image:
    r = ref.copy(); r.thumbnail((H, H)); i = im.copy(); i.thumbnail((int(H * 1.6), H))
    canvas = Image.new("RGB", (r.width + 24 + i.width, max(r.height, i.height)), (255, 255, 255))
    canvas.paste(r, (0, 0)); canvas.paste(i, (r.width + 24, 0))
    return canvas


def _judge_rows(idx, judge, rows, face_rows, question, ref_img=None, batch=48):
    out = []
    for s in range(0, len(rows), batch):
        ims, ok = [], []
        for r, fr in zip(rows[s:s + batch], face_rows[s:s + batch]):
            try:
                im = _frame_for(idx, int(r), int(fr))
                if im is not None:
                    im = side_by_side(ref_img, im) if ref_img is not None else _boxed(idx, im, int(fr))
            except Exception:
                im = None
            ok.append(im is not None)
            if im is not None:
                ims.append(im)
        ps = iter(judge.p_yes(ims, question) if ims else [])
        out.extend(next(ps) if k else 0.0 for k in ok)
    return np.asarray(out, np.float32)


def run_album(idx: Index, spec: AlbumSpec, enc, judge, refs: np.ndarray | None, ref_face_row: int | None = None,
              th: Thresholds = Thresholds(), seed: int = 0) -> AlbumResult:
    rng = np.random.default_rng(seed)
    scope = scope_mask(idx, spec)
    in_scope = np.where(scope)[0]
    has_look = bool(spec.looks or spec.avoid)
    look = look_scores(idx, enc, spec.looks, spec.avoid)
    person_mode = bool(spec.person) and refs is not None and len(refs) > 0
    if person_mode:
        pscore, best_face = item_person_scores(idx, refs)
        ref_img = reference_crop(idx, int(ref_face_row)) if ref_face_row is not None else None
        # fast score: identity first; appearance breaks ties among plausible identity matches
        fast = pscore + (0.25 * look if has_look else 0.0)
    else:
        pscore = np.full(idx.n_items, np.nan, np.float32); best_face = np.full(idx.n_items, -1)
        ref_img = None
        fast = look
    order = in_scope[np.argsort(-fast[in_scope])]
    head, tail = order[: th.head_size], order[th.head_size:]
    # tail sample fixed BEFORE any label is seen
    n_tail_sample = min(len(tail), th.tail_budget)
    tail_sample = rng.choice(tail, size=n_tail_sample, replace=False) if n_tail_sample else np.array([], int)

    def label(rows):
        rows = np.asarray(rows, int)
        if len(rows) == 0:
            return np.zeros(0, bool), np.zeros(0), np.zeros(0)
        p_attr = _judge_rows(idx, judge, rows, best_face[rows], spec.judge_question) if has_look else np.ones(len(rows))
        if person_mode:
            sure = pscore[rows] >= th.person_sure
            p_id = np.where(sure, 1.0, 0.0).astype(np.float32)
            unsure = np.where(~sure)[0]
            if len(unsure) and ref_img is not None:
                p_id[unsure] = _judge_rows(idx, judge, rows[unsure], best_face[rows[unsure]], IDENTITY_Q, ref_img=ref_img)
            elif len(unsure):
                p_id[unsure] = (pscore[rows[unsure]] >= th.person_maybe).astype(np.float32)
        else:
            p_id = np.ones(len(rows), np.float32)
        y = (p_attr >= th.judge_accept) & (p_id >= th.judge_accept)
        return y, p_attr, p_id

    yh, pah, pih = label(head)
    yt, pat, pit = label(tail_sample)
    judged = pd.concat([
        pd.DataFrame(dict(item_row=head, where="head", y=yh, p_attr=pah, p_id=pih)),
        pd.DataFrame(dict(item_row=tail_sample, where="tail_sample", y=yt, p_attr=pat, p_id=pit))], ignore_index=True)
    judged["item_id"] = idx.items["item_id"].to_numpy()[judged.item_row]
    judged["path"] = idx.items["path"].to_numpy()[judged.item_row]
    judged["person"] = pscore[judged.item_row]
    judged["look"] = look[judged.item_row]
    judged["fast"] = fast[judged.item_row]
    judged["face_row"] = best_face[judged.item_row]
    ret = judged[judged.y].copy()
    ret["reason"] = np.where(ret["where"] == "head", "judge yes", "found by random audit")
    if spec.want == "best":
        ret = ret.sort_values(["p_attr", "fast"], ascending=False)
        if spec.max_items:
            ret = ret.head(spec.max_items)
    cert = None
    if spec.want == "all":
        cert = A.certify(found=int(yh.sum()) + int(yt.sum()), n_tail=len(tail), tail_labels=list(yt), alpha=th.alpha)
    report = _report(spec, idx.n_items, len(in_scope), len(head), ret, cert)
    return AlbumResult(spec, idx.n_items, len(in_scope), ret, cert, report, judged)


def _report(spec, n_all, n_scope, n_head, ret, cert):
    lines = [f"Album '{spec.name}': {len(ret)} items.",
             f"  Library: {n_all:,} items; in scope after date/media filters: {n_scope:,}."]
    if cert:
        lines.append("  " + A.describe(cert, n_scope, n_head))
    else:
        lines.append(f"  'Best-of' request: judge looked at the top {n_head:,}; completeness is not estimated.")
    return "\n".join(lines)
