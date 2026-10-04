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

import os
import re
from concurrent.futures import ThreadPoolExecutor
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
    person_accept: float = 0.40  # face cosine to the person's references: identity accepted (testlib: 324/325 precise at 0.45)
    person_maybe: float = 0.30   # [maybe, accept) -> "possible" list for the user to confirm; never auto-accepted
    head_chunk: int = 200       # concept queries: judge the head in chunks...
    head_max: int = 6000        # ...and keep extending while the last chunk's yes-rate >= head_stop_rate
    head_stop_rate: float = 0.03
    judge_accept: float = 0.7   # object/scene questions (0.5->0.7: dog keeps 1560 vs 1566 of 1586, bread 216 vs 225 of 264; removes the mostly-wrong 0.5-0.7 band seen on "dog on a beach")
    attr_accept: float = 0.3    # (legacy absolute cut; person-appearance albums now use rel_cut, see below)
    rel_cut: float = 0.5        # person-appearance albums: keep photos in the top (1-rel_cut) of THIS person's own photos.
    # Why relative: the judge's absolute scale differs by person (Jonah Hill's lean-era mean P(heavier) 0.63 > Chris
    # Pratt's heavy-era 0.47). Fixed cut 0.3 + raw-score exclusivity put 1/52 of Pratt's heavy-era photos in 'heavier';
    # within-person ranks put 78/114 heavy-era photos (3 people) in 'heavier' with 29/131 lean-era leaking in.
    head_size: int = 600        # minimum head (concept queries)
    tail_budget: int = 1000     # random tail judge calls
    alpha: float = 0.05
    stream: bool = False        # keep judging after the first answer: head doubles each round until every item is judged


@dataclass
class AlbumResult:
    spec: AlbumSpec
    n_scanned: int
    n_in_scope: int
    returned: pd.DataFrame
    cert: dict | None
    report: str
    judged: pd.DataFrame = field(default_factory=pd.DataFrame)


def local_clock(items: pd.DataFrame) -> pd.Series:
    """Wall-clock capture time where each item was taken (NaT when unknown). Indexes scanned before taken_local existed:
    EXIF / Flickr-style / sidecar-EXIF dates were stored as naive local times labelled UTC, so their clock is usable;
    Takeout/Apple-CSV instants and file mtimes are true UTC and are NOT a local clock."""
    if "taken_local" in items:
        loc = pd.to_datetime(items["taken_local"], errors="coerce", format="ISO8601")
    else:
        loc = pd.Series(pd.NaT, index=items.index, dtype="datetime64[ns]")
    if "date_source" in items:
        old = loc.isna() & items["date_source"].isin(["exif", "metadata_json", "sidecar"])
        if old.any():
            t = pd.to_datetime(items.loc[old, "taken"], utc=True, errors="coerce", format="ISO8601").dt.tz_localize(None)
            loc = loc.copy(); loc[old] = t
    # date-only metadata gets a placeholder clock (12:00:00 or 00:00:00: both public test libraries use noon for every
    # photo); a placeholder is not a time of day, so it is unknown rather than "noon"
    ph = loc.notna() & (loc.dt.second == 0) & (loc.dt.minute == 0) & loc.dt.hour.isin([0, 12])
    loc = loc.mask(ph)
    return loc


def time_of_day_mask(items: pd.DataFrame, rng: str) -> np.ndarray:
    """'20:00-23:00' (or wrapping '22:00-04:00'): local clock inside the range. Items with no local clock are OUT
    (and counted in the report), never guessed."""
    a, b = [int(x[:2]) * 60 + int(x[3:5]) for x in rng.split("-")]
    loc = local_clock(items)
    mins = (loc.dt.hour * 60 + loc.dt.minute).to_numpy()
    known = ~np.isnan(mins)
    inside = (mins >= a) & (mins < b) if a < b else (mins >= a) | (mins < b)
    return known & inside


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
    if getattr(spec, "time_of_day", None):
        m &= time_of_day_mask(it, spec.time_of_day)
    if getattr(spec, "place", None) and "place" in it:
        # all words of the place name must appear in the item's place text (Apple names, or offline-geocoded GPS)
        words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", spec.place.lower()).split() if len(w) > 1]
        txt = it["place"].fillna("").str.lower()
        pm = np.ones(len(it), bool)
        for w in words:
            pm &= txt.str.contains(rf"\b{re.escape(w)}\b", regex=True).to_numpy()
        m &= pm
    return m


def look_scores(idx: Index, enc, looks: list[str], avoid: list[str]) -> np.ndarray:
    if not looks and not avoid:
        return np.zeros(idx.n_items, np.float32)
    if getattr(idx, "clip_model", None) and idx.clip_model != enc.name:
        raise ValueError(f"text encoder {enc.name} != index vectors {idx.clip_model}")
    C = idx.clip
    s = np.zeros(len(C), np.float32)
    if looks:
        T = enc.texts(looks).T.astype(np.float16)
        s += (C @ T).astype(np.float32).mean(1)
        CT = getattr(idx, "clip_tiles", None)
        if CT is not None and len(CT) == len(C):     # best tile catches small objects the whole-photo vector misses
            st = (CT @ T).astype(np.float32).mean(2).max(1)
            s = np.maximum(s, st)
    if avoid:
        s -= (C @ enc.texts(avoid).T.astype(np.float16)).astype(np.float32).mean(1)
    # remember which sampled frame of each video matched best, so the judge sees THAT frame, not an arbitrary one
    u = pd.DataFrame({"item_row": idx.units["item_row"].to_numpy(), "s": s, "t": idx.units["frame_t"].to_numpy()})
    best = u.loc[u.groupby("item_row")["s"].idxmax()]
    idx.best_frame_t = dict(zip(best.item_row.astype(int), best.t.astype(float)))
    # the best VIDEO_FRAMES frames per video (see VIDEO_FRAMES for why it is 1)
    top = u.sort_values("s", ascending=False).groupby("item_row").head(VIDEO_FRAMES)
    idx.top_frames_t = top.groupby("item_row")["t"].apply(lambda x: [float(v) for v in x]).to_dict()
    return per_item_max(s, idx.units["item_row"].to_numpy(), idx.n_items)


VIDEO_FRAMES = 1   # 3 was tried (10-03): it raised "videos with any yes frame" 211 -> 269 of 312, but a full-res look at
                   # 8 newly found videos: 5 judge false positives, 1 inconsistency, 2 real only loosely -> mostly extra
                   # chances for the judge to cross 0.7, not real matches. Kept at 1; mechanism stays for future tests.


def _frames_for(idx: Index, item_row: int, face_row: int) -> list:
    """Images the judge sees for one item: a photo -> [photo]; a video with a matched face -> [that frame]; any other
    video -> its best VIDEO_FRAMES frames by the cheap score (the judge's answer for the video = max over them)."""
    ts = getattr(idx, "top_frames_t", {}).get(int(item_row))
    if face_row >= 0 or not ts or len(ts) < 2 or idx.items.iloc[item_row]["media"] != "video":
        im = _frame_for(idx, item_row, face_row)
        return [] if im is None else [im]
    frames = sample_video_frames(idx.items.iloc[item_row]["path"])
    return [min(frames, key=lambda x: abs(x[0] - t))[1] for t in ts] if frames else []


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
    t = getattr(idx, "best_frame_t", {}).get(int(item_row))
    if t is not None and t >= 0:
        return min(frames, key=lambda x: abs(x[0] - t))[1]
    return frames[len(frames) // 2][1]


def _frame_ts(idx: Index, rows, face_rows) -> np.ndarray:
    """Second of each video the result was decided on (the matched face's frame, else the best-scoring frame);
    NaN for photos. Previews and the full view show this frame, so a reviewer sees what the search saw."""
    out = np.full(len(rows), np.nan)
    media = idx.items["media"].to_numpy()
    bt = getattr(idx, "best_frame_t", {})
    for k, (r, f) in enumerate(zip(rows, face_rows)):
        if media[int(r)] != "video":
            continue
        if f >= 0:
            out[k] = float(idx.faces.iloc[int(f)]["frame_t"])
        elif int(r) in bt:
            out[k] = float(bt[int(r)])
    return out


def person_crop(idx: Index, im: Image.Image, face_row: int) -> Image.Image:
    """Crop to one person: face box widened 1.6x each side, from just above the head to ~4.5 face-heights below
    (head + torso), so an appearance question is about THIS person, not whoever else is in a group photo.
    (Measured need: on the 'Cage heavier' demo, top-scored photos were driven by other people in the frame.)"""
    f = idx.faces.iloc[face_row]
    sx, sy = im.width / float(f["img_w"]), im.height / float(f["img_h"])
    x1, y1, x2, y2 = f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy
    w, h = x2 - x1, y2 - y1
    box = (max(0, x1 - 1.6 * w), max(0, y1 - 0.6 * h), min(im.width, x2 + 1.6 * w), min(im.height, y2 + 4.5 * h))
    return im.crop(tuple(int(v) for v in box))


def person_crop_boxed(idx: Index, im: Image.Image, face_row: int) -> Image.Image:
    """person_crop + a red box around THIS person's face inside the crop: in tight group shots the crop still holds
    neighbours, and the judge rated a heavier neighbour instead (Reza's sample 10-04: 2/8 sampled 'heavier' photos)."""
    f = idx.faces.iloc[face_row]
    sx, sy = im.width / float(f["img_w"]), im.height / float(f["img_h"])
    x1, y1, x2, y2 = f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy
    w, h = x2 - x1, y2 - y1
    cx, cy = max(0, x1 - 1.6 * w), max(0, y1 - 0.6 * h)
    crop = person_crop(idx, im, face_row)
    return draw_box(crop, (x1 - cx, y1 - cy, x2 - cx, y2 - cy))


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


_WORKERS = ThreadPoolExecutor(max(2, min(16, len(os.sched_getaffinity(0)))))   # cores Slurm gave us
_PREFETCH = ThreadPoolExecutor(1)


def _judge_rows(idx, judge, rows, face_rows, question, ref_img=None, batch=96, crop_person=False):
    """Judge P(yes) per row. Photos for the NEXT batch are decoded on all allocated cores while the GPU judges this one
    (measured 10-02: serial decode 46 ms/photo vs ~52 ms/photo end to end, i.e. the GPU mostly waited on the CPU)."""
    def one(im, fr):
        if ref_img is not None:
            return side_by_side(ref_img, im)
        if crop_person == "box" and int(fr) >= 0:
            return person_crop_boxed(idx, im, int(fr))
        if crop_person and int(fr) >= 0:
            return person_crop(idx, im, int(fr))
        return _boxed(idx, im, int(fr))

    def prep(r, fr):
        try:
            return [one(im, fr) for im in _frames_for(idx, int(r), int(fr))]
        except Exception:
            return []

    def load(s):
        return list(_WORKERS.map(prep, rows[s:s + batch], face_rows[s:s + batch]))

    out = []
    nxt = _PREFETCH.submit(load, 0) if len(rows) else None
    for s in range(0, len(rows), batch):
        got = nxt.result()
        nxt = _PREFETCH.submit(load, s + batch) if s + batch < len(rows) else None
        flat = [im for ims in got for im in ims]
        ps = iter(judge.p_yes(flat, question) if flat else [])
        out.extend(max((next(ps) for _ in ims), default=0.0) for ims in got)   # a video: max over its frames
    return np.asarray(out, np.float32)


def run_album(idx: Index, spec: AlbumSpec, enc, judge, refs: np.ndarray | None, ref_face_row: int | None = None,
              th: Thresholds = Thresholds(), seed: int = 0) -> AlbumResult:
    """The first answer (see stream_album)."""
    return next(stream_album(idx, spec, enc, judge, refs, ref_face_row, th, seed))


def stream_album(idx: Index, spec: AlbumSpec, enc, judge, refs: np.ndarray | None, ref_face_row: int | None = None,
                 th: Thresholds = Thresholds(), seed: int = 0, fast_override: np.ndarray | None = None, ref_img=None):
    """Yields AlbumResults: the first answer, then (if th.stream, object/scene albums asking for "all") a better one
    after each round, until the judge has looked at every in-scope item.

    Two regimes.

    Person albums: identity comes ONLY from face vectors (the VLM is not a face recognizer: on the test library its
    side-by-side identity answers were right 4 times out of 191 yeses). Items with face similarity >= person_accept are
    the person; [person_maybe, person_accept) is a counted "possible" list for the user. If the album also has an
    appearance condition, the judge rates EVERY identity-accepted item (no sampling needed: the set is small).
    Identity completeness cannot be certified by a machine judge; the report says so and the review page shows a
    random sample of the rest for a human.

    Object/scene albums: rank everything by the fast score, judge the head in chunks while it keeps finding matches,
    then a uniform random tail sample -> Clopper-Pearson certificate (relative to the judge).

    Streaming: round k>=2 doubles the judged head (highest fast scores first, so most matches arrive early) and draws a
    fresh uniform sample of what is still unjudged. The person may stop at ANY round, including because the bound looks
    good, so the error budget is split in advance (union bound): round 1 gets alpha/2, round k>=2 gets
    alpha/(2 n_later), n_later = number of later rounds with an unjudged tail (fixed by the doubling schedule once round 1
    is done); these sum to alpha, so every bound shown holds simultaneously. The random sample doubles each round
    (no extra total cost: every item is judged exactly once by the end).

    Reference subjects that are not faces ("my dog Max", "my bike"): fast_override = similarity of each item's image
    vectors to the reference photos, and ref_img makes the judge compare [reference | candidate] side by side.
    """
    rng = np.random.default_rng(seed)
    scope = scope_mask(idx, spec)
    in_scope = np.where(scope)[0]
    has_look = bool((spec.looks or spec.avoid) and spec.judge_question)
    if fast_override is not None:
        look = np.asarray(fast_override, np.float32)
    else:
        look = look_scores(idx, enc, spec.looks, spec.avoid) if has_look or not spec.person else np.zeros(idx.n_items, np.float32)
    person_mode = bool(spec.person) and refs is not None and len(refs) > 0
    if not person_mode and not spec.judge_question and fast_override is None:
        # no condition at all ("all photos from that week", "all my 2019 videos"): every in-scope item, no judge calls,
        # complete by construction (DISBench q3/q4 crashed here: the judge was asked a None question)
        d = pd.DataFrame(dict(item_row=in_scope, where="all_in_scope", y=True, p_attr=1.0, rel=np.nan))
        d["item_id"] = idx.items["item_id"].to_numpy()[d.item_row]; d["path"] = idx.items["path"].to_numpy()[d.item_row]
        d["person"] = np.nan; d["look"] = 0.0; d["fast"] = 0.0; d["face_row"] = -1; d["reason"] = "in scope"
        cert = A.certify(found=len(d), n_tail=0, tail_labels=[], alpha=th.alpha)
        rep = (f"Album '{spec.name}': {len(d)} items.\n  Every item matching the dates/place/media/moment you gave "
               f"({len(d):,} of {idx.n_items:,}); no visual condition, so nothing needed judging.")
        res = AlbumResult(spec, idx.n_items, len(in_scope), d.copy(), cert, rep, d); res.possible = pd.DataFrame()
        yield res
        return
    if person_mode:
        pscore, best_face = item_person_scores(idx, refs)
        fast = pscore + (0.25 * look if has_look else 0.0)
    else:
        pscore = np.full(idx.n_items, np.nan, np.float32); best_face = np.full(idx.n_items, -1)
        fast = look

    def frame(rows, where, y, p_attr, rel=None):
        d = pd.DataFrame(dict(item_row=np.asarray(rows, int), where=where, y=y, p_attr=p_attr,
                              rel=rel if rel is not None else np.full(len(rows), np.nan)))
        d["item_id"] = idx.items["item_id"].to_numpy()[d.item_row]; d["path"] = idx.items["path"].to_numpy()[d.item_row]
        d["person"] = pscore[d.item_row]; d["look"] = look[d.item_row]; d["fast"] = fast[d.item_row]
        d["face_row"] = best_face[d.item_row]
        d["frame_t"] = _frame_ts(idx, d.item_row.to_numpy(), d.face_row.to_numpy())
        return d

    possible = pd.DataFrame()
    cert = None
    if person_mode:
        ident = in_scope[pscore[in_scope] >= th.person_accept]
        maybe = in_scope[(pscore[in_scope] >= th.person_maybe) & (pscore[in_scope] < th.person_accept)]
        if has_look:
            # crop around THIS person + a red box on their face, and ask about the person in the box. Crop alone let the
            # judge rate a neighbour in tight group shots: Reza's sample (10-04), 'heavier' 2023-vs-2026 AUC on photos
            # with 2+ faces 0.571 -> 0.792 (all photos 0.84 -> 0.925); disagreements checked by eye (8/8 box right).
            q = re.sub(r"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)",
                       "the person in the red box", spec.judge_question)
            p_id = _judge_rows(idx, judge, ident, best_face[ident], q, crop_person="box")
            p_mb = _judge_rows(idx, judge, maybe, best_face[maybe], q, crop_person="box") if len(maybe) else np.zeros(0)
            # rank within this person's own photos (fraction of their photos scoring at or below this one)
            srt = np.sort(p_id)
            rel_id = np.searchsorted(srt, p_id, side="right") / max(len(srt), 1)
            rel_mb = np.searchsorted(srt, p_mb, side="right") / max(len(srt), 1)
            y_id = rel_id > th.rel_cut; y_mb = rel_mb > th.rel_cut
        else:
            p_id = np.ones(len(ident), np.float32); p_mb = np.ones(len(maybe), np.float32)
            rel_id = rel_mb = None
            y_id = np.ones(len(ident), bool); y_mb = np.ones(len(maybe), bool)
        judged = frame(ident, "identity_match", y_id, p_id, rel_id)
        possible = frame(maybe, "possible_identity", y_mb, p_mb, rel_mb)
        possible = possible[possible.y]
        rest = np.setdiff1d(in_scope, np.r_[ident, maybe])
        audit_rows = rng.choice(rest, size=min(len(rest), 200), replace=False) if len(rest) else np.zeros(0, int)
        judged = pd.concat([judged, frame(audit_rows, "human_audit_sample", np.zeros(len(audit_rows), bool),
                                          np.zeros(len(audit_rows)))], ignore_index=True)
        ret = judged[judged.y & (judged["where"] == "identity_match")].copy()
        ret["reason"] = "face match" + (" + judge yes" if has_look else "")
        n_head = len(ident) + len(maybe) if has_look else 0
    else:
        order = in_scope[np.argsort(-fast[in_scope])]
        seen: dict[int, float] = {}       # item row -> judge P(yes): every item is judged at most once

        def pj(rows):
            new = np.asarray([r for r in rows if int(r) not in seen], int)
            if len(new):
                seen.update(zip(map(int, new), map(float, _judge_rows(idx, judge, new, best_face[new], spec.judge_question,
                                                                      ref_img=ref_img))))
            return np.asarray([seen[int(r)] for r in rows], np.float32)

        n_head = min(th.head_size, len(order))
        pj(order[:n_head])
        while n_head < min(th.head_max, len(order)):
            if (pj(order[max(0, n_head - th.head_chunk):n_head]) >= th.judge_accept).mean() < th.head_stop_rate:
                break
            n_head = min(n_head + th.head_chunk, len(order)); pj(order[:n_head])
        streaming = th.stream and spec.want == "all"
        # rounds after the first that still have an unjudged tail: known once round 1's head is fixed (the doubling
        # schedule depends only on that), so alpha/2 can be split EVENLY over them before any of their samples are drawn
        h, n_later = n_head, 0
        while (h := min(len(order), 2 * h)) < len(order):
            n_later += 1
        perm, k = None, 0
        while True:
            k += 1
            head, tail = order[:n_head], order[n_head:]
            # later rounds sample more of the tail: free in total (everything gets judged once anyway), and it is what
            # lets the bound tighten as the search goes on instead of drifting down
            n_t = min(len(tail), th.tail_budget * 2 ** (k - 1))
            if k == 1:
                ts = rng.choice(tail, size=n_t, replace=False) if n_t else np.zeros(0, int)   # fixed before any tail label
            else:   # first n_t still-unjudged items of a random order fixed in advance = uniform sample of the new tail
                perm = np.random.default_rng(seed + 1).permutation(order) if perm is None else perm
                ts = perm[np.isin(perm, tail)][:n_t]
            ph = pj(head); pt = pj(ts) if n_t else np.zeros(0, np.float32)
            yh = ph >= th.judge_accept; yt = pt >= th.judge_accept
            # matches found by an EARLIER round's random sample stay found (they are in the tail now, outside this sample)
            prev = np.setdiff1d([r for r, p in seen.items() if p >= th.judge_accept], np.r_[head, ts]).astype(int)
            judged = pd.concat([frame(head, "head", yh, ph), frame(ts, "tail_sample", yt, pt),
                                frame(prev, "found_earlier", np.ones(len(prev), bool), pj(prev))], ignore_index=True)
            ret = judged[judged.y].copy()
            ret["reason"] = np.where(ret["where"] == "head", "judge yes", "found by random audit")
            if spec.want == "all":   # counting `prev` as found while also counting the tail as possibly-missed is conservative
                a_k = th.alpha if not streaming else th.alpha / 2 if k == 1 else th.alpha / 2 / max(n_later, 1)
                cert = A.certify(found=int(yh.sum()) + int(yt.sum()) + len(prev), n_tail=len(tail), tail_labels=list(yt), alpha=a_k)
                cert["round"] = k; cert["judged"] = len(seen)
            if not streaming or n_head >= len(order):
                break
            yield _finish(idx, spec, in_scope, n_head, ret, cert, person_mode, possible, judged, has_look)
            n_head = min(len(order), 2 * n_head)
    yield _finish(idx, spec, in_scope, n_head, ret, cert, person_mode, possible, judged, has_look)


def _finish(idx, spec, in_scope, n_head, ret, cert, person_mode, possible, judged, has_look):
    if spec.want == "best":
        # "best" is a curated subset, not everything that passed: confident yes only, top quarter (>=12) unless a number was asked
        if person_mode and has_look:   # relative to the person: top of their own photos
            ret = ret.sort_values(["rel", "p_attr"], ascending=False)
        else:
            ret = ret.sort_values(["p_attr", "fast"], ascending=False)
            ret = ret[ret.p_attr >= 0.5]
        cap = spec.max_items or max(12, int(0.25 * len(judged[judged["where"].isin(["identity_match", "head"])])))
        ret = ret.head(cap)
    report = _report(spec, idx.n_items, len(in_scope), n_head, ret, cert, person_mode, possible)
    res = AlbumResult(spec, idx.n_items, len(in_scope), ret, cert, report, judged)
    res.possible = possible
    return res


def _exclusive_by_score(rs):
    """Fallback without ranks: a photo stays only in the album whose question the judge answered most confidently."""
    scores = []
    for r in rs:
        j = getattr(r, "judged", None)
        sc = dict(zip(j.item_id, j["p_attr"].fillna(-1.0))) if j is not None and len(j) else {}
        sc.update(dict(zip(r.returned.item_id, r.returned.p_attr)))
        scores.append(sc)
    for k, r in enumerate(rs):
        keep = [all(scores[k][i] >= scores[m].get(i, -1.0) for m in range(len(rs)) if m != k) for i in r.returned.item_id]
        n0 = len(r.returned); r.returned = r.returned[keep]; moved = n0 - len(r.returned)
        if moved:
            r.report = r.report.replace(f"': {n0} items.", f"': {len(r.returned)} items.", 1) + \
                f"\n  {moved} photo(s) removed: the judge rated them higher for another album about the same person."


PAIR_MARGIN = 0.3   # "A vs B" of the same person: a photo goes to A only if it ranks >= 0.3 higher for A than for B
PAIR_SURE = 0.9     # two-group split: a photo goes to an album only if the split is >= 90% sure it belongs there
PAIR_MIN = 20       # fewer shared photos than this: too few to see two groups, use the rank margin


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _two_groups(x: np.ndarray):
    """1-D two-component Gaussian mixture by EM (numpy only: runs on a phone). Returns (P(upper group) per value,
    True if two groups describe x clearly better than one by BIC (> 10), the midpoint between the group means)."""
    x = np.asarray(x, float); n = len(x)
    mu = np.percentile(x, [25, 75]).astype(float); sd = np.full(2, max(x.std(), 1e-3)); w = np.full(2, .5)
    for _ in range(200):
        ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        m = ll.max(1, keepdims=True); r = np.exp(ll - m); r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        mu_new = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu_new) ** 2).sum(0) / nk).clip(1e-2); w = nk / n
        if np.allclose(mu_new, mu, atol=1e-6):
            mu = mu_new; break
        mu = mu_new
    ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w) - 0.5 * np.log(2 * np.pi)
    m = ll.max(1, keepdims=True); L2 = float((m.ravel() + np.log(np.exp(ll - m).sum(1))).sum())
    s1 = max(x.std(), 1e-2); L1 = float((-0.5 * ((x - x.mean()) / s1) ** 2 - np.log(s1) - 0.5 * np.log(2 * np.pi)).sum())
    bic1, bic2 = 2 * np.log(n) - 2 * L1, 5 * np.log(n) - 2 * L2
    up = int(np.argmax(mu)); post = np.exp(ll - m); post = post[:, up] / post.sum(1)
    return post, bool(bic2 < bic1 - 10), float(mu.mean())


def _split_pair(pools, event_of=None):
    """Two opposite albums of one person ("heavier" vs "fit"): per shared photo, how much more the judge said yes to A
    than to B (log-odds difference); within one event (photos <= 3 h apart) the median, since a lasting look does not
    change within hours and a suited / distant / face-only shot otherwise gets a near-random score; then the two groups
    those values form. Returns {item_id: [album index]} or None (no clear two groups -> caller uses the rank margin).
    Why (Reza's sample 10-04, every photo labeled by him): the rank-margin rule kept only 116 of his 225 heavier-era
    photos and put 8 heavier-era photos in 'fit'; this split: 225/225 in heavier, 0 heavier-era photos in fit (5 of
    his fit-era photos went to heavier: screenshots and face-only selfies, where no look-based rule can see his build).
    Public eras (Pratt / Hill / Rogen face crops): no better and no worse than the rank margin (precision within a few
    points per person, either direction)."""
    a, b = pools
    pa = dict(zip(a.item_id, a["p_attr"])); pb = dict(zip(b.item_id, b["p_attr"]))
    ids = [i for i in pa if i in pb]
    if len(ids) < PAIR_MIN:
        return None
    x = pd.Series(_logit([pa[i] for i in ids]) - _logit([pb[i] for i in ids]), index=ids)
    if event_of:
        ev = pd.Series([event_of.get(i, f"_{i}") for i in ids], index=ids)
        x = x.groupby(ev).transform("median")
    post, clear, mid = _two_groups(x.to_numpy())
    if not clear:
        return None
    out = {}
    for i, v, p in zip(ids, x.to_numpy(), post):
        # the posterior must agree with the side of the midpoint: an unequal-width mixture can otherwise hand the far
        # tail of the narrow group to the wide one
        out[i] = [0] if (p >= PAIR_SURE and v > mid) else ([1] if (p <= 1 - PAIR_SURE and v < mid) else None)
    return out


def make_exclusive(results: list, margin: float = PAIR_MARGIN, event_of: dict | None = None) -> list:
    """Opposite appearance albums of the SAME person ("heavier" vs "fit"): each of the person's photos goes to the album
    it matches clearly MORE (its within-person rank for that album beats every other album by `margin`); photos in
    between go to neither and are counted ("not clearly either"). The old rule (top half of each album, then exclusive)
    padded the smaller era's album: Reza's sample 10-04, 'fit' had 52/168 photos from his heavier year. Margin 0.3 is a
    middle setting chosen on principle (0.2 / 0.4 bracket it), not tuned to his photos.
    (Earlier: 04:2x Kevin Bacon run, a photo scored fit 0.82 but heavier 0.35 sat in 'heavier'.)"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.spec.person and r.spec.judge_question:
            groups[r.spec.person.lower()].append(r)
    for rs in groups.values():
        if len(rs) < 2:
            continue
        rels, pools = [], []
        for r in rs:
            j = getattr(r, "judged", None)
            pool = None
            if j is not None and len(j) and "rel" in j:
                pool = j[j["where"] == "identity_match"] if "where" in j else j
            if pool is None or not pool["rel"].notna().any():
                rels = None; break
            pools.append(pool); rels.append(dict(zip(pool.item_id, pool["rel"].fillna(-1.0))))
        if rels is None:                 # no within-person ranks (e.g. not person albums): compare raw judge scores
            _exclusive_by_score(rs)
            continue
        ids = set().union(*[set(x) for x in rels])
        best = {}
        split = _split_pair(pools, event_of) if len(rels) == 2 else None
        own_cut = Thresholds().rel_cut
        for i in ids:
            have = [m for m, x in enumerate(rels) if i in x]
            if len(have) < len(rels):
                # outside another album's scope (e.g. "fit, 2010-2015" vs "heavier" any year): nothing to compare
                # against, so the photo follows its own album's usual rule (top half) -- Kevin Bacon regress 10-04
                best[i] = [m for m in have if rels[m][i] > own_cut]
                continue
            if split is not None:
                best[i] = split.get(i)
                continue
            sc = [x[i] for x in rels]
            k = int(np.argmax(sc)); others = [v for m, v in enumerate(sc) if m != k]
            best[i] = [k] if sc[k] - max(others) >= margin else None
        between = sum(v is None for v in best.values())
        for k, r in enumerate(rs):
            n0 = len(r.returned)
            orig = set(r.returned.item_id)
            keep_fn = (lambda i: bool(best.get(i)) and k in best[i] and (i in orig if r.spec.want == "best" else True))
            new = pools[k][pools[k].item_id.map(keep_fn)].copy()   # "best" albums keep their short list
            new["y"] = True
            if "reason" in r.returned:
                new["reason"] = "face match + judge: matches this album more than the other"
            r.returned = new.sort_values("rel", ascending=False)
            r.report = r.report.replace(f"': {n0} items.", f"': {len(r.returned)} items.", 1) + (
                f"\n  Paired with another album about the same person: each photo goes to the album it matches clearly "
                f"more. {between} photo(s) are not clearly either and are in neither album.")
    return results


def _report(spec, n_all, n_scope, n_head, ret, cert, person_mode=False, possible=None):
    lines = [f"Album '{spec.name}': {len(ret)} items.",
             f"  Library: {n_all:,} items; in scope after date/media filters: {n_scope:,}. Every in-scope item was scored by the fast models."]
    if getattr(spec, "time_of_day", None):
        lines.append(f"  Time of day {spec.time_of_day} (local clock where it was taken). Items whose local time is unknown "
                     f"(often videos and copied files) cannot pass this filter and are not in scope.")
    if person_mode:
        lines.append(f"  Identity from face matching only. {0 if possible is None else len(possible)} more 'possible' items "
                     f"(weaker face match) are listed for you to confirm; they are NOT in the album.")
        if n_head:
            lines.append(f"  The AI judge rated the appearance condition on all {n_head:,} face-matched items (no sampling); "
                         f"the album keeps the photos that rank highest among THIS person's own photos (the judge's absolute "
                         f"scale differs from person to person). Check the ranked list on the review page.")
        lines.append("  Completeness for a person cannot be certified by the AI judge (it is not a face recognizer). On a public "
                     "test library, face matching found 88-98% of each person's untagged photos. Photos where the face is hidden, "
                     "tiny or in profile are the usual misses; check the random sample on the review page.")
    elif cert:
        lines.append("  " + A.describe(cert, n_scope, n_head))
    else:
        lines.append(f"  'Best-of' request: judge looked at the top {n_head:,}; completeness is not estimated.")
    return "\n".join(lines)
