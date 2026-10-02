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
    person_accept: float = 0.40  # face cosine to the person's references: identity accepted (testlib: 324/325 precise at 0.45)
    person_maybe: float = 0.30   # [maybe, accept) -> "possible" list for the user to confirm; never auto-accepted
    head_chunk: int = 200       # concept queries: judge the head in chunks...
    head_max: int = 6000        # ...and keep extending while the last chunk's yes-rate >= head_stop_rate
    head_stop_rate: float = 0.03
    judge_accept: float = 0.5   # identity and object questions
    attr_accept: float = 0.3    # appearance judgments about a person (CelebA: judge is conservative; 0.3 -> 339/600 hits, 63/1400 FA)
    head_size: int = 600        # minimum head (concept queries)
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
    if getattr(idx, "clip_model", None) and idx.clip_model != enc.name:
        raise ValueError(f"text encoder {enc.name} != index vectors {idx.clip_model}")
    C = idx.clip
    s = np.zeros(len(C), np.float32)
    if looks:
        s += (C @ enc.texts(looks).T.astype(np.float16)).astype(np.float32).mean(1)
    if avoid:
        s -= (C @ enc.texts(avoid).T.astype(np.float16)).astype(np.float32).mean(1)
    # remember which sampled frame of each video matched best, so the judge sees THAT frame, not an arbitrary one
    u = pd.DataFrame({"item_row": idx.units["item_row"].to_numpy(), "s": s, "t": idx.units["frame_t"].to_numpy()})
    best = u.loc[u.groupby("item_row")["s"].idxmax()]
    idx.best_frame_t = dict(zip(best.item_row.astype(int), best.t.astype(float)))
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
    t = getattr(idx, "best_frame_t", {}).get(int(item_row))
    if t is not None and t >= 0:
        return min(frames, key=lambda x: abs(x[0] - t))[1]
    return frames[len(frames) // 2][1]


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


def _judge_rows(idx, judge, rows, face_rows, question, ref_img=None, batch=48, crop_person=False):
    out = []
    for s in range(0, len(rows), batch):
        ims, ok = [], []
        for r, fr in zip(rows[s:s + batch], face_rows[s:s + batch]):
            try:
                im = _frame_for(idx, int(r), int(fr))
                if im is not None:
                    if ref_img is not None:
                        im = side_by_side(ref_img, im)
                    elif crop_person and int(fr) >= 0:
                        im = person_crop(idx, im, int(fr))
                    else:
                        im = _boxed(idx, im, int(fr))
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
    """Two regimes.

    Person albums: identity comes ONLY from face vectors (the VLM is not a face recognizer: on the test library its
    side-by-side identity answers were right 4 times out of 191 yeses). Items with face similarity >= person_accept are
    the person; [person_maybe, person_accept) is a counted "possible" list for the user. If the album also has an
    appearance condition, the judge rates EVERY identity-accepted item (no sampling needed: the set is small).
    Identity completeness cannot be certified by a machine judge; the report says so and the review page shows a
    random sample of the rest for a human.

    Object/scene albums: rank everything by the fast score, judge the head in chunks while it keeps finding matches,
    then a uniform random tail sample -> Clopper-Pearson certificate (relative to the judge).
    """
    rng = np.random.default_rng(seed)
    scope = scope_mask(idx, spec)
    in_scope = np.where(scope)[0]
    has_look = bool((spec.looks or spec.avoid) and spec.judge_question)
    look = look_scores(idx, enc, spec.looks, spec.avoid) if has_look or not spec.person else np.zeros(idx.n_items, np.float32)
    person_mode = bool(spec.person) and refs is not None and len(refs) > 0
    if person_mode:
        pscore, best_face = item_person_scores(idx, refs)
        fast = pscore + (0.25 * look if has_look else 0.0)
    else:
        pscore = np.full(idx.n_items, np.nan, np.float32); best_face = np.full(idx.n_items, -1)
        fast = look

    def frame(rows, where, y, p_attr):
        d = pd.DataFrame(dict(item_row=np.asarray(rows, int), where=where, y=y, p_attr=p_attr))
        d["item_id"] = idx.items["item_id"].to_numpy()[d.item_row]; d["path"] = idx.items["path"].to_numpy()[d.item_row]
        d["person"] = pscore[d.item_row]; d["look"] = look[d.item_row]; d["fast"] = fast[d.item_row]
        d["face_row"] = best_face[d.item_row]
        return d

    possible = pd.DataFrame()
    cert = None
    if person_mode:
        ident = in_scope[pscore[in_scope] >= th.person_accept]
        maybe = in_scope[(pscore[in_scope] >= th.person_maybe) & (pscore[in_scope] < th.person_accept)]
        if has_look:
            q = spec.judge_question.replace("the person in the red box", "the person in this photo")
            p_id = _judge_rows(idx, judge, ident, best_face[ident], q, crop_person=True)
            p_mb = _judge_rows(idx, judge, maybe, best_face[maybe], q, crop_person=True) if len(maybe) else np.zeros(0)
            y_id = p_id >= th.attr_accept; y_mb = p_mb >= th.attr_accept
        else:
            p_id = np.ones(len(ident), np.float32); p_mb = np.ones(len(maybe), np.float32)
            y_id = np.ones(len(ident), bool); y_mb = np.ones(len(maybe), bool)
        judged = frame(ident, "identity_match", y_id, p_id)
        possible = frame(maybe, "possible_identity", y_mb, p_mb)
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
        n_head = min(th.head_size, len(order))
        ph = list(_judge_rows(idx, judge, order[:n_head], best_face[order[:n_head]], spec.judge_question))
        while n_head < min(th.head_max, len(order)):
            last = np.asarray(ph[-th.head_chunk:]) >= th.judge_accept
            if last.mean() < th.head_stop_rate:
                break
            nxt = order[n_head:n_head + th.head_chunk]
            ph += list(_judge_rows(idx, judge, nxt, best_face[nxt], spec.judge_question))
            n_head += len(nxt)
        head, tail = order[:n_head], order[n_head:]
        ph = np.asarray(ph, np.float32); yh = ph >= th.judge_accept
        n_t = min(len(tail), th.tail_budget)
        ts = rng.choice(tail, size=n_t, replace=False) if n_t else np.zeros(0, int)   # fixed before any tail label
        pt = _judge_rows(idx, judge, ts, best_face[ts], spec.judge_question) if n_t else np.zeros(0, np.float32)
        yt = pt >= th.judge_accept
        judged = pd.concat([frame(head, "head", yh, ph), frame(ts, "tail_sample", yt, pt)], ignore_index=True)
        ret = judged[judged.y].copy()
        ret["reason"] = np.where(ret["where"] == "head", "judge yes", "found by random audit")
        if spec.want == "all":
            cert = A.certify(found=int(yh.sum()) + int(yt.sum()), n_tail=len(tail), tail_labels=list(yt), alpha=th.alpha)
    if spec.want == "best":
        # "best" is a curated subset, not everything that passed: confident yes only, top quarter (>=12) unless a number was asked
        ret = ret.sort_values(["p_attr", "fast"], ascending=False)
        ret = ret[ret.p_attr >= 0.5]
        cap = spec.max_items or max(12, int(0.25 * len(judged[judged["where"].isin(["identity_match", "head"])])))
        ret = ret.head(cap)
    report = _report(spec, idx.n_items, len(in_scope), n_head, ret, cert, person_mode, possible)
    res = AlbumResult(spec, idx.n_items, len(in_scope), ret, cert, report, judged)
    res.possible = possible
    return res


def make_exclusive(results: list) -> list:
    """Opposite appearance albums of the SAME person (e.g. 'heavier' vs 'fit'): a photo may only stay in the album whose
    question the judge answered most confidently, comparing the judge's scores on ALL the questions, not just album
    membership. (04:2x Kevin Bacon run: a photo scored fit 0.82 but heavier 0.35 sat in 'heavier' because 'fit' was capped.)"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.spec.person and r.spec.judge_question:
            groups[r.spec.person.lower()].append(r)
    for rs in groups.values():
        if len(rs) < 2:
            continue
        scores = []
        for r in rs:
            j = getattr(r, "judged", None)
            sc = dict(zip(j.item_id, j.p_attr)) if j is not None and len(j) else {}
            sc.update(dict(zip(r.returned.item_id, r.returned.p_attr)))
            scores.append(sc)
        for k, r in enumerate(rs):
            keep = [all(scores[k][i] >= scores[m].get(i, -1.0) for m in range(len(rs)) if m != k) for i in r.returned.item_id]
            n0 = len(r.returned); r.returned = r.returned[keep]; moved = n0 - len(r.returned)
            if moved:
                r.report = r.report.replace(f"': {n0} items.", f"': {len(r.returned)} items.", 1) + \
                    f"\n  {moved} photo(s) removed: the judge rated them higher for another album about the same person."
    return results


def _report(spec, n_all, n_scope, n_head, ret, cert, person_mode=False, possible=None):
    lines = [f"Album '{spec.name}': {len(ret)} items.",
             f"  Library: {n_all:,} items; in scope after date/media filters: {n_scope:,}. Every in-scope item was scored by the fast models."]
    if person_mode:
        lines.append(f"  Identity from face matching only. {0 if possible is None else len(possible)} more 'possible' items "
                     f"(weaker face match) are listed for you to confirm; they are NOT in the album.")
        if n_head:
            lines.append(f"  The AI judge rated the appearance condition on all {n_head:,} face-matched items (no sampling).")
        lines.append("  Completeness for a person cannot be certified by the AI judge (it is not a face recognizer). On a public "
                     "test library, face matching found 88-98% of each person's untagged photos. Photos where the face is hidden, "
                     "tiny or in profile are the usual misses; check the random sample on the review page.")
    elif cert:
        lines.append("  " + A.describe(cert, n_scope, n_head))
    else:
        lines.append(f"  'Best-of' request: judge looked at the top {n_head:,}; completeness is not estimated.")
    return "\n".join(lines)
