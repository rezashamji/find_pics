"""Execute one album spec against an index: filter -> cheap scoring of ALL items -> judge candidates -> audit misses.

Scores:
  person_score[i]  = max cosine between any face in item i and the person's reference faces (-1 if no face)
  look_score[i]    = max over the item's units of (mean cos to `looks` texts - mean cos to `avoid` texts)
Judge: VLM P(yes) on the album's judge_question (and, for person albums, an identity check against a reference crop
when the face score is in the uncertain band).
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


@dataclass
class Thresholds:
    person_sure: float = 0.45      # ArcFace cosine above which identity is accepted without the judge (set from eval)
    person_maybe: float = 0.20     # below this a face is treated as not the person
    judge_accept: float = 0.5      # P(yes) cut
    max_candidates: int = 3000     # judge budget for candidates
    audit_budget: int = 400        # judge budget for the recall audit


@dataclass
class AlbumResult:
    spec: AlbumSpec
    n_scanned: int
    n_in_scope: int
    returned: pd.DataFrame            # item rows with scores, judge p, reason
    audit: dict | None
    report: str
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def judge_image(idx: Index, item_row: int, face_row: int | None):
    """Image shown to the judge: the photo (or the video frame where the person's best face is), red box on the face."""
    r = idx.items.iloc[item_row]
    t = None
    if face_row is not None and face_row >= 0:
        t = float(idx.faces.iloc[face_row]["frame_t"]) if "frame_t" in idx.faces else None
    if r["media"] == "video":
        frames = sample_video_frames(r["path"])
        if not frames:
            return None
        if t is not None and t >= 0:
            im = min(frames, key=lambda x: abs(x[0] - t))[1]
        else:
            im = frames[len(frames) // 2][1]
    else:
        im = load_image(r["path"])
    if face_row is not None and face_row >= 0:
        f = idx.faces.iloc[face_row]
        sx = im.width / float(f["img_w"]); sy = im.height / float(f["img_h"])
        im = draw_box(im, (f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy))
    return im


def _judge(idx, judge, rows, face_rows, question, batch=64):
    ps = []
    for s in range(0, len(rows), batch):
        ims, keep = [], []
        for r, fr in zip(rows[s:s + batch], face_rows[s:s + batch]):
            try:
                im = judge_image(idx, int(r), int(fr) if fr is not None else None)
            except Exception:
                im = None
            keep.append(im is not None)
            if im is not None:
                ims.append(im)
        out = iter(judge.p_yes(ims, question) if ims else [])
        ps.extend(next(out) if k else 0.0 for k in keep)
    return np.array(ps, np.float32)


def run_album(idx: Index, spec: AlbumSpec, enc, judge, refs: np.ndarray | None, th: Thresholds = Thresholds(),
              seed: int = 0) -> AlbumResult:
    rng = np.random.default_rng(seed)
    scope = scope_mask(idx, spec)
    n_scope = int(scope.sum())
    look = look_scores(idx, enc, spec.looks, spec.avoid)
    if spec.person and refs is not None and len(refs):
        pscore, best_face = item_person_scores(idx, refs)
    else:
        pscore, best_face = np.full(idx.n_items, np.nan, np.float32), np.full(idx.n_items, -1)

    person_mode = spec.person is not None and refs is not None and len(refs) > 0
    # ---- cheap score over everything in scope; candidates = top of it
    if person_mode:
        cand = scope & (pscore >= th.person_maybe)
        # rank candidates by attribute look score (identity is gated, appearance is ranked)
        cheap = np.where(cand, look, -np.inf) if (spec.looks or spec.avoid) else np.where(cand, pscore, -np.inf)
        audit_score = np.where(scope, pscore, -np.inf)  # misses hide in the near-threshold identity band
    else:
        cheap = np.where(scope, look, -np.inf)
        cand = scope & np.isfinite(cheap)
        audit_score = cheap
    order = np.argsort(-cheap)
    order = order[np.isfinite(cheap[order])][: th.max_candidates]

    # ---- judge candidates
    q = spec.judge_question
    if person_mode:
        p = _judge(idx, judge, order, best_face[order], q) if (spec.looks or spec.avoid) else np.ones(len(order), np.float32)
        ident_ok = pscore[order] >= th.person_sure
        # uncertain identity band: accept only if the judge also says it's a match for the attribute; flag it
        accept = (p >= th.judge_accept) & (pscore[order] >= th.person_maybe)
        reason = np.where(ident_ok, "face match", "weak face match")
    else:
        p = _judge(idx, judge, order, [None] * len(order), q)
        accept = p >= th.judge_accept
        reason = np.array(["judge yes"] * len(order))
    cands = pd.DataFrame(dict(item_row=order, item_id=idx.items["item_id"].to_numpy()[order],
                              path=idx.items["path"].to_numpy()[order], cheap=cheap[order],
                              person=pscore[order], p_yes=p, accepted=accept, reason=reason,
                              face_row=best_face[order]))
    ret = cands[cands.accepted].copy()
    if spec.want == "best":
        ret = ret.sort_values(["p_yes", "cheap"], ascending=False)
    if spec.max_items:
        ret = ret.head(spec.max_items)

    # ---- audit: what did we miss? stratified sample of the in-scope items NOT returned
    est = None
    if spec.want == "all":
        returned_rows = set(ret.item_row)
        judged_rows = set(cands.item_row)
        pool = np.array([i for i in np.where(scope)[0] if i not in returned_rows])
        if len(pool):
            ps = audit_score[pool]
            ps = np.where(np.isfinite(ps), ps, -9.0)
            qs = np.quantile(ps, [0.99, 0.95, 0.8, 0.5]) if len(pool) > 50 else [np.median(ps)]
            edges = sorted(set(float(x) for x in qs), reverse=True) + [-np.inf]
            strata = A.make_strata(ps, pool, edges)
            alloc = A.allocate([len(ids) for _, ids in strata], th.audit_budget)
            S = A.sample_strata(strata, alloc, rng)
            already = dict(zip(cands.item_row, cands.p_yes))
            for s in S:
                rows = [int(r) for r in s.sampled_ids]
                need = [r for r in rows if r not in already]
                if need:
                    pq = _judge(idx, judge, need, best_face[need] if person_mode else [None] * len(need),
                                q if not person_mode or (spec.looks or spec.avoid) else _identity_q())
                    already.update(zip(need, pq))
                labs = []
                for r in rows:
                    y = already[r] >= th.judge_accept
                    if person_mode:
                        y = y and pscore[r] >= th.person_maybe  # judge cannot know identity without a face match
                    labs.append(bool(y))
                s.labels = labs
            n_ret = len(ret)
            est = A.estimate_recall(n_ret, int((ret.p_yes >= th.judge_accept).sum()), n_ret, S)
    report = _report(spec, idx.n_items, n_scope, len(cands), ret, est)
    return AlbumResult(spec, idx.n_items, n_scope, ret, est, report, cands)


def _identity_q():
    return "Is there a clearly visible person in the red box?"


def _report(spec, n_all, n_scope, n_judged, ret, est):
    lines = [f"Album '{spec.name}': {len(ret)} items.",
             f"  Library: {n_all:,} items. In scope (date/media filters): {n_scope:,}. All in-scope items scored by the fast models.",
             f"  Judge (VLM) looked at {n_judged:,} candidates."]
    if est:
        lines.append("  " + A.describe(est, n_scope))
        lines.append("  Completeness here is measured against the VLM judge's labels on a random audit sample; "
                     "if the judge is wrong, this number is wrong in the same direction.")
    elif spec.want == "best":
        lines.append("  'Best' album: ranked top items; completeness is not estimated for best-of requests.")
    return "\n".join(lines)
