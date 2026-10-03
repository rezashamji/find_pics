"""People: a person is a set of reference identity vectors. Matching = nearest reference, not a single centroid.

Why a set and not an average: if someone changed a lot (10 years, large weight change), the average of
"heavy face" and "lean face" vectors can sit between both and match neither well. Keeping every reference and
taking the max similarity lets each era match its own examples.

Expansion: after the first pass, confidently matched faces (high similarity) are added as new references, which
pulls in photos that only resemble the person's other-era photos (same idea as clustering, done query-time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store import Index


def face_sims(idx: Index, refs: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Max cosine similarity of every library face to any reference vector. -> [n_faces]."""
    if len(idx.face_emb) == 0 or len(refs) == 0:
        return np.zeros(len(idx.face_emb), np.float32)
    R = refs.astype(np.float32).T
    out = np.empty(len(idx.face_emb), np.float32)
    for s in range(0, len(idx.face_emb), chunk):
        S = idx.face_emb[s:s + chunk].astype(np.float32) @ R
        # a face that became a reference (query-time expansion) would otherwise match itself at 1.00, making the
        # reported similarity meaningless; ignore exact self-matches so the score reflects OTHER reference faces
        S[S > 0.999] = -1.0
        out[s:s + chunk] = S.max(1)
    return out


def item_person_scores(idx: Index, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (item_score [n_items], best_face_row_per_item [n_items], -1 if no face)."""
    s = face_sims(idx, refs)
    score = np.full(idx.n_items, -1.0, np.float32)
    best = np.full(idx.n_items, -1, np.int64)
    if len(s):
        df = pd.DataFrame({"item_row": idx.faces["item_row"].to_numpy(), "s": s, "f": np.arange(len(s))})
        g = df.loc[df.groupby("item_row")["s"].idxmax()]
        score[g["item_row"].to_numpy()] = g["s"].to_numpy()
        best[g["item_row"].to_numpy()] = g["f"].to_numpy()
    return score, best


def refs_from_items(idx: Index, item_rows, min_face_px: float = 40.0, sim_floor: float | None = 0.2,
                    return_rows: bool = False):
    """Reference vectors from items known to contain the person (e.g. Apple's People tags, or user picks).

    An item may contain several faces; we keep the face that is most self-consistent with the other references
    (the person appears in all these photos; strangers do not), which strips bystanders automatically.
    """
    item_rows = set(int(r) for r in item_rows)
    f = idx.faces[idx.faces["item_row"].isin(item_rows) & (idx.faces["face_px"] >= min_face_px)]
    if len(f) == 0:
        z = np.zeros((0, idx.face_emb.shape[1]), np.float16)
        return (z, np.zeros(0, np.int64)) if return_rows else z
    f = f.sort_values("item_row")
    E = idx.face_emb[f.index.to_numpy()].astype(np.float32)
    rows = f["item_row"].to_numpy()
    # consensus: does this face appear in the OTHER tagged photos? For face i and each other photo j take the best
    # matching face in j (max), then the median over photos. Bystanders score low (they appear in few tagged photos);
    # the real person scores high. Max-per-photo matters: a plain median over all faces is swamped by bystanders.
    S = E @ E.T
    starts = np.r_[0, np.where(np.diff(rows) != 0)[0] + 1]
    per_item = np.maximum.reduceat(S, starts, axis=1)          # [n_faces, n_items]
    own = np.searchsorted(rows[starts], rows)                   # column index of each face's own photo
    per_item[np.arange(len(rows)), own] = np.nan
    cons = np.nan_to_num(np.nanmedian(per_item, axis=1), nan=-1.0) if per_item.shape[1] > 1 else np.zeros(len(rows))
    keep = pd.DataFrame({"r": rows, "c": cons, "i": np.arange(len(rows))}).sort_values("c", ascending=False)
    keep = keep.drop_duplicates("r")
    if sim_floor is not None:
        keep = keep[keep["c"] >= sim_floor]
    refs = E[keep["i"].to_numpy()].astype(np.float16)
    if return_rows:  # face rows ordered by consensus: row 0 is the most typical face of this person
        return refs, f.index.to_numpy()[keep["i"].to_numpy()]
    return refs


def expand_refs(idx: Index, refs: np.ndarray, accept: float, rounds: int = 2, max_new: int = 2000) -> np.ndarray:
    """Add faces with similarity >= accept as new references (query-time clustering)."""
    for _ in range(rounds):
        s = face_sims(idx, refs)
        new = np.where(s >= accept)[0]
        if len(new) <= len(refs):
            break
        new = new[np.argsort(-s[new])][:max_new]
        refs = idx.face_emb[new]
    return refs


def face_groups(idx: Index, top: int = 12, sample: int = 20_000, accept: float = 0.55, min_px: float = 40.0,
                min_det: float = 0.7, seed: int = 0) -> list[dict]:
    """The most frequent people in a library WITHOUT names (Apple's data copy has no People tags): greedy grouping of a
    face sample. Repeatedly take the face with the most neighbours (cosine >= accept, the same cut expand_refs uses),
    make it and its neighbours a group, remove them. Returns the `top` largest groups as
    {"faces": face rows (sample), "items": distinct item rows, "rep": the face row nearest the group mean}.
    Faces smaller than min_px are skipped (blurry crowd faces chain different people together)."""
    f = idx.faces
    ok = np.ones(len(f), bool)
    if "face_px" in f:
        ok &= f["face_px"].to_numpy() >= min_px
    if "det_score" in f:   # detector hits on dogs, flowers, backs of heads (det ~0.55) formed a junk "person" group
        ok &= f["det_score"].to_numpy() >= min_det
    rows = np.where(ok)[0]
    rng = np.random.default_rng(seed)
    if len(rows) > sample:
        rows = np.sort(rng.choice(rows, sample, replace=False))
    E = idx.face_emb[rows].astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    nb = [None] * len(rows); deg = np.zeros(len(rows), int)
    for s in range(0, len(rows), 2048):          # neighbour lists, chunked (20k x 20k floats would be 1.6 GB)
        S = E[s:s + 2048] @ E.T
        for i, r in enumerate(S):
            nb[s + i] = np.where(r >= accept)[0]; deg[s + i] = len(nb[s + i])
    alive = np.ones(len(rows), bool); groups = []
    item_row = f["item_row"].to_numpy()
    while len(groups) < top and alive.any():
        d = np.where(alive, [np.count_nonzero(alive[n]) for n in nb], -1)
        c = int(np.argmax(d))
        if d[c] < 2:
            break
        mem = nb[c][alive[nb[c]]]
        alive[mem] = False; alive[c] = False
        fr = rows[mem]
        mean = E[mem].mean(0); rep = int(fr[np.argmax(E[mem] @ mean)])
        groups.append(dict(faces=fr.tolist(), items=sorted(set(item_row[fr].tolist())), rep=rep))
    groups.sort(key=lambda g: -len(g["items"]))
    return groups


def group_sheet(idx: Index, groups: list[dict], per: int = 8, tile: int = 220, seed: int = 0):
    """One row per group: number + `per` face crops (the group's representative first, then a random sample), so a
    person can say "I am group 3". Returns a PIL image."""
    from PIL import Image, ImageDraw
    from .engine import reference_crop
    rng = np.random.default_rng(seed)
    W = 60 + per * (tile + 6); H = len(groups) * (tile + 10)
    sheet = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for gi, g in enumerate(groups):
        y = gi * (tile + 10)
        d.text((10, y + tile // 2), str(gi + 1), fill=(200, 0, 0))
        rest = [f for f in g["faces"] if f != g["rep"]]
        pick = [g["rep"]] + list(rng.choice(rest, min(per - 1, len(rest)), replace=False))
        for j, fr in enumerate(pick):
            try:
                c = reference_crop(idx, int(fr)).convert("RGB")
            except Exception:
                continue
            c.thumbnail((tile, tile)); sheet.paste(c, (60 + j * (tile + 6), y))
    return sheet
