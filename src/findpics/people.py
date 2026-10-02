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
