"""Load a built index (all shards) into memory. 150k items x ~1k dims float16 is ~300 MB: fits in RAM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Index:
    root: Path
    items: pd.DataFrame        # one row per item (photo/video)
    units: pd.DataFrame        # one row per analyzed image unit; column `item_row` -> items row
    clip: np.ndarray           # [n_units, D] float16
    faces: pd.DataFrame        # one row per face; column `unit_row` -> units row
    face_emb: np.ndarray       # [n_faces, 512] float16
    errors: pd.DataFrame

    @property
    def n_items(self):
        return len(self.items)


def load(root: str | Path, clip_model: str | None = None) -> Index:
    """clip_model: which image-encoder's vectors to load (must match the text encoder used at query time).
    Defaults to models.DEFAULT_CLIP if that model was indexed, else the first one."""
    import json as _json
    root = Path(root)
    done = sorted((root / "shards").glob("*/DONE"))
    names = []
    if done:
        st = _json.loads((done[0].parent / "stats.json").read_text())
        names = st.get("clip_models") or [st.get("clip_model")]
    if clip_model is None:
        from .models import DEFAULT_CLIP
        clip_model = DEFAULT_CLIP if DEFAULT_CLIP in names else (names[0] if names else None)
    if names and clip_model not in names:
        raise ValueError(f"{clip_model} not in index (has {names})")
    k = names.index(clip_model) if names else 0
    clip_file = "clip.npy" if k == 0 else f"clip_{k}.npy"
    items = pd.read_parquet(root / "items.parquet").reset_index(drop=True)
    row_of = {iid: i for i, iid in enumerate(items["item_id"])}
    U, C, F, FE, E = [], [], [], [], []
    off_u = 0
    for sd in sorted((root / "shards").glob("*/DONE")):
        sd = sd.parent
        u = pd.read_parquet(sd / "units.parquet")
        c = np.load(sd / clip_file)
        f = pd.read_parquet(sd / "faces.parquet")
        fe = np.load(sd / "face_emb.npy")
        if len(f):
            f["unit_row"] = f["unit"] + off_u
        U.append(u); C.append(c); F.append(f); FE.append(fe)
        if (sd / "errors.parquet").exists():
            E.append(pd.read_parquet(sd / "errors.parquet"))
        off_u += len(u)
    units = pd.concat(U, ignore_index=True) if U else pd.DataFrame(columns=["item_id", "frame_t"])
    units["item_row"] = units["item_id"].map(row_of)
    faces = pd.concat([f for f in F if len(f)], ignore_index=True) if any(len(f) for f in F) else pd.DataFrame(
        columns=["unit_row", "item_id", "x1", "y1", "x2", "y2", "det_score", "face_px"])
    if len(faces):
        faces["item_row"] = faces["item_id"].map(row_of)
    idx = Index(root, items, units, np.concatenate(C) if C else np.zeros((0, 1), np.float16), faces,
                 np.concatenate([x for x in FE if len(x)]) if any(len(x) for x in FE) else np.zeros((0, 512), np.float16),
                 pd.concat(E, ignore_index=True) if E else pd.DataFrame())
    idx.clip_model = clip_model
    return idx


def per_item_max(unit_scores: np.ndarray, unit_item_row: np.ndarray, n_items: int) -> np.ndarray:
    """Item score = best score over its units (a video matches if any sampled frame matches)."""
    out = np.full(n_items, -np.inf, dtype=np.float32)
    np.maximum.at(out, unit_item_row, unit_scores.astype(np.float32))
    return out


def subset(idx: Index, item_rows) -> Index:
    """A view of the index restricted to some items (one user's library, or the current album for refinement edits).
    Units and faces are filtered and their row pointers remapped; vectors are sliced (copies)."""
    item_rows = np.asarray(sorted(set(int(r) for r in item_rows)), dtype=np.int64)
    new_of_old = -np.ones(idx.n_items, np.int64); new_of_old[item_rows] = np.arange(len(item_rows))
    items = idx.items.iloc[item_rows].reset_index(drop=True)
    um = new_of_old[idx.units["item_row"].to_numpy()] >= 0
    units = idx.units[um].copy(); old_unit_rows = np.where(um)[0]
    units["item_row"] = new_of_old[units["item_row"].to_numpy()]; units = units.reset_index(drop=True)
    new_unit = -np.ones(len(idx.units), np.int64); new_unit[old_unit_rows] = np.arange(len(old_unit_rows))
    if len(idx.faces):
        fm = new_of_old[idx.faces["item_row"].to_numpy()] >= 0
        faces = idx.faces[fm].copy(); fe = idx.face_emb[np.where(fm)[0]]
        faces["item_row"] = new_of_old[faces["item_row"].to_numpy()]
        if "unit_row" in faces:
            faces["unit_row"] = new_unit[faces["unit_row"].to_numpy()]
        faces = faces.reset_index(drop=True)
    else:
        faces, fe = idx.faces, idx.face_emb
    out = Index(idx.root, items, units, idx.clip[old_unit_rows], faces, fe, idx.errors)
    out.clip_model = getattr(idx, "clip_model", None)
    return out
