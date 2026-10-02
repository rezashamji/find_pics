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


def load(root: str | Path) -> Index:
    root = Path(root)
    items = pd.read_parquet(root / "items.parquet").reset_index(drop=True)
    row_of = {iid: i for i, iid in enumerate(items["item_id"])}
    U, C, F, FE, E = [], [], [], [], []
    off_u = 0
    for sd in sorted((root / "shards").glob("*/DONE")):
        sd = sd.parent
        u = pd.read_parquet(sd / "units.parquet")
        c = np.load(sd / "clip.npy")
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
    return Index(root, items, units, np.concatenate(C) if C else np.zeros((0, 1), np.float16), faces,
                 np.concatenate([x for x in FE if len(x)]) if any(len(x) for x in FE) else np.zeros((0, 512), np.float16),
                 pd.concat(E, ignore_index=True) if E else pd.DataFrame())


def per_item_max(unit_scores: np.ndarray, unit_item_row: np.ndarray, n_items: int) -> np.ndarray:
    """Item score = best score over its units (a video matches if any sampled frame matches)."""
    out = np.full(n_items, -np.inf, dtype=np.float32)
    np.maximum.at(out, unit_item_row, unit_scores.astype(np.float32))
    return out
