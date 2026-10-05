"""Bursts: near-identical shots of one moment, shown as one stack ("+N similar") instead of N tiles. Display only:
every photo stays in the album. Measured need (Reza's sample 10-04): 'heavier' 139 items were 86 moments, 'fit' 127 were
59; the 12 largest groups at these settings were all the same moment by eye (pose variants of one burst)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BURST_SIM = 0.90      # image-vector cosine
BURST_MINUTES = 10    # and taken at most this far apart


def _item_vectors(idx) -> np.ndarray:
    v = getattr(idx, "_burst_vec", None)
    if v is None:
        C = idx.clip.astype(np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        first = pd.Series(np.arange(len(idx.units)), index=idx.units["item_row"].to_numpy()).groupby(level=0).first()
        v = np.zeros((idx.n_items, C.shape[1]), np.float32)
        v[first.index.to_numpy()] = C[first.to_numpy()]
        idx._burst_vec = v
    return v


def burst_ids(idx, item_rows, sim: float = BURST_SIM, minutes: float = BURST_MINUTES) -> list[int]:
    """Group id per item (same order as item_rows); group ids are numbered in order of first appearance, so the first
    item of each group (in the album's own order) is its cover. Items without a date are never stacked."""
    rows = np.asarray(item_rows, int)
    if len(rows) == 0:
        return []
    V = _item_vectors(idx)[rows]
    t = pd.to_datetime(idx.items["taken"].iloc[rows].reset_index(drop=True), utc=True, errors="coerce", format="ISO8601")
    dated = t.notna().to_numpy()
    tm = ((t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().fillna(0.0) / 60.0).to_numpy()
    A = (V @ V.T >= sim) & (np.abs(tm[:, None] - tm[None, :]) <= minutes) & dated[:, None] & dated[None, :]
    gid = -np.ones(len(rows), int); g = 0
    for i in range(len(rows)):
        if gid[i] >= 0:
            continue
        gid[i] = g; stack = [i]
        while stack:
            j = stack.pop()
            for k in np.where(A[j] & (gid < 0))[0]:
                gid[k] = g; stack.append(k)
        g += 1
    return gid.tolist()
