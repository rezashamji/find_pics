import numpy as np
import pandas as pd

from findpics.people import face_groups
from findpics.store import Index


def _idx(n_a=30, n_b=12, n_junk=10, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=512), rng.normal(size=512)
    E = np.vstack([a + 0.3 * rng.normal(size=(n_a, 512)), b + 0.3 * rng.normal(size=(n_b, 512)),
                   a + 0.3 * rng.normal(size=(n_junk, 512))])     # junk looks like person a but is a low-confidence hit
    n = len(E)
    faces = pd.DataFrame(dict(item_row=np.arange(n), face_px=np.full(n, 80.0),
                              det_score=np.r_[np.full(n_a + n_b, 0.85), np.full(n_junk, 0.5)]))
    items = pd.DataFrame(dict(item_id=[f"i{k}" for k in range(n)]))
    return Index("x", items, pd.DataFrame(), np.zeros((0, 1), np.float16), faces, E.astype(np.float16), [])


def test_face_groups_largest_first_and_low_confidence_ignored():
    G = face_groups(_idx(), top=5)
    assert len(G[0]["items"]) == 30 and set(G[0]["items"]) == set(range(30))     # junk rows 42..51 not included
    assert set(G[1]["items"]) == set(range(30, 42))
