import numpy as np
import pandas as pd
from types import SimpleNamespace as NS

from findpics.bursts import burst_ids


def test_bursts_group_same_moment_only():
    e = np.eye(3, dtype=np.float32)
    clip = np.stack([e[0], e[0] * .99 + e[1] * .1, e[0], e[2], e[0]]).astype(np.float16)
    taken = ["2023-05-01T10:00:00Z", "2023-05-01T10:02:00Z", "2023-05-01T13:00:00Z", "2023-05-01T10:01:00Z", None]
    idx = NS(clip=clip, units=pd.DataFrame(dict(item_row=range(5))), n_items=5, items=pd.DataFrame(dict(taken=taken)))
    g = burst_ids(idx, [0, 1, 2, 3, 4])
    assert g[0] == g[1]                  # same scene, 2 min apart: one stack
    assert g[2] != g[0]                  # same scene 3 h later: its own moment
    assert g[3] != g[0]                  # different scene, same minute
    assert g[4] not in g[:4]             # undated: never stacked
    assert g == sorted(g) or g[0] == 0   # numbered in album order: the first item is the cover
