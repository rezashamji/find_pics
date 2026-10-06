"""Golden cases for the Swift port of the subject path's vector ranking (converse._dba k=2 + mean over references +
best unit per item). Synthetic clustered vectors (no real data)."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from findpics.converse import _dba

rng = np.random.default_rng(7)
C = rng.normal(size=(12, 32)).astype(np.float32)
units = (C[rng.integers(0, 12, 400)] + 0.6 * rng.normal(size=(400, 32))).astype(np.float32)
units /= np.linalg.norm(units, axis=1, keepdims=True)
unit_item = np.sort(rng.integers(0, 300, 400)); unit_item[:300] = np.arange(300); unit_item = np.sort(unit_item)
refs = (C[3] + 0.6 * rng.normal(size=(3, 32))).astype(np.float32); refs /= np.linalg.norm(refs, axis=1, keepdims=True)
Xd, Vd = _dba(units, refs, k=2)
s = (Xd @ Vd.T).mean(1)
scores = np.full(300, -np.inf, np.float32); np.maximum.at(scores, unit_item, s)
json.dump(dict(units=units.tolist(), unitItem=unit_item.tolist(), refs=refs.tolist(), k=2, scores=scores.tolist()),
          open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/subject.json", "w"))
print("SUBJECT_FIXTURE", len(units), "units", len(scores), "items")
