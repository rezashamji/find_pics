"""Face-model profiles: one table, shared with the phone; cuts follow the index's face model; no mixing of models."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from findpics.face_profiles import FACE_PROFILES, SHIPPED, face_profile, index_profile

FIX = Path(__file__).resolve().parents[1] / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures"


def test_phone_table_is_this_table():
    assert json.loads((FIX / "face_profiles.json").read_text()) == FACE_PROFILES   # FaceProfileTests checks Swift == this
    assert json.loads((FIX / "faces.json").read_text())["profile"] == SHIPPED == "auraface_flip"


def test_cuts_follow_the_index_model():
    assert index_profile(NS())["id"] == "buffalo_l"                 # indexes from before 10-07
    assert index_profile(NS(face_model="auraface_flip"))["accept"] == FACE_PROFILES["auraface_flip"]["accept"]
    with pytest.raises(ValueError):
        face_profile("unknown_model")
    from findpics.engine import Thresholds, person_cuts
    assert person_cuts(NS(face_model="buffalo_l"), Thresholds()) == (0.40, 0.30)
    a = FACE_PROFILES["auraface_flip"]
    assert person_cuts(NS(face_model="auraface_flip"), Thresholds()) == (a["accept"], a["maybe"])
    assert person_cuts(NS(face_model="auraface_flip"), Thresholds(person_accept=0.9))[0] == 0.9


def test_people_sheet_of_another_model_is_ignored(tmp_path):
    from findpics.people import other_identities, save_groups
    e = np.eye(4, dtype=np.float32)
    old = NS(face_emb=e.astype(np.float16), faces=pd.DataFrame(dict(item_row=[0, 1, 2, 3])), n_items=4, root=tmp_path)
    save_groups(old, [dict(faces=[2, 3])], tmp_path)                 # sheet made on a buffalo_l index
    assert len(other_identities(old, e[:1])) == 2
    new = NS(face_emb=old.face_emb, faces=old.faces, n_items=4, root=tmp_path, face_model="auraface_flip")
    assert len(other_identities(new, e[:1])) == 0                    # never compared with AuraFace vectors


def test_index_refuses_shards_of_two_face_models(tmp_path):
    from findpics import store
    pd.DataFrame(dict(item_id=["a", "b"])).to_parquet(tmp_path / "items.parquet")
    for k, m in enumerate(["buffalo_l", "auraface_flip"]):
        d = tmp_path / "shards" / f"{k:04d}"; d.mkdir(parents=True)
        (d / "stats.json").write_text(json.dumps(dict(clip_models=["c"], face_model=m))); (d / "DONE").write_text("")
    with pytest.raises(ValueError, match="different face models"):
        store.load(tmp_path, clip_model="c")
