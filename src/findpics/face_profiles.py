"""Face-model profiles: THE place that says which face model is used and which cuts go with it.

Every face cut depends on the model: cosine scores of two models are on different scales (AuraFace + flip puts
different people higher than buffalo_l does), so a cut is only meaningful together with the model that set it.
The phone's twin is FindPicsCore/FaceProfile.swift; tests/test_face_profiles.py + the Swift FaceProfileTests check
that both tables are identical (fixture ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/face_profiles.json).

Fields (cosine cuts):
  group      face_groups ("the faces I see most often") and assignedGroups (a group already belongs to a named person)
  expand     expand_refs: faces at least this close to a reference become references (query-time clustering)
  accept     a photo is the person (item_person_scores)
  other      other_identities: a face group whose faces match the person below this on average is SOMEONE ELSE, and a
             face closer to such a group than to the person does not count ("closer to someone else" rule)
  maybe      server only: [maybe, accept) = "possible" list for the user to confirm, never auto-accepted
  pick_floor pickRefFaces: below this best mean cosine the picked photos share no face (take the largest face)
  consensus  refs_from_items: references whose consensus with the other tagged photos is below this are dropped
How each was set: eval/face_calibrate.py, RESULTS section 36 (AuraFace) / JOURNAL 10-02..10-04 (buffalo_l).
"""
from __future__ import annotations

import os

FACE_PROFILES = {
    # InsightFace buffalo_l w600k_r50. NON-COMMERCIAL weights: dev / personal testing on the server only, not shipped.
    "buffalo_l": dict(id="buffalo_l", group=0.55, expand=0.55, accept=0.40, other=0.40, maybe=0.30, pick_floor=0.30,
                      consensus=0.20),
    # fal AuraFace-v1 (glintr100, Apache-2.0) + flip averaging: the model the iPhone app ships (Reza, 10-07).
    "auraface_flip": dict(id="auraface_flip", group=0.62, expand=0.60, accept=0.53, other=0.53, maybe=0.42,
                          pick_floor=0.42, consensus=0.30),
}
SHIPPED = "auraface_flip"


def face_profile(name: str | None = None) -> dict:
    """Profile of a face model (default: the server's FP_FACE_MODEL, as findpics.models.DEFAULT_FACE)."""
    name = name or os.environ.get("FP_FACE_MODEL", "buffalo_l")
    if name not in FACE_PROFILES:
        raise ValueError(f"no thresholds for face model {name!r} (known: {sorted(FACE_PROFILES)}); calibrate it first "
                         f"(eval/face_calibrate.py)")
    return FACE_PROFILES[name]


def index_profile(idx) -> dict:
    """Profile of the model that made THIS index's face vectors (stats.json face_model; old indexes: buffalo_l)."""
    return face_profile(getattr(idx, "face_model", None) or "buffalo_l")
