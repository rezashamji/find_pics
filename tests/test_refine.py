"""Refinement edits on an existing album (fake judge / fake LLM; no GPU). Originals must never change."""
import json
import os

import numpy as np
import pandas as pd

from findpics.albums import write_folder_album
from findpics.refine import EditPlan, Op, apply_ops, plan_edits, write_edit
from findpics.store import Index

N = 8
items = pd.DataFrame(dict(item_id=[f"i{k}" for k in range(N)], path=[f"/x/{k}.jpg" for k in range(N)], media="photo",
                          taken="2020-01-01T00:00:00+00:00"))
units = pd.DataFrame(dict(item_id=items.item_id, frame_t=-1.0, item_row=np.arange(N)))
clip = np.zeros((N, 3), np.float16); clip[:4, 0] = 1; clip[4:, 1] = 1; clip[3] = [0.7, 0.7, 0]   # two look-alike groups
IDX = Index(None, items, units, clip, pd.DataFrame(), np.zeros((0, 512), np.float16), pd.DataFrame())
blurry = {1, 5}                                                    # fake truth for "is it blurry?"
fake = lambda idx, J, rows, q: [1.0 if (("blurry" in q) and r in blurry) or ("album q" in q and r >= 4) else 0.0 for r in rows]
ALBUM = pd.DataFrame(dict(item_row=[0, 1, 2, 3], item_id=["i0", "i1", "i2", "i3"], path=[f"/x/{k}.jpg" for k in range(4)]))


def test_remove_if_and_keep_if():
    out, log = apply_ops(IDX, ALBUM, EditPlan(ops=[Op(op="remove_if", question="Is it blurry?")]), None, None, judge_rows=fake)
    assert list(out.item_id) == ["i0", "i2", "i3"] and "1 removed" in log[0]
    out, _ = apply_ops(IDX, ALBUM, EditPlan(ops=[Op(op="keep_if", question="Is it blurry?")]), None, None, judge_rows=fake)
    assert list(out.item_id) == ["i1"]


def test_remove_like_drops_lookalikes():
    out, log = apply_ops(IDX, ALBUM, EditPlan(ops=[Op(op="remove_like", ids=["i0"], sim=0.95)]), None, None, judge_rows=fake)
    assert list(out.item_id) == ["i3"]          # i0 itself + exact look-alikes i1, i2; i3 (0.71 similar) stays


def test_add_like_judges_with_album_question():
    out, log = apply_ops(IDX, ALBUM, EditPlan(ops=[Op(op="add_like", ids=["i3"], k=3)]), None, "album q", judge_rows=fake)
    assert set(out.item_id) - set(ALBUM.item_id) <= {"i4", "i5", "i6", "i7"} and len(out) > len(ALBUM)


def test_plan_edits_only_uses_selected_ids():
    reply = json.dumps({"album": "A", "ops": [{"op": "remove_like", "ids": ["i0", "i9"]}]})
    P = plan_edits("get rid of ones like this", lambda p: reply, ["A"], ["i0"])
    assert P.ops[0].ids == ["i0"]             # i9 was never selected -> dropped


def test_write_edit_versions_and_never_touches_originals(tmp_path):
    photos = []
    for k in range(4):
        p = tmp_path / f"{k}.jpg"; p.write_bytes(b"x"); photos.append(str(p))
    album = pd.DataFrame(dict(item_id=[f"i{k}" for k in range(4)], path=photos))
    d = write_folder_album("A", album, tmp_path / "albums")
    write_edit(d, album.iloc[[0, 2]], ["removed 2"])
    assert all(os.path.exists(p) for p in photos)                       # originals intact
    assert sorted(os.readlink(l) for l in d.iterdir() if l.is_symlink()) == sorted([photos[0], photos[2]])
    assert (d / "manifest.v1.json").exists() and json.loads((d / "manifest.json").read_text())["edits"] == [["removed 2"]]
