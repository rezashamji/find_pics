"""Engine logic on a synthetic index with a fake judge (no GPU, no images decoded)."""
import numpy as np
import pandas as pd
import pytest

from findpics import engine as E
from findpics.planner import AlbumSpec
from findpics.store import Index


class FakeEnc:
    name = "fake"

    def texts(self, t):
        v = np.zeros((len(t), 4), np.float32); v[:, 0] = 1
        return v


class FakeJudge:
    """Says yes iff the item is a true positive; lets tests control the answer exactly."""

    def __init__(self, truth):
        self.truth = truth; self.calls = 0

    def p_yes(self, ims, q):
        self.calls += len(ims)
        return [1.0 if self.truth[int(im)] else 0.0 for im in ims]


@pytest.fixture
def lib(monkeypatch):
    rng = np.random.default_rng(0)
    N = 3000
    truth = np.zeros(N, bool); truth[rng.choice(N, 150, replace=False)] = True
    clip = np.zeros((N, 4), np.float16)
    clip[:, 0] = (rng.normal(0, 0.1, N) + 0.5 * truth).astype(np.float16)  # positives score higher, overlapping
    items = pd.DataFrame(dict(item_id=[f"i{i}" for i in range(N)], path=[f"/x/{i}.jpg" for i in range(N)],
                              media="photo", taken="2020-01-01T00:00:00+00:00"))
    units = pd.DataFrame(dict(item_id=items.item_id, frame_t=-1.0, item_row=np.arange(N)))
    idx = Index(None, items, units, clip, pd.DataFrame(columns=["item_row", "unit_row"]), np.zeros((0, 512), np.float16), pd.DataFrame())
    # the "image" handed to the judge is just the row number
    monkeypatch.setattr(E, "_frame_for", lambda idx, r, fr: r)
    monkeypatch.setattr(E, "_boxed", lambda idx, im, fr: im)
    return idx, truth


def test_concept_query_finds_positives_and_certificate_holds(lib):
    idx, truth = lib
    J = FakeJudge(truth)
    spec = AlbumSpec(name="x", looks=["thing"], judge_question="thing?")
    th = E.Thresholds(head_size=200, head_chunk=100, tail_budget=500)
    res = E.run_album(idx, spec, FakeEnc(), J, None, th=th)
    found = set(res.returned.item_row)
    true_recall = len(found & set(np.where(truth)[0])) / truth.sum()
    assert len(found - set(np.where(truth)[0])) == 0          # perfect judge -> no false positives
    assert res.cert["recall_lower"] <= true_recall + 1e-9       # bound must not overclaim
    assert J.calls <= len(truth) + 1                            # judge never called more than once per item


def test_adaptive_head_extends_while_finding(lib):
    idx, truth = lib
    th_small = E.Thresholds(head_size=50, head_chunk=50, head_max=3000, head_stop_rate=0.03, tail_budget=100)
    res = E.run_album(idx, AlbumSpec(name="x", looks=["thing"], judge_question="q"), FakeEnc(), FakeJudge(truth), None, th=th_small)
    n_head = int((res.judged["where"] == "head").sum())
    assert n_head > 50  # kept going because early chunks were dense with matches


def test_date_scope_excludes_out_of_range(lib):
    idx, truth = lib
    spec = AlbumSpec(name="x", looks=["thing"], judge_question="q", date_from="2021-01-01")
    res = E.run_album(idx, spec, FakeEnc(), FakeJudge(truth), None)
    assert res.n_in_scope == 0 and len(res.returned) == 0


def test_make_exclusive_keeps_shared_photo_in_more_confident_album():
    from types import SimpleNamespace as NS
    a = NS(spec=AlbumSpec(name="heavier", person="Reza", judge_question="heavy?"), report="Album 'heavier': 2 items.",
           returned=pd.DataFrame(dict(item_id=["x", "y"], p_attr=[0.38, 0.9])))
    b = NS(spec=AlbumSpec(name="fit", person="Reza", judge_question="fit?"), report="Album 'fit': 2 items.",
           returned=pd.DataFrame(dict(item_id=["x", "z"], p_attr=[0.82, 0.7])))
    a.judged = pd.DataFrame(dict(item_id=["x", "y", "w"], p_attr=[0.38, 0.9, 0.35]))
    b.judged = pd.DataFrame(dict(item_id=["x", "z", "w", "y"], p_attr=[0.82, 0.7, 0.82, 0.1]))
    a.returned = pd.concat([a.returned, pd.DataFrame(dict(item_id=["w"], p_attr=[0.35]))], ignore_index=True)
    a.report = "Album 'heavier': 3 items."
    E.make_exclusive([a, b])
    # w is not in 'fit' (capped) but the judge rated it fit 0.82 > heavier 0.35 -> removed from heavier
    assert list(a.returned.item_id) == ["y"] and list(b.returned.item_id) == ["x", "z"]
    assert "'heavier': 1 items." in a.report


def test_make_exclusive_uses_within_person_rank_not_raw_score():
    # Pratt-like: the judge's raw P(fit) is high on everything, but within-person RANK says photo x is among his heaviest
    from types import SimpleNamespace as NS
    j_h = pd.DataFrame(dict(item_id=["x", "y"], p_attr=[0.30, 0.10], rel=[1.0, 0.5]))
    j_f = pd.DataFrame(dict(item_id=["x", "y"], p_attr=[0.80, 0.95], rel=[0.5, 1.0]))
    a = NS(spec=AlbumSpec(name="heavier", person="P", judge_question="h?"), report="Album 'heavier': 1 items.",
           returned=j_h[j_h.item_id == "x"].copy(), judged=j_h)
    b = NS(spec=AlbumSpec(name="fit", person="P", judge_question="f?"), report="Album 'fit': 2 items.",
           returned=j_f.copy(), judged=j_f)
    E.make_exclusive([a, b])
    assert list(a.returned.item_id) == ["x"]          # raw scores would have moved x to 'fit' (0.80 > 0.30)
    assert list(b.returned.item_id) == ["y"]


def test_store_subset_remaps_rows():
    from findpics.store import subset
    items = pd.DataFrame(dict(item_id=list("abcd"), path=list("abcd"), media="photo", taken="2020-01-01T00:00:00+00:00"))
    units = pd.DataFrame(dict(item_id=list("abcdd"), frame_t=-1.0, item_row=[0, 1, 2, 3, 3]))
    clip = np.arange(5 * 2, dtype=np.float16).reshape(5, 2)
    faces = pd.DataFrame(dict(item_row=[1, 3], unit_row=[1, 4], item_id=["b", "d"]))
    fe = np.array([[1] * 512, [2] * 512], np.float16)
    sub = subset(Index(None, items, units, clip, faces, fe, pd.DataFrame()), [1, 3])
    assert list(sub.items.item_id) == ["b", "d"] and list(sub.units.item_row) == [0, 1, 1]
    assert sub.clip.tolist() == [[2, 3], [6, 7], [8, 9]]
    assert list(sub.faces.item_row) == [0, 1] and list(sub.faces.unit_row) == [0, 2] and sub.face_emb[1, 0] == 2
