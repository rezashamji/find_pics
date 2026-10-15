"""Person searches' first answer (findpics.person_first, RESULTS 62): the ports and the rule's exactness at the end."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from findpics import engine as E
from findpics import person_first as PF
from findpics.planner import AlbumSpec
from findpics.store import Index

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures"
SHIPPED = "timed round_calls=120 inter=0 topup=30 head=150 a1=0.2 fast_stop=480 stop_topup=150"


def test_fnv1a64_known_vectors():
    assert PF.fnv1a64("") == 0xcbf29ce484222325
    assert PF.fnv1a64("a") == 0xaf63dc4c8601ec8c
    assert PF.fnv1a64("foobar") == 0x85944171f73967e8


def test_walk_key_mixes_the_last_bytes():
    """RESULTS 62b: ids differing only in their last characters must not sort next to each other."""
    assert PF.fmix64(0) == 0
    ids = [f"e{e:05d}p{j:03d}" for e in range(300) for j in range(5)]
    o = sorted(ids, key=lambda i: PF.walk_key(0, i))
    same = np.mean([a[:6] == b[:6] for a, b in zip(o[:-1], o[1:])])
    assert same < 0.02                                   # random: 4/1499 = .003; plain FNV-1a: ~.8
    plain = sorted(ids, key=lambda i: PF.fnv1a64(f"0|{i}"))
    assert np.mean([a[:6] == b[:6] for a, b in zip(plain[:-1], plain[1:])]) > 0.5


def test_stream_timed_is_the_replay_mirror():
    """Same calls, rounds, bounds and first answer as eval/tune_stop_time.sim_timed (= FindPicsCore streamTimed)."""
    sys.path.insert(0, str(ROOT / "eval"))
    import tune_stop_time as TT
    _, r = TT.parse(SHIPPED)
    rng = np.random.default_rng(62)
    for _ in range(12):
        n = int(rng.integers(30, 2500))
        y = rng.random(n) < np.linspace(rng.uniform(0.2, 0.9), 0.005, n)
        perm = rng.permutation(n)
        o_ex, r_ex = TT.sim_timed(y, r, 0, perm=perm, exhaustive=True)
        rounds = list(PF.stream_timed(n, lambda q: float(y[q]), PF.PHONE_FAST, perm=perm))
        assert rounds[-1].calls == list(o_ex)
        assert [(a[0], a[2], a[3]) for a in r_ex] == [(b.judged, b.k, b.last) for b in rounds]
        for a, b in zip(r_ex, rounds):
            assert a[1]["missed_upper"] == pytest.approx(b.cert["missed_upper"], abs=1e-9)
            assert a[1]["found"] == b.cert["found"]
        o_f, r_f = TT.sim_timed(y, r, 0, perm=perm)
        k = next(i for i, b in enumerate(rounds) if PF.first_answer_done(b))
        assert len(r_f) == k + 1 and r_f[-1][0] == rounds[k].judged


def test_weights_of_one_change_nothing():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(20, 300))
        x = np.r_[rng.normal(-2, rng.uniform(0.3, 2), n // 2), rng.normal(rng.uniform(0, 4), 1, n - n // 2)]
        g = rng.integers(0, max(2, n // 4), n)
        a = E._two_groups(x, groups=g); b = E._two_groups(x, groups=g, weights=np.ones(n))
        assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]


def test_walk_order_caps_events_then_takes_the_rest():
    rng = np.random.default_rng(5)
    ids = [f"p{k:04d}" for k in range(600)]
    ev = {i: f"e{int(e)}" for i, e in zip(ids, rng.zipf(1.6, 600) % 90)}
    o = PF.walk_order(ids, ev, seed=7)
    assert sorted(o) == list(range(600)) and o == PF.walk_order(ids, ev, seed=7) and o != PF.walk_order(ids, ev, seed=8)
    n_first = sum(min(3, c) for c in pd.Series(ev).value_counts())
    first = pd.Series([ev[ids[k]] for k in o[:n_first]]).value_counts()
    assert first.max() <= 3 and set(first.index) == set(ev.values())
    plain = PF.walk_order(ids, ev, seed=7, cap=None)
    assert plain == sorted(range(600), key=lambda k: (PF.walk_key(7, ids[k]), ids[k]))
    rest = [k for k in plain if k not in set(o[:n_first])]
    assert o[n_first:] == rest


def _events(rng, n):
    return {f"i{k:04d}": f"e{int(e)}" for k, e in enumerate(np.cumsum(rng.random(n) < 0.15))}


def test_size_walk_reaches_the_largest_events_first():
    ids = [f"p{k:03d}" for k in range(60)]
    ev = {i: ("big" if k < 30 else "mid" if k < 45 else f"s{k}") for k, i in enumerate(ids)}
    o = PF.walk_order(ids, ev, seed=1, by_size=True)
    assert sorted(o) == list(range(60))
    assert ev[ids[o[0]]] == "big" and ev[ids[o[1]]] == "mid"          # pass 1: one photo per event, largest first
    assert len({ev[ids[k]] for k in o[:17]}) == 17                      # 17 events, each reached once
    assert [ev[ids[k]] for k in o[17:21]] == ["big", "big", "mid", "mid"]   # pass 2: photos 2-3, same order


def test_single_look_with_everything_judged_is_the_full_rule():
    rng = np.random.default_rng(11)
    for _ in range(15):
        n = int(rng.integers(5, 400))
        ev = _events(rng, n); ids = list(ev)
        p = {i: float(v) for i, v in zip(ids, np.round(rng.beta(0.5, 0.5, n), int(rng.integers(1, 6))))}  # ties too
        got = PF.single_partial(ids, ev, p)
        pv = np.array([p[i] for i in ids]); rel = np.searchsorted(np.sort(pv), pv, side="right") / n
        want = E.person_keep(pv, rel, "Does the person look tired?", E.Thresholds())
        assert [got[i] for i in ids] == list(want)


def test_single_look_partial_sorts_whole_events_from_judged_photos():
    ids = ["a1", "a2", "a3", "b1", "b2", "c1"]
    ev = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "c1": "C"}
    got = PF.single_partial(ids, ev, {"a1": 0.9, "b1": 0.1})
    # a1 stands for 3 photos (rel of 0.9 = 1.0), b1 for 2 (rel of 0.1 = 2/5): event A in, B out, C not sorted yet
    assert got == {"a1": True, "a2": True, "a3": True, "b1": False, "b2": False}
    assert PF.single_partial(ids, ev, {"a1": 0.9, "b1": 0.1}, propagate=False) == {"a1": True, "b1": False}


def _pools(pa, pb):
    return [pd.DataFrame(dict(item_id=list(pa), p_attr=list(pa.values()))),
            pd.DataFrame(dict(item_id=list(pb), p_attr=list(pb.values())))]


def test_pair_with_everything_judged_is_the_engine_split():
    cases = json.loads((FIX / "pairing.json").read_text())
    rng = np.random.default_rng(13)
    for _ in range(10):
        n = int(rng.integers(30, 300)); ev = _events(rng, n); ids = list(ev)
        hi = np.cumsum(rng.random(n) < 0.02) % 2 == 0           # two eras
        pa = {i: float(1 / (1 + np.exp(-(1.5 if h else -1.5) - rng.normal(0, 1.2)))) for i, h in zip(ids, hi)}
        pb = {i: float(1 / (1 + np.exp(-(-1.5 if h else 1.5) - rng.normal(0, 1.2)))) for i, h in zip(ids, hi)}
        cases.append(dict(name="rand", pA=pa, pB=pb, eventOf=ev))
    n_split = 0
    for c in cases:
        ids = sorted(c["pA"]); ev = c.get("eventOf")
        sp = E._split_pair(_pools({i: c["pA"][i] for i in ids}, {i: c["pB"][i] for i in ids}), ev)
        got = PF.pair_partial(ids, ev, c["pA"], c["pB"])
        if sp is None:
            assert got is None, c["name"]
            continue
        n_split += 1
        assert got == {i: (v[0] if v else None) for i, v in sp.items()}, c["name"]
    assert n_split >= 8


def test_pair_partial_labels_an_event_from_its_judged_photos():
    rng = np.random.default_rng(17)
    ids, ev, pa, pb = [], {}, {}, {}
    for e in range(40):                    # 40 events of 6 photos: even events 'A' (heavier), odd 'B'
        for j in range(6):
            i = f"e{e:02d}p{j}"; ids.append(i); ev[i] = f"E{e}"
            s = 2.0 if e % 2 == 0 else -2.0
            pa[i] = float(1 / (1 + np.exp(-s - rng.normal(0, .5)))); pb[i] = float(1 / (1 + np.exp(s - rng.normal(0, .5))))
    judged = [i for i in ids if i.endswith("p0")]          # one photo per event
    got = PF.pair_partial(ids, ev, {i: pa[i] for i in judged}, {i: pb[i] for i in judged})
    assert got is not None and len(got) == len(ids)          # every event reached -> every photo sorted
    assert all(got[i] == (0 if int(ev[i][1:]) % 2 == 0 else 1) for i in ids)
    few = PF.pair_partial(ids, ev, {i: pa[i] for i in judged[:19]}, {i: pb[i] for i in judged[:19]})
    assert few is None                                        # < PAIR_MIN judged photos: no split yet


def test_stream_look_updates_and_ends_at_the_full_rule():
    rng = np.random.default_rng(19)
    n = 700; ev = _events(rng, n); ids = list(ev)
    pa = {i: float(v) for i, v in zip(ids, rng.random(n))}; pb = {i: float(v) for i, v in zip(ids, rng.random(n))}
    ups = list(PF.stream_look(ids, ev, 2, lambda i, q: (pa, pb)[q][i], seed=3))
    calls = [u["calls"] for u in ups]
    assert calls[:24] == list(range(20, 481, 20)) and calls[24] == 720 and calls[-1] == 1400
    assert [u["first_answer"] for u in ups].index(True) == 23 and ups[-1]["last"]
    assert ups[-1]["albums"] == PF.pair_albums(ids, ev, pa, pb)
    one = list(PF.stream_look(ids, ev, 1, lambda i, q: pa[i], seed=3))
    assert one[-1]["albums"] == PF.single_partial(ids, ev, pa) and one[-1]["calls"] == 700


def test_fixture_matches_python():
    """The Swift golden fixture (eval/person_first/fixtures.py) is what these functions return today."""
    f = FIX / "person_first.json"
    d = json.loads(f.read_text())
    for c in d["walk"]:
        assert PF.walk_order(c["ids"], c["eventOf"], c["seed"], c["cap"], by_size=c["bySize"]) == c["order"]
    for c in d["single"]:
        got = PF.single_partial(c["ids"], c["eventOf"], c["p"], propagate=c["propagate"])
        assert got == c["expected"], c["name"]
    for c in d["pair"]:
        got = PF.pair_albums(c["ids"], c["eventOf"], c["pA"], c["pB"], propagate=c["propagate"])
        assert got == {k: v for k, v in c["expected"].items()}, c["name"]
    for c in d["fact"]:
        assert PF.fact_order(c["cos"], c.get("head")).tolist() == c["order"], c["name"]
    for c in d["stream"]:
        P = [c["pA"], c["pB"]][:c["questions"]]
        ups = list(PF.stream_look(c["ids"], c["eventOf"], c["questions"], lambda i, q: P[q][i], seed=c["seed"],
                                  by_size=c["bySize"]))
        assert [[u["calls"], u["sorted"], u["first_answer"]] for u in ups] == c["updates"], c["name"]


# ------------------------------------------------------------------------------------------------ engine integration
@pytest.fixture
def person_lib(monkeypatch):
    rng = np.random.default_rng(0)
    N = 1500
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    taken = [(t0 + pd.Timedelta(hours=float(h))).isoformat() for h in np.cumsum(rng.exponential(2.0, N))]
    items = pd.DataFrame(dict(item_id=[f"i{i}" for i in range(N)], path=[f"/x/{i}.jpg" for i in range(N)],
                              media="photo", taken=taken))
    clip = np.zeros((N, 4), np.float16)
    truth = np.zeros(N, bool); truth[rng.choice(N, 120, replace=False)] = True
    clip[:, 0] = (rng.normal(0, 0.1, N) + 0.4 * truth).astype(np.float16)
    units = pd.DataFrame(dict(item_id=items.item_id, frame_t=-1.0, item_row=np.arange(N)))
    idx = Index(None, items, units, clip, pd.DataFrame(columns=["item_row", "unit_row"]), np.zeros((0, 512), np.float16),
                pd.DataFrame())
    me = np.zeros(N, np.float32); me[rng.choice(N, 1100, replace=False)] = 0.8      # 1,100 photos of "me"
    look_p = np.clip(rng.beta(2, 2, N), 0, 1)
    monkeypatch.setattr(E, "item_person_scores", lambda idx, refs: (me.copy(), np.zeros(N, int)))
    monkeypatch.setattr(E, "person_cuts", lambda idx, th: (0.4, 0.3))
    calls = {"n": 0}

    def judge_rows(idx_, judge, rows, face_rows, question, ref_img=None, batch=96, crop_person=False):
        calls["n"] += len(rows)
        return np.asarray([(1.0 if truth[r] else 0.0) if "hat" in question else look_p[r] for r in rows], np.float32)
    monkeypatch.setattr(E, "_judge_rows", judge_rows)
    return idx, calls


class _Enc:
    name = "fake"

    def texts(self, t):
        v = np.zeros((len(t), 4), np.float32); v[:, 0] = 1
        return v


@pytest.mark.parametrize("q", ["Is the person wearing a hat?", "Does the person look tired?"])
def test_engine_person_first_ends_at_the_judge_everything_album(person_lib, q):
    idx, calls = person_lib
    spec = AlbumSpec(name="x", person="me", looks=["hat"], judge_question=q)
    refs = np.ones((1, 4), np.float32)
    calls["n"] = 0
    full = E.run_album(idx, spec, _Enc(), None, refs)
    n_full = calls["n"]
    want = set(full.returned.item_id)
    calls["n"] = 0
    first = E.run_album(idx, spec, _Enc(), None, refs, th=E.Thresholds(person_first=True))
    assert calls["n"] < n_full and calls["n"] <= 700
    assert set(first.returned.item_id) <= want or "tired" in q        # a fact's first answer: judge-yes photos only
    calls["n"] = 0
    rounds = list(E.stream_album(idx, spec, _Enc(), None, refs, th=E.Thresholds(person_first=True, stream=True)))
    assert set(rounds[-1].returned.item_id) == want and calls["n"] == n_full
