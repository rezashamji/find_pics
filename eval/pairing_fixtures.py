"""Golden cases for the Swift pairing split (FindPicsCore splitPair) = engine._split_pair.
Keeps the inputs of the 7 original cases (public eras Pratt / Hill / Rogen + 4 synthetic), recomputes their expected
answers with the current engine, and adds SYNTHETIC regression cases shaped like the 10-09 collapse (RESULTS 39): one
long event whose 137 photos share one median value (a tie spike) inside the heavier group, a second smaller group
(fit) far below, drawn from a seeded generator; plus one hand-built case (one_event_near_tie) that only the
one-event test catches. Everything is synthetic: nothing here is derived from anyone's photos.
Usage: python eval/pairing_fixtures.py   (rewrites ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/pairing.json)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from findpics import engine as E   # noqa: E402

FIX = ROOT / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/pairing.json"


def spike_case(name, seed, gap, sp_up, bl_mu, bl_sd, at=-11.75, near=0):
    """x = logit(pA) - logit(pB), shaped like the collapse: fit group 120 ~ N(-20, 1.6) + one 30-photo event tied at -21;
    heavier group = 30 ~ N(bl_mu, bl_sd) below, ONE 137-photo event tied at `at` (its median), 75 just above it
    (at + gap + |N(0, sp_up)|). The quartile start sits on the tie (75th percentile = `at`). pB = sigmoid(10), so pA
    stays inside the logit clip. The two tied blocks are events e0 (137) and e1 (30); every other photo is its own
    event. near > 0: that many of the 75 above-tie photos (own events) sit within ~0.1 of the tie instead, so a collapsed
    component is a NEAR-point mass (spread above the floor) that only the one-event test recognises."""
    r = np.random.default_rng(seed)
    x = np.r_[r.normal(-20, 1.6, 120), np.full(30, -21.0), r.normal(bl_mu, bl_sd, 30), np.full(137, at),
              at + gap + np.abs(r.normal(0, sp_up, 75))]
    if near:
        x[317:317 + near] = at + r.normal(0, 0.1, near)
    ids = [f"s{k:03d}" for k in range(len(x))]
    ev = {i: ("e1" if 120 <= k < 150 else "e0" if 180 <= k < 317 else f"e_{k}") for k, i in enumerate(ids)}
    sig = lambda v: 1 / (1 + np.exp(-v))
    return dict(name=name, pA={i: float(sig(v + 10)) for i, v in zip(ids, x)}, pB={i: float(sig(10.0)) for i in ids},
                eventOf=ev)


def one_event_case(name="one_event_near_tie"):
    """Deterministic (no random numbers). x = logit(pA) - logit(pB):
    group B (album 1): 120 photos, each its own event, x evenly spaced on [-24, -16];
    group A (album 0): ONE event of 137 photos tied at x = 0, 40 single-photo events evenly spread on [-0.1, 0.1] around
    it, and 60 single-photo events evenly spaced on [-8, -2].
    From the quartile start EM ends on a near-point mass: the tie + its 40 neighbours (sd 0.027, ABOVE the 0.01 floor,
    so the floor test does not fire), 0.78 of that component's weight in the one tied event, vs one wide component
    (sd 7.5) holding B and A's spread photos. It has the higher likelihood, so without events it is chosen. With events
    the one-event test (>= 0.75) rejects it and the two-means fit wins: B vs A (sd 2.33 / 2.35, largest one-event
    share 0.58). pB = sigmoid(12) keeps pA inside the logit clip."""
    x = np.r_[np.linspace(-24, -16, 120), np.zeros(137), np.linspace(-0.1, 0.1, 40), np.linspace(-8, -2, 60)]
    ids = [f"o{k:03d}" for k in range(len(x))]
    ev = {i: ("tied" if 120 <= k < 257 else f"e{k}") for k, i in enumerate(ids)}
    sig = lambda v: 1 / (1 + np.exp(-v))
    return dict(name=name, pA={i: float(sig(v + 12)) for i, v in zip(ids, x)}, pB={i: float(sig(12.0)) for i in ids},
                eventOf=ev)


def expected(c):
    A = pd.DataFrame(dict(item_id=list(c["pA"]), p_attr=list(c["pA"].values())))
    B = pd.DataFrame(dict(item_id=list(c["pB"]), p_attr=list(c["pB"].values())))
    s = E._split_pair([A, B], c.get("eventOf"))
    return None if s is None else {i: (None if v is None else v[0]) for i, v in s.items()}


if __name__ == "__main__":
    cases = [{k: v for k, v in c.items() if k != "expected"} for c in json.loads(FIX.read_text())][:7]
    cases.append(one_event_case())
    # the old single (quartile) start collapsed onto the tie here (sd 0.01: 'that event' vs everything else); the
    # two-means start finds the real groups (search: 6 of 2,000 such draws collapsed the old start, eval note RESULTS 39)
    cases += [spike_case("tied_spike_quartile_collapse", 536, 0.113, 1.271, -14.165, 0.765),
              # every start collapses (upper group hugs the tie): last resort keeps the best degenerate fit (old answer)
              spike_case("tied_spike_all_starts_collapse", 210, 0.04, 0.70, -14.18, 1.13)]
    for c in cases:
        c["expected"] = expected(c)
        e = c["expected"]
        print(c["name"], len(c["pA"]), "None" if e is None else
              f"A {sum(v == 0 for v in e.values())} B {sum(v == 1 for v in e.values())} neither {sum(v is None for v in e.values())}")
    FIX.write_text(json.dumps(cases))
