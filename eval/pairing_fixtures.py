"""Golden cases for the Swift pairing split (FindPicsCore splitPair) = engine._split_pair.
Keeps the inputs of the 7 original cases (public eras Pratt / Hill / Rogen + 4 synthetic), recomputes their expected
answers with the current engine, and adds SYNTHETIC regression cases shaped like the 10-09 collapse (RESULTS 39): one
long event whose 137 photos share one median value (a tie spike) inside the heavier group, a second smaller group
(fit) far below, drawn from a seeded generator; plus ONE anonymised case ("anon_near_tie_one_event"): logit differences
of a saved demo run + logit noise N(0, 0.1), event medians, rounded to 0.001, ids and event labels replaced by shuffled
anonymous ones (values only: no photo ids, names or labels). It is the case only the one-event test catches (the
collapsed component's spread is 0.11, above the floor). Its inputs are kept from the fixture file like the 7 originals.
Usage: python eval/pairing_fixtures.py [--add-anon case.json]   (rewrites .../Fixtures/pairing.json)"""
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


def expected(c):
    A = pd.DataFrame(dict(item_id=list(c["pA"]), p_attr=list(c["pA"].values())))
    B = pd.DataFrame(dict(item_id=list(c["pB"]), p_attr=list(c["pB"].values())))
    s = E._split_pair([A, B], c.get("eventOf"))
    return None if s is None else {i: (None if v is None else v[0]) for i, v in s.items()}


if __name__ == "__main__":
    old = [{k: v for k, v in c.items() if k != "expected"} for c in json.loads(FIX.read_text())]
    cases = old[:7] + [c for c in old if c["name"].startswith("anon_")]
    if "--add-anon" in sys.argv:
        add = json.loads(Path(sys.argv[sys.argv.index("--add-anon") + 1]).read_text())
        cases = [c for c in cases if c["name"] != add["name"]] + [add]
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
