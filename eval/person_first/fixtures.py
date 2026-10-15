"""Golden fixture for FindPicsCore PersonFirst.swift (RESULTS 62): what findpics.person_first returns, on PUBLIC data
(IMDB-WIKI eras, DISBench 'tired' person sets) and synthetic sets only.

  python eval/person_first/fixtures.py   -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/person_first.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/person_first")); sys.path.insert(0, str(ROOT / "src"))
import r62  # noqa: E402
from findpics import person_first as PF  # noqa: E402

OUT = ROOT / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/person_first.json"


def synth_pair(rng, n, gap=0.15, eras=0.02, noise=1.2):
    ids = [f"s{k:04d}" for k in range(n)]
    ev = {i: f"e{int(e)}" for i, e in zip(ids, np.cumsum(rng.random(n) < gap))}
    hi = np.cumsum(rng.random(n) < eras) % 2 == 0
    sig = lambda z: float(1 / (1 + np.exp(-z)))
    pa = {i: round(sig((1.5 if h else -1.5) + rng.normal(0, noise)), 6) for i, h in zip(ids, hi)}
    pb = {i: round(sig((-1.5 if h else 1.5) + rng.normal(0, noise)), 6) for i, h in zip(ids, hi)}
    return ids, ev, pa, pb


def main():
    rng = np.random.default_rng(6262)
    D = r62.people()
    tired = r62.tired_sets(D)[:6]
    imdb = r62.imdb_sets()
    synth = [synth_pair(rng, n) for n in (60, 250, 400)]
    walk, single, pair, fact, stream = [], [], [], [], []
    # walk orders
    for (per, ids, ev, p) in tired[:3]:
        for seed, cap, bs in ((0, 3, False), (5, 1, False), (9, None, False), (4, 3, True), (6, 1, True)):
            walk.append(dict(ids=ids, eventOf=ev, seed=seed, cap=cap, bySize=bs, order=PF.walk_order(ids, ev, seed, cap, by_size=bs)))
    for ids, ev, pa, pb in synth[:2]:
        for bs in (False, True):
            walk.append(dict(ids=ids, eventOf=ev, seed=3, cap=3, bySize=bs, order=PF.walk_order(ids, ev, 3, 3, by_size=bs)))
    ids = [f"e{e:05d}p{j:03d}" for e in range(40) for j in range(1 + e % 7)]       # ids differing only at the end
    ev = {i: i[:6] for i in ids}
    for bs in (False, True):
        walk.append(dict(ids=ids, eventOf=ev, seed=0, cap=3, bySize=bs, order=PF.walk_order(ids, ev, 0, 3, by_size=bs)))
    # single partial: prefixes of the walk
    for (per, ids, ev, p) in tired:
        pos = PF.walk_order(ids, ev, 1)
        for m in (5, 20, len(ids) // 2, len(ids)):
            pj = {ids[k]: float(p[ids[k]]) for k in pos[:m]}
            for prop in (True, False):
                single.append(dict(name=f"tired {per} m{m} prop{prop}", ids=ids, eventOf=ev, p=pj, propagate=prop,
                                   expected=PF.single_partial(ids, ev, pj, propagate=prop)))
    for name, ids, ev, pa, pb, lab in imdb:
        pos = PF.walk_order(ids, ev, 2)
        for m in (10, 40, len(ids)):
            pj = {ids[k]: float(pa[ids[k]]) for k in pos[:m]}
            single.append(dict(name=f"imdb {name} m{m}", ids=ids, eventOf=None, p=pj, propagate=True,
                               expected=PF.single_partial(ids, None, pj)))
    tie_ids = [f"t{k:02d}" for k in range(30)]
    tie_p = {i: float(v) for i, v in zip(tie_ids, np.repeat([0.1, 0.5, 0.5, 0.9, 0.2], 6))}
    single.append(dict(name="ties", ids=tie_ids, eventOf={i: f"e{k // 4}" for k, i in enumerate(tie_ids)}, p=tie_p,
                       propagate=True, expected=PF.single_partial(tie_ids, {i: f"e{k // 4}" for k, i in enumerate(tie_ids)}, tie_p)))
    # pair partial / albums
    n_split = 0
    for name, ids, ev, pa, pb, lab in imdb:
        pos = PF.walk_order(ids, ev, 4)
        for m in (15, 30, 60, len(ids)):
            done = [ids[k] for k in pos[:m]]
            a = {i: float(pa[i]) for i in done}; b = {i: float(pb[i]) for i in done}
            exp = PF.pair_albums(ids, ev, a, b)
            n_split += PF.pair_partial(ids, ev, a, b) is not None
            pair.append(dict(name=f"imdb {name} m{m}", ids=ids, eventOf=None, pA=a, pB=b, propagate=True, expected=exp))
    for t, (ids, ev, pa, pb) in enumerate(synth):
        pos = PF.walk_order(ids, ev, 6)
        for m in (25, 60, 150, len(ids)):
            done = [ids[k] for k in pos[:m]]
            a = {i: pa[i] for i in done}; b = {i: pb[i] for i in done}
            for prop in (True, False):
                n_split += PF.pair_partial(ids, ev, a, b) is not None
                pair.append(dict(name=f"synth{t} m{m} prop{prop}", ids=ids, eventOf=ev, pA=a, pB=b, propagate=prop,
                                 expected=PF.pair_albums(ids, ev, a, b, propagate=prop)))
    pc = json.loads((ROOT / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/pairing.json").read_text())
    for c in pc:
        if not c.get("eventOf"):
            continue
        ids = sorted(c["pA"]); ev = c["eventOf"]
        pos = PF.walk_order(ids, ev, 8)
        for m in (40, len(ids) // 3, len(ids)):
            done = [ids[k] for k in pos[:m]]
            a = {i: c["pA"][i] for i in done}; b = {i: c["pB"][i] for i in done}
            n_split += PF.pair_partial(ids, ev, a, b) is not None
            pair.append(dict(name=f"{c['name']} m{m}", ids=ids, eventOf=ev, pA=a, pB=b, propagate=True,
                             expected=PF.pair_albums(ids, ev, a, b)))
    # fact order: public person sets (look cosine + head logit), and one with ties
    for c in ("hat", "beach", "smiling"):
        g = D[(D.cond == c) & D.person.isin(sorted(D.person.unique())[:3])]
        cos = [float(np.float32(v)) for v in g.s]; head = [float(np.float32(v)) for v in g.h]
        fact.append(dict(name=f"{c} cos", cos=cos, order=PF.fact_order(cos).tolist()))
        fact.append(dict(name=f"{c} blend", cos=cos, head=head, order=PF.fact_order(cos, head).tolist()))
    fact.append(dict(name="ties", cos=[0.2, 0.1, 0.2, 0.3, 0.1], order=PF.fact_order([0.2, 0.1, 0.2, 0.3, 0.1]).tolist()))
    # streams: update sequences
    for nm, (ids, ev, pa, pb), q, bs in (("synth pair", synth[2], 2, False), ("synth single", synth[1], 1, False),
                                         ("synth pair by size", synth[2], 2, True)):
        ups = list(PF.stream_look(ids, ev, q, lambda i, k: (pa, pb)[k][i], seed=11, by_size=bs))
        stream.append(dict(name=nm, ids=ids, eventOf=ev, pA=pa, pB=pb, questions=q, seed=11, bySize=bs,
                           updates=[[u["calls"], u["sorted"], u["first_answer"]] for u in ups],
                           final=ups[-1]["albums"] if q == 2 else {i: (0 if v else None) for i, v in ups[-1]["albums"].items()}))
    per, ids, ev, p = tired[0]
    ups = list(PF.stream_look(ids, ev, 1, lambda i, k: p[i], seed=2))
    stream.append(dict(name="tired single", ids=ids, eventOf=ev, pA={i: float(v) for i, v in p.items()}, pB={},
                       questions=1, seed=2, bySize=False, updates=[[u["calls"], u["sorted"], u["first_answer"]] for u in ups],
                       final={i: (0 if v else None) for i, v in ups[-1]["albums"].items()}))
    d = dict(walk=walk, single=single, pair=pair, fact=fact, stream=stream)
    OUT.write_text(json.dumps(d, separators=(",", ":")))
    print("FIXTURE", {k: len(v) for k, v in d.items()}, "pair cases with a split:", n_split, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
