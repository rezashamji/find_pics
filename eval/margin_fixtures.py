"""Golden cases for the Swift rank-margin pairing fallback: the same rule as engine.make_exclusive's margin branch
(rel = within-person rank, margin 0.3; one-album photos keep their album if rel > 0.5). Synthetic scores."""
import json
import numpy as np

rng = np.random.default_rng(3)
ids = [f"p{k}" for k in range(40)]
pa = {i: float(rng.random()) for i in ids[:35]}
pb = {i: float(rng.random()) for i in ids[5:]}


def rel(p):
    srt = np.sort(list(p.values()))
    return {k: float(np.searchsorted(srt, v, side="right") / len(srt)) for k, v in p.items()}


ra, rb = rel(pa), rel(pb); out = {}
for i in set(pa) | set(pb):
    if i in ra and i in rb:
        sc = [ra[i], rb[i]]; k = int(np.argmax(sc))
        out[i] = k if sc[k] - sc[1 - k] >= 0.3 else None
    else:
        r, k = (ra[i], 0) if i in ra else (rb[i], 1)
        out[i] = k if r > 0.5 else None
json.dump(dict(pA=pa, pB=pb, expected=out), open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/margin.json", "w"))
print("MARGIN_FIXTURE", len(out), sum(v is None for v in out.values()), "neither")
