"""Side-by-side per-category F1 (final round) of several scored runs of the SAME split, with paired differences.

Each run dir must hold scored.json (score.py). The first run is the reference: for every other run the table gives the
per-category mean F1 and the paired difference vs the reference with a 95% bootstrap interval over requests (2,000
resamples of the request set, same requests in both runs), plus how many requests got better / worse by > 0.05.
Usage: python eval/complex_bench/compare.py <name=run_dir> <name=run_dir> ... [--out=file]
"""
import json
import sys
from pathlib import Path

import numpy as np

CATS = ["a", "b", "c", "d", "e", "f", "g", "h"]
NAMES = {"a": "objects/scenes", "b": "attributes", "c": "person+look", "d": "time/place/event", "e": "multi-hop",
         "f": "vague 1 photo", "g": "two albums", "h": "follow-ups"}


def load(d):
    return {r["qid"]: r for r in json.loads((Path(d) / "scored.json").read_text())}


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    R = [(n, load(d)) for n, d in runs]
    ref_name, ref = R[0]
    common = sorted(set.intersection(*[set(r) for _, r in R]))
    rng = np.random.default_rng(0)
    lines = [f"Complex bench, final-round F1 per category on {len(common)} requests (reference = {ref_name}); "
             f"diff = paired mean difference vs the reference [95% bootstrap over requests]; +/- = requests better/worse by > 0.05"]
    head = "cat (n)".ljust(24) + "".join(n[:14].rjust(16) for n, _ in R)
    lines.append(head)
    groups = [(c, [q for q in common if ref[q]["cat"] == c]) for c in CATS] + \
             [("ALL", common), ("ALL verified", [q for q in common if ref[q].get("verified", True)])]
    for c, qs in groups:
        if not qs:
            continue
        row = f"{c} {NAMES.get(c, '')} ({len(qs)})".ljust(24)
        base = np.array([ref[q]["final"]["f1"] for q in qs])
        for n, r in R:
            v = np.array([r[q]["final"]["f1"] for q in qs])
            row += f"{v.mean():.3f}".rjust(16)
        lines.append(row)
        for n, r in R[1:]:
            v = np.array([r[q]["final"]["f1"] for q in qs])
            d = v - base
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(" " * 6 + f"{n} - {ref_name}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  +{int((d > 0.05).sum())} / "
                         f"-{int((d < -0.05).sum())}")
    f_hit = [(n, sum(r[q]["final"].get("hit", False) for q in common if r[q]["cat"] == "f")) for n, r in R]
    lines.append("f hit (target photo returned): " + ", ".join(f"{n} {h}" for n, h in f_hit))
    lines.append("nothing returned: " + ", ".join(f"{n} {sum(r[q]['final']['n_got'] == 0 for q in common)}" for n, r in R))
    txt = "\n".join(lines)
    print(txt)
    if out:
        Path(out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
