"""RESULTS 71 shared helpers: in-process scoring identical to eval/complex_bench/score.py (final round, RESULTS-63 truth in
queries/<split>.json; checked equal to score.py's scored.json on the R69 runs), returned sets, recall / precision of one
run against another run's answer (AJ = all-judge) with score.py's own matching (union of albums; g = best album order)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/complex_bench"))
from score import prf, score_turn  # noqa: E402

SPLITS = ("train", "dev")
CATS = list("abcdefgh")
_Q = {}


def truth(split):
    if split not in _Q:
        _Q[split] = {q["qid"]: q for q in json.loads((ROOT / "eval/complex_bench/queries" / f"{split}.json").read_text())["queries"]}
    return _Q[split]


def load(d):
    """{qid: last-turn final albums (list of lists) or None if errored / missing}; requeued shards re-append: last wins."""
    R = {}
    for f in sorted((ROOT / d).glob("*part*.json")):
        for r in json.loads(f.read_text()):
            R[r["qid"]] = r
    return R


def final_albums(r, q):
    if r is None or r.get("error") or not r["turns"] or len(r["turns"]) < len(q["turns"]) or r["turns"][-1]["final"] is None:
        return None
    return r["turns"][-1]["final"]["albums"]


def f1s(d, split):
    Q = truth(split); R = load(d); out = {}
    for qid, r in R.items():
        q = Q.get(qid)
        if q is None:
            continue
        al = final_albums(r, q)
        out[qid] = 0.0 if al is None else score_turn(q["cat"], al, q["answers"][-1])["f1"]
    return out


def vs_ref(d, ref, split):
    """per request: (cat, tp, n_sys, n_ref) with score.py's matching, the REF run's final answer taken as the truth."""
    Q = truth(split); R, A = load(d), load(ref); out = {}
    for qid, q in Q.items():
        if qid not in A:
            continue
        s = final_albums(R.get(qid), q) or []
        a = final_albums(A.get(qid), q) or []
        if q["cat"] == "g":
            S = list(s) + [[]] * max(0, 2 - len(s)); T = list(a) + [[]] * max(0, 2 - len(a))
            best = None
            for order in ([0, 1], [1, 0]):
                rows = [prf(S[order[k]], T[k]) for k in range(2)]
                m = np.mean([x[2] for x in rows])
                if best is None or m > best[0]:
                    best = (m, sum(x[3] for x in rows), sum(len(set(S[order[k]])) for k in range(2)), sum(len(set(T[k])) for k in range(2)))
            out[qid] = (q["cat"], best[1], best[2], best[3])
        else:
            got = set().union(*map(set, s)) if s else set()
            gt = set().union(*map(set, a)) if a else set()
            out[qid] = (q["cat"], len(got & gt), len(got), len(gt))
    return out


def boot(x, seed=0, n=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi


def boot_ratio(num, den, seed=0, n=2000):
    """pooled ratio sum(num)/sum(den) with a request-level (cluster) bootstrap."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(num), len(num))
        bs.append(num[i].sum() / max(den[i].sum(), 1e-9))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return num.sum() / max(den.sum(), 1e-9), lo, hi


def fmt(t, sign=True):
    return (f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]" if sign else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]")
