"""Honest completeness (recall) estimation for a search result.

Mechanism (the "elusion test" idea from legal e-discovery):
  - The search returns a set R. Everything else is the unreturned pool U.
  - Recall = true matches in R / (true matches in R + true matches hiding in U).
  - We cannot look at all of U, so we split U into strata by the cheap model's score (items that scored just
    below the cut are far more likely to be misses than items that scored at the bottom), draw a random
    sample from each stratum, have the judge label the samples, and scale each stratum's hit rate up to its size.
  - Uncertainty: each stratum's hit rate gets a Beta posterior (Jeffreys prior). Monte Carlo draws of
    (precision in R, hit rate in every stratum) give a distribution over recall -> interval + one-sided lower bound.
  - Optional judge-error correction: if we know the judge's sensitivity/specificity (from a sample a human/Claude
    labeled), a judged hit rate q maps to a true rate p = (q + spec - 1) / (sens + spec - 1) (Rogan-Gladen).

Coverage of these intervals is MEASURED on the ground-truth test library (eval/), not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stratum:
    name: str
    size: int                      # N_h: items in this stratum of the unreturned pool
    sampled_ids: list = field(default_factory=list)
    labels: list = field(default_factory=list)  # judge labels (bool) for sampled_ids


def make_strata(scores_unreturned: np.ndarray, ids_unreturned: np.ndarray, edges: list[float]):
    """Split the unreturned pool by score. `edges` descending, e.g. [0.3, 0.2, 0.1, -inf]."""
    strata, upper = [], np.inf
    for e in edges:
        m = (scores_unreturned < upper) & (scores_unreturned >= e)
        strata.append((f"[{e:.3g},{upper:.3g})", ids_unreturned[m]))
        upper = e
    return strata


def allocate(sizes: list[int], budget: int, decay: float = 0.6, min_per: int = 20) -> list[int]:
    """Split a labeling budget across strata (ordered from most to least suspicious).

    Higher-score strata get geometrically more samples per item (misses concentrate there), but every stratum
    gets at least `min_per`, because the huge bottom stratum is where an unsampled miss count could hide.
    """
    k = len(sizes)
    w = np.array([decay ** i for i in range(k)], float) * np.sqrt(np.maximum(sizes, 1))
    alloc = np.floor(budget * w / w.sum()).astype(int)
    alloc = np.maximum(alloc, min_per)
    return [int(min(a, s)) for a, s in zip(alloc, sizes)]


def sample_strata(strata, alloc, rng: np.random.Generator):
    out = []
    for (name, ids), n in zip(strata, alloc):
        pick = rng.choice(ids, size=min(n, len(ids)), replace=False) if len(ids) else np.array([])
        out.append(Stratum(name=name, size=len(ids), sampled_ids=list(pick)))
    return out


def _corrected(p, sens, spec):
    if sens is None or spec is None:
        return p
    denom = sens + spec - 1.0
    if denom <= 0.05:
        return p
    return np.clip((p + spec - 1.0) / denom, 0.0, 1.0)


def estimate_recall(n_returned: int, returned_judged_pos: int, returned_judged_n: int, strata: list[Stratum],
                    conf: float = 0.95, draws: int = 20000, sens: float | None = None, spec: float | None = None,
                    seed: int = 0) -> dict:
    """Monte Carlo over Beta posteriors. Returns point estimate, central interval, one-sided lower bound."""
    rng = np.random.default_rng(seed)
    a = 0.5  # Jeffreys prior Beta(0.5, 0.5)
    prec = rng.beta(returned_judged_pos + a, returned_judged_n - returned_judged_pos + a, draws)
    tp = _corrected(prec, sens, spec) * n_returned
    missed = np.zeros(draws)
    detail = []
    for s in strata:
        y = int(sum(bool(x) for x in s.labels)); n = len(s.labels)
        if s.size == 0:
            continue
        if n == 0:  # unsampled stratum: anything is possible -> uniform prior, keeps the interval honest (wide)
            p = rng.uniform(0, 1, draws)
        else:
            p = rng.beta(y + a, n - y + a, draws)
        p = _corrected(p, sens, spec)
        missed += p * s.size
        detail.append(dict(stratum=s.name, size=s.size, sampled=n, judged_pos=y,
                           est_missed=float(s.size * (y + a) / (n + 2 * a)) if n else None))
    rec = tp / np.maximum(tp + missed, 1e-9)
    lo, hi = np.quantile(rec, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    tp_hat = n_returned * (returned_judged_pos / max(returned_judged_n, 1))
    miss_hat = sum(d["est_missed"] or 0 for d in detail)
    return dict(
        recall_point=float(tp_hat / max(tp_hat + miss_hat, 1e-9)),
        recall_median=float(np.median(rec)),
        recall_ci=(float(lo), float(hi)),
        recall_lower_bound=float(np.quantile(rec, 1 - conf)),  # one-sided: "recall >= this with conf"
        est_true_in_returned=float(tp_hat),
        est_missed=float(miss_hat),
        missed_ci=tuple(float(x) for x in np.quantile(missed, [(1 - conf) / 2, 1 - (1 - conf) / 2])),
        judged_total=int(returned_judged_n + sum(len(s.labels) for s in strata)),
        strata=detail,
        conf=conf,
    )


def describe(est: dict, n_scanned: int) -> str:
    lo, hi = est["recall_ci"]
    return (f"Scanned {n_scanned:,} items with the fast model; the slower judge looked at {est['judged_total']:,}. "
            f"Estimated completeness: {est['recall_point']:.0%} ({est['conf']:.0%} interval {lo:.0%}-{hi:.0%}). "
            f"Estimated matches still missing: {est['est_missed']:.0f} "
            f"(interval {est['missed_ci'][0]:.0f}-{est['missed_ci'][1]:.0f}).")
