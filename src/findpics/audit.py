"""Honest completeness (recall) certificate for a search result.

Design (from research/04; elusion test as in e-discovery, Lewis/Yang/Frieder 2021 style):
  1. The fast models score every in-scope item.
  2. The judge (VLM) looks at EVERY item in the "head": the top-K by fast score (results + the next several
     times as many). Fully judged strata have no sampling error. Judge-yes items in the head become results.
  3. Everything below the head is the "tail". We draw a uniform random sample of the tail, decided BEFORE looking
     at any label, and have the judge label it.
  4. Exact one-sided Clopper-Pearson upper bound on the tail hit rate -> upper bound on matches hiding in the tail:
        missed_upper = N_tail * p_upper(hits, n, alpha)
     recall_lower = found / (found + missed_upper)          (valid at level 1-alpha, relative to the judge)
     recall_point = found / (found + N_tail * hits / n)
  5. Cost law: to certify at most m misses in a tail of N_tail with zero hits, n ~ N_tail * ln(1/alpha) / m.
     e.g. N_tail=145k, m=100, alpha=0.05 -> n ~ 4,350 judge calls.

"Relative to the judge": if the VLM says no to a true match, the certificate cannot see it. So we also report a
human (or Claude) check of a random sample of judge decisions, and say explicitly which labels the number is relative to.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as _beta


def cp_upper(hits: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial proportion."""
    if n == 0:
        return 1.0
    if hits >= n:
        return 1.0
    return float(_beta.ppf(1 - alpha, hits + 1, n - hits))


def tail_sample_size(n_tail: int, max_missed: float, alpha: float = 0.05) -> int:
    """Judge calls needed so that zero hits certifies <= max_missed misses in the tail."""
    if n_tail <= 0:
        return 0
    return int(min(n_tail, math.ceil(n_tail * math.log(1 / alpha) / max(max_missed, 1e-9))))


def certify(found: int, n_tail: int, tail_labels: list[bool], alpha: float = 0.05) -> dict:
    n = len(tail_labels)
    hits = int(sum(bool(x) for x in tail_labels))
    if n_tail == 0:
        return dict(found=found, n_tail=0, tail_sampled=0, tail_hits=0, missed_point=0.0, missed_upper=0.0,
                    recall_point=1.0 if found else float("nan"), recall_lower=1.0 if found else float("nan"), alpha=alpha)
    pu = cp_upper(hits, n, alpha)
    missed_up = n_tail * pu
    missed_pt = n_tail * hits / n if n else float("nan")
    return dict(found=found, n_tail=n_tail, tail_sampled=n, tail_hits=hits,
                missed_point=missed_pt, missed_upper=missed_up,
                recall_point=found / (found + missed_pt) if found + missed_pt > 0 else float("nan"),
                recall_lower=found / (found + missed_up) if found + missed_up > 0 else 0.0,
                alpha=alpha)


def describe(c: dict, n_scanned: int, n_head_judged: int, n_human: int = 0) -> str:
    conf = 1 - c["alpha"]
    if c["n_tail"] == 0:
        return (f"The judge looked at every one of the {n_scanned:,} in-scope items, so nothing went unchecked: completeness is "
                f"100% relative to the AI judge's yes/no answers (not to a person's).")
    s = (f"Scored all {n_scanned:,} in-scope items with the fast models; the judge looked at the top {n_head_judged:,} "
         f"plus a random {c['tail_sampled']:,} of the remaining {c['n_tail']:,}. ")
    if c["tail_sampled"]:
        s += (f"The random check found {c['tail_hits']} more match(es). "
              f"Completeness: about {c['recall_point']:.0%}; at least {c['recall_lower']:.0%} with {conf:.0%} confidence "
              f"(at most ~{c['missed_upper']:.0f} matches could still be hiding). ")
    s += "These numbers are relative to the AI judge's yes/no answers"
    s += f"; {n_human} of its answers were checked by a person." if n_human else "; no human has checked the judge yet."
    return s
