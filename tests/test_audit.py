"""Certificate validity on simulated libraries with known answers (judge = truth here; judge error tested separately)."""
import numpy as np

from findpics import audit as A


def simulate(N=150_000, n_true=2_000, head=4_000, tail_budget=3_000, sep=2.5, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.zeros(N, bool); y[rng.choice(N, n_true, replace=False)] = True
    score = rng.normal(0, 1, N) + sep * y
    order = np.argsort(-score)
    H, T = order[:head], order[head:]
    found_head = int(y[H].sum())
    samp = rng.choice(T, size=min(tail_budget, len(T)), replace=False)
    labels = list(y[samp])
    c = A.certify(found_head + int(y[samp].sum()), len(T), labels, alpha)
    found_total = found_head + int(y[samp].sum())
    true_recall = found_total / n_true
    return true_recall, c


def test_lower_bound_holds_at_least_95pct():
    R, ok = 200, 0
    for s in range(R):
        tr, c = simulate(seed=s)
        ok += c["recall_lower"] <= tr + 1e-12
    assert ok / R >= 0.94, ok / R


def test_point_estimate_close():
    tr, c = simulate(seed=7)
    assert abs(c["recall_point"] - tr) < 0.05, (tr, c)


def test_tail_sample_size_cost_law():
    # 145k tail, certify <=100 misses at 95% -> ~4,344 calls
    n = A.tail_sample_size(145_000, 100, 0.05)
    assert 4300 <= n <= 4400, n


def test_cp_upper_rule_of_three():
    assert abs(A.cp_upper(0, 1000, 0.05) - 0.002991) < 1e-4
