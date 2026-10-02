"""Does the recall interval contain the true recall ~95% of the time? Simulated libraries with known answers."""
import numpy as np

from findpics import audit as A


def simulate(N=150_000, n_true=2_000, true_recall=0.9, budget=400, seed=0, judge_fn=None):
    rng = np.random.default_rng(seed)
    y = np.zeros(N, bool); y[rng.choice(N, n_true, replace=False)] = True
    # cheap score: positives higher on average, overlapping
    score = rng.normal(0, 1, N) + 2.5 * y
    # returned set: the top items, sized to hit roughly the target recall, plus a few false positives
    cut = np.quantile(score[y], 1 - true_recall)
    returned = score >= cut
    tp = int((returned & y).sum()); actual = tp / n_true
    pool = np.where(~returned)[0]
    ps = score[pool]
    edges = sorted(set(np.quantile(ps, [0.99, 0.95, 0.8, 0.5]).tolist()), reverse=True) + [-np.inf]
    strata = A.make_strata(ps, pool, edges)
    alloc = A.allocate([len(i) for _, i in strata], budget)
    S = A.sample_strata(strata, alloc, rng)
    jf = judge_fn or (lambda idx: y[idx])
    for s in S:
        s.labels = [bool(jf(int(i))) for i in s.sampled_ids]
    n_ret = int(returned.sum())
    # precision in returned set judged on all returned items (we judge every candidate in the product)
    est = A.estimate_recall(n_ret, tp, n_ret, S, seed=seed)
    return actual, est


def test_point_estimate_reasonable():
    actual, est = simulate(seed=1)
    assert abs(est["recall_point"] - actual) < 0.1, (actual, est["recall_point"])


def test_interval_coverage():
    hits, widths = 0, []
    R = 60
    for seed in range(R):
        actual, est = simulate(seed=seed, true_recall=0.85)
        lo, hi = est["recall_ci"]
        hits += lo <= actual <= hi
        widths.append(hi - lo)
    cov = hits / R
    print(f"coverage {hits}/{R} = {cov:.2f}, mean width {np.mean(widths):.3f}")
    assert cov >= 0.88, cov


def test_small_budget_is_wide_not_wrong():
    actual, est = simulate(seed=3, budget=60)
    lo, hi = est["recall_ci"]
    assert lo <= actual <= hi
