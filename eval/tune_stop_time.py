"""Fast mode re-tuned against WALL-CLOCK time on the phone (RESULTS 43). No new judge calls.

Why: RESULTS 41 tuned fast mode's stop rule assuming ~1 judge call per second. On Reza's iPhone 18 Pro the Qwen3-VL-4B
judge runs ~0.67 photos/s sustained (MAC 10-10 13:56: ~1/s for the first 1.5 min, falling to ~0.52/s by 14 min), so
round 1 (head up to 2,000 photos, THEN a 150-photo random tail sample) can take ~45 min before the completeness bound
appears. Here every rule is replayed as the SEQUENCE of judge calls it makes, so recall can be read at 1/3/5/10/20 min.

Rules
  old       FindPicsCore.streamRounds as shipped (RESULTS 41): head 150, +50 while >= 3% of the last 100 head photos are
            yes, cap 2,000, then 150 random photos of the rest; round 2 (head x2, 300 random) by itself if >= 3 random hits.
  timed     the new round structure (StreamParams.roundCalls > 0; Streaming.swift streamTimed), mirrored exactly here:
            * a uniform permutation `perm` of all positions is fixed before any label;
            * the head is judged in rank order; after every `inter` head photos one random photo is judged: the next
              item of perm not yet judged and not inside the head (inter 0 = no interleaving);
            * round k ends when the calls so far reach roundCalls * 2^(k-1) (a time budget: the person gets a bound
              every doubling of time), or earlier when fast mode's head rule is satisfied (window rule as RESULTS 41:
              < rate yes in the last `window` head photos, checked every `chunk` head photos from `head` on; or
              head >= head_max). At a round's end the random sample of the rest is topped up to >= `tail` photos;
            * certificate: tail = positions >= head boundary b; sample = the perm items drawn so far that are >= b.
              That set is the first m items of perm restricted to [b, n): drawn items are only ever skipped when they
              were inside the head at draw time (< b), and b, the draw times and the budget never depend on a tail
              label, so given m it is a uniform m-subset of the tail (exchangeable under any permutation of [b, n)).
              alpha: round 1 alpha/2, rounds 2..K alpha/2/(K-1); K = max_rounds(n) fixed in advance, round K judges all;
            * rounds that end on time are topped up to `topup` random photos (0 = only what interleaving drew);
            * fast mode continues by itself while the head rule is not yet satisfied (round ended on time), and for
              `extra` round(s) after it is satisfied if that round's random sample found >= `hits` matches (= RESULTS
              41's auto round 2). Rounds after the satisfied one double the calls so far, all on the head (no window
              rule) plus interleaved draws.

  python eval/tune_stop_time.py sweep  [--set=audit4|ev16]   judge-relative found vs time, all candidates
  python eval/tune_stop_time.py eye RULE...                  eye-label recall vs time (RESULTS 34 truth)
  python eval/tune_stop_time.py over                         overclaim replay (every round of every seed, both sets)
  python eval/tune_stop_time.py fixture                      stream.json (Swift fixture) golden call sequences
  python eval/tune_stop_time.py heat RULE... [--tag=]        RESULTS 44: every rule on the MEASURED phone clocks (m4 = shipped
                                                             checkpoint, m8 = 8-bit vision tower, fit4 = m4's exponential
                                                             fit), recall at 3/10/20/40 min and at the 68-min wall, done time
  python eval/tune_stop_time.py eye RULE... --time=m4 --minutes=3:10:20:40:999 --tag=_x   (999 = everything before the wall)
  Rule knobs added for RESULTS 44: fast_stop (fast mode ends after the first round reaching that many calls) and
  stop_topup (that round's random sample is topped up to this many) = StreamParams.fastStopCalls / fastStopTopUp.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
import tune_stop_rule as T  # noqa: E402

OUT = Path("eval/stop_time")
RATE = 0.67                                    # photos/s sustained on the iPhone 18 Pro (MAC 10-10 13:56)
MINUTES = (1, 3, 5, 10, 20)
# measured cumulative judged photos vs seconds after the judge loaded (MAC 10-10 13:42:47 -> 13:56:19)
THERMAL = np.array([[0, 0], [91, 92], [330, 287], [585, 443], [812, 561]], float)
ALPHA = 0.05


# ---- the MEASURED phone clock (RESULTS 44; MAC 10-10 15:50 and 16:25 in JOURNAL.md, MAC_INBOX M37): photos judged per
# minute in 5-minute buckets of continuous judging on Reza's iPhone 18 Pro. (The Mac labelled them "2-minute"; 5 is
# right: the 14 buckets of the 68-min run x 5 min = 1,175 photos vs 1,181 logged, and they match the 0-5 / 5-10 / 10-15
# / 15-30 / 30-65 / 65-70 min rates of the 15:50 entry.) After ~68 min iOS halted charging and locked the phone: the
# app is suspended, nothing more is judged ("wall").
BUCKET_MIN = 5.0
MEASURED = {
    "m4": ([35, 26, 15, 12, 12, 14, 19, 18, 16, 16, 18, 17, 14, 3], 68.0),  # shipped checkpoint (bf16 vision tower)
    "m8": ([30, 36, 33, 24, 24, 26, 24, 22, 19, 21, 21, 16, 6], 65.0),      # 8-bit vision tower (log ends at 65 min)
}


def fit_decay(model):
    """r(t) = rs + (r0 - rs) exp(-t / tau) photos/min, least squares on the bucket MEANS (the integral over each
    bucket), the last (halting) bucket left out."""
    from scipy.optimize import curve_fit
    rates, _ = MEASURED[model]
    y = np.array(rates[:-1], float); a = np.arange(len(y)) * BUCKET_MIN; b = a + BUCKET_MIN

    def mean_rate(_, r0, rs, tau):
        return rs + (r0 - rs) * tau * (np.exp(-a / tau) - np.exp(-b / tau)) / BUCKET_MIN
    (r0, rs, tau), _ = curve_fit(mean_rate, a, y, p0=(40, 15, 8), bounds=([0, 0, 0.5], [200, 100, 100]))
    return r0, rs, tau


_FIT = {}


def cum_calls(minutes, model):
    """Photos judged after `minutes` of continuous judging (float), capped at the wall."""
    if model in MEASURED:
        rates, wall = MEASURED[model]
        t = min(minutes, wall); c = 0.0
        for i, r in enumerate(rates):
            lo = i * BUCKET_MIN
            if t <= lo:
                break
            c += r * (min(t, lo + BUCKET_MIN) - lo)
        return c
    if model.startswith("fit"):
        base = "m" + model[3:]
        if base not in _FIT:
            _FIT[base] = fit_decay(base)
        r0, rs, tau = _FIT[base]
        t = min(minutes, MEASURED[base][1])
        return rs * t + (r0 - rs) * tau * (1 - np.exp(-t / tau))
    raise ValueError(model)


def minutes_of(calls, model):
    """Inverse of cum_calls: minutes until `calls` photos are judged; inf when the wall comes first."""
    if model == "const":
        return calls / RATE / 60
    wall = MEASURED["m" + model[-1]][1]
    if calls > cum_calls(wall, model) + 1e-9:
        return np.inf
    lo, hi = 0.0, wall
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cum_calls(mid, model) < calls else (lo, mid)
    return hi


def calls_at(sec, model="const"):
    if model == "const":
        return int(sec * RATE)
    if model in MEASURED or model.startswith("fit"):
        return int(cum_calls(sec / 60, model))
    t, c = THERMAL[:, 0], THERMAL[:, 1]
    if sec <= t[-1]:
        return int(np.interp(sec, t, c))
    return int(c[-1] + (sec - t[-1]) * 0.52)          # last measured segment: 118 photos in 227 s


# ------------------------------------------------------------------------------------------------------------ old rule
def sim_old(y, r, seed):
    """streamRounds (Streaming.swift, roundCalls 0) as a call sequence. Returns calls (positions in judge order) and
    rounds [(calls_so_far, cert, k, last)]. Same head rule and RNG use as T.simulate (numpy RNG, not SplitMix)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    seen, order = np.zeros(n, bool), []

    def judge(pos):
        for q in pos:
            if not seen[q]:
                seen[q] = True; order.append(int(q))

    n_head = min(r["head"], n); judge(range(n_head))
    while n_head < min(r["head_max"], n):
        last = y[max(0, n_head - r["window"]):n_head]
        if last.sum() / max(len(last), 1) < r["rate"]:
            break
        n_head = min(n_head + r["chunk"], n); judge(range(n_head))
    h, n_later = n_head, 0
    while True:
        h = min(n, 2 * h)
        if h < n:
            n_later += 1
        else:
            break
    perm = rng.permutation(n)
    rounds, k = [], 0
    while True:
        k += 1
        tail = n - n_head
        n_t = min(tail, r["tail"] * 2 ** (k - 1))
        ts = rng.permutation(np.arange(n_head, n))[:n_t] if k == 1 else perm[perm >= n_head][:n_t]
        judge(range(n_head)); judge(ts)
        a_k = ALPHA / 2 if k == 1 else ALPHA / 2 / max(n_later, 1)
        cert = T.certify(int((seen & y).sum()), tail, y[ts], a_k)
        last = n_head >= n
        rounds.append((len(order), cert, k, last))
        more = (r["more"] is not None and k < r["max_rounds"] and T.keep_going(cert, r["more"]))
        if last or not more:
            break
        n_head = min(n, 2 * n_head)
    return np.array(order), rounds


# ---------------------------------------------------------------------------------------------------------- timed rule
TIMED = dict(head=150, chunk=50, window=100, rate=0.03, head_max=2000, tail=150, round_calls=150, inter=2,
             hits=3, extra=1, fast_cap=None, topup=0, a1=0.5, fast_stop=0, stop_topup=0)


def max_rounds(n, r):
    """Most rounds a timed run can have, fixed by n before any label: every round but the satisfied one ends with
    calls >= its budget, so round k >= 2 ends with calls >= m0 * 2^(k-2), m0 = min(round_calls, head); round K judges
    everything (enforced: its budget is n)."""
    m0 = max(1, min(r["round_calls"], r["head"]))
    return 2 + int(np.ceil(np.log2(max(n / m0, 1))))


def alpha_k(k, K, a1=0.5, alpha=ALPHA):
    """Round 1: a1 * alpha; rounds 2..K: (1 - a1) * alpha / (K-1) each. Every bound shown holds simultaneously
    (union bound)."""
    return a1 * alpha if k == 1 else (1 - a1) * alpha / max(K - 1, 1)


def sim_timed(y, r, seed, exhaustive=False, perm=None):
    """Mirror of Streaming.swift streamTimed. exhaustive: always continue (every round until all judged)."""
    n = len(y)
    if perm is None:
        perm = np.random.default_rng(seed).permutation(n)
    seen, order = np.zeros(n, bool), []
    K = max_rounds(n, r)
    b = pp = since = 0
    sat_round = 0                         # round in which the head rule was satisfied (0 = not yet)

    def judge(q):
        if not seen[q]:
            seen[q] = True; order.append(int(q))

    def draw():
        nonlocal pp
        while pp < n and (perm[pp] < b or seen[perm[pp]]):
            pp += 1
        if pp >= n:
            return False
        judge(perm[pp]); pp += 1
        return True

    def sample():
        d = perm[:pp]
        return d[d >= b]

    def head_rule():                      # checked after each head photo
        if b >= min(r["head_max"], n):
            return True
        if b < r["head"] or (b - r["head"]) % r["chunk"]:
            return False
        last = y[max(0, b - r["window"]):b]
        return last.sum() / max(len(last), 1) < r["rate"]

    rounds, k = [], 0
    while True:
        k += 1
        # time budget: doubles every round (rounds after the head rule was satisfied: double the calls so far)
        budget = min(n, r["round_calls"] * 2 ** (k - 1) if sat_round == 0 else 2 * len(order))
        if k >= K:
            budget = n
        while len(order) < budget and b < n:
            if r["inter"] and since >= r["inter"]:
                since = 0
                if draw():
                    continue
            judge(b); b += 1; since += 1
            if sat_round == 0 and head_rule():
                sat_round = k
                break
        if len(order) >= n:
            b = n
        want = r["tail"] if sat_round else r["topup"]       # top the random sample up: always once fast mode is done
        if r.get("fast_stop") and r.get("stop_topup") and len(order) >= r["fast_stop"]:
            want = max(want, r["stop_topup"])                # the round that spends fast mode's budget (RESULTS 44)
        while b < n and len(sample()) < want and draw():
            pass
        s = sample()
        cert = T.certify(int((seen & y).sum()), n - b, y[s], alpha_k(k, K, r["a1"]))
        last = b >= n
        rounds.append((len(order), cert, k, last))
        if last:
            break
        if exhaustive:
            continue
        if r.get("fast_stop") and len(order) >= r["fast_stop"]:     # fast mode's call budget (RESULTS 44)
            break
        if sat_round == 0:
            if r["fast_cap"] and len(order) >= r["fast_cap"]:
                break
            continue
        if k < sat_round + r["extra"] and cert["tail_hits"] >= r["hits"]:
            continue
        break
    return np.array(order), rounds


def rule(**kw):
    d = dict(TIMED); d.update(kw); return d


OLD = dict(T.CURRENT, window=100, head_max=2000, more=("hits", 3), max_rounds=2)


def candidates():
    R = {"old (RESULTS 41, shipped)": ("old", OLD)}
    for rc in (90, 100, 120, 150):
        for inter in (0, 4):
            for topup in (30, 40, 50, 75):
                for head in (100, 150):
                    R[f"timed round_calls={rc} inter={inter} topup={topup} head={head}"] = (
                        "timed", rule(round_calls=rc, inter=inter, topup=topup, head=head))
    return R


def simulate(kind, y, r, seed):
    return sim_old(y, r, seed) if kind == "old" else sim_timed(y, r, seed)


def curves(cases, kind, r, seeds=T.SEEDS, model="const"):
    """Per case and seed: found at each minute mark, first-match time, first-bound time, done time, final found,
    bound shown at first and last round."""
    rows = []
    marks = [calls_at(60 * m, model) for m in MINUTES]
    for c in cases:
        y = c["y"]
        for s in range(seeds):
            order, rounds = simulate(kind, y, r, s)
            hit = y[order]
            cum = np.cumsum(hit)
            row = dict(key=c["key"], seed=s, k_last=rounds[-1][2], alpha_last=rounds[-1][1].get("alpha", np.nan), kept=int(y.sum()), n=len(y), calls=len(order),
                       first_match=int(np.argmax(hit)) + 1 if hit.any() else np.nan,
                       first_bound=rounds[0][0], bound1=rounds[0][1]["missed_upper"],
                       bound_last=rounds[-1][1]["missed_upper"], rounds=len(rounds), found=int(cum[-1]))
            for m, cm in zip(MINUTES, marks):
                row[f"f{m}"] = int(cum[min(cm, len(cum)) - 1]) if cm > 0 else 0
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(D, label):
    k = D.kept.sum() / T.SEEDS
    cells = [f"{D[f'f{m}'].sum() / T.SEEDS / k:.3f}" for m in MINUTES]
    return (f"{label:44s} " + " ".join(cells) + f" | final {D.found.sum() / T.SEEDS / k:.3f}"
            f" calls {D.calls.sum() / T.SEEDS:7.0f}  bound@ med {D.first_bound.median() / RATE / 60:5.1f} min"
            f" (max {D.first_bound.max() / RATE / 60:5.1f})  done med {D.calls.median() / RATE / 60:5.1f} min")


def sweep(which, rules=None, tag=""):
    cases = T.audit4() if which == "audit4" else T.ev16()
    lines = [f"{which}: judge-relative found / judge-yes at {MINUTES} min (0.67 photos/s), 20 seeds"]
    allr = []
    todo = candidates() if not rules else {s: parse(s) for s in rules}
    for label, (kind, r) in todo.items():
        D = curves(cases, kind, r); D.insert(0, "rule", label); allr.append(D)
        lines.append(summarize(D, label)); print(lines[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(allr).to_csv(OUT / f"sweep_{which}{tag}.tsv", sep="\t", index=False)
    (OUT / f"sweep_{which}{tag}.txt").write_text("\n".join(lines))


def table(which, tag=""):
    """Per rule: found / judge-yes at each minute mark, pooled (all searches' matches together) and per-search mean
    (each search counts once), final, judge calls, when the first bound appears, and when fast mode is done."""
    D = pd.read_csv(OUT / f"sweep_{which}{tag}.tsv", sep="\t")
    R = RATE * 60
    rows = []
    for rule_, g in D.groupby("rule", sort=False):
        r = dict(rule=rule_)
        for m in MINUTES:
            r[f"pool{m}"] = g[f"f{m}"].sum() / g.kept.sum()
            r[f"mean{m}"] = (g[f"f{m}"] / g.kept.clip(lower=1)).groupby(g.key).mean().mean()
        r["pool_final"] = g.found.sum() / g.kept.sum()
        r["mean_final"] = (g.found / g.kept.clip(lower=1)).groupby(g.key).mean().mean()
        r["calls_per_search"] = g.calls.mean()
        r["bound_min_med"] = g.first_bound.median() / R; r["bound_min_max"] = g.first_bound.max() / R
        r["done_min_med"] = g.calls.median() / R; r["done_min_p90"] = g.calls.quantile(0.9) / R
        r["first_match_s_med"] = g.first_match.median() / RATE
        r["bound1_share_med"] = (g.bound1 / g.n).median(); r["boundlast_share_med"] = (g.bound_last / g.n).median()
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"table_{which}{tag}.tsv", sep="\t", index=False, float_format="%.4g")
    return S


def parse(s):
    if s.startswith("old"):
        return "old", OLD
    r = dict(TIMED)
    for p in s.split()[1:]:
        k, v = p.split("=")
        r[k] = None if v == "None" else (float(v) if k in ("rate", "a1") else int(v))
    return "timed", r


def eye(rules, model="const"):
    """Eye-label recall (RESULTS 34 truth, score_set: unsure = no match) of the matches on screen at each minute mark,
    mean over 20 seeds, per search and pooled; plus judge calls and bound timing."""
    from eval_recall import QUERIES, _labels_eye, load_scores, old_labels, score_set
    cases = T.audit4()
    df = load_scores(); old = old_labels(); eye_l = _labels_eye()
    marks = [calls_at(60 * m, model) for m in MINUTES]
    out, rows = [], []
    for s in rules:
        kind, r = parse(s)
        per = []
        for sd in range(T.SEEDS):
            sims = {c["key"]: simulate(kind, c["y"], r, sd) for c in cases}
            for m, cm in list(zip(MINUTES, marks)) + [("final", None)]:
                keep = {}
                for c in cases:
                    order = sims[c["key"]][0]
                    o = order if cm is None else order[:cm]
                    keep[c["key"]] = set(c["ids"][o[c["y"][o]]])
                R = score_set(df, old, eye_l, keep, B=10)
                R["minute"] = m; R["seed"] = sd
                per.append(R[["query", "minute", "seed", "recall", "real_returned_est", "real_est"]])
        P = pd.concat(per, ignore_index=True); P["rule"] = s; rows.append(P)
        pool = P.groupby(["minute", "seed"]).apply(lambda d: d.real_returned_est.sum() / d.real_est.sum(),
                                                    include_groups=False).groupby(level=0).mean()
        q = P.groupby(["query", "minute"]).recall.mean().unstack()
        cols = [m for m in list(MINUTES) + ["final"]]
        out.append(f"-- {s} [{model}]")
        out.append("   pooled " + " ".join(f"{m}:{pool[m]:.3f}" for m in cols))
        for qq in QUERIES:
            out.append(f"   {qq:22s} " + " ".join(f"{m}:{q.loc[qq, m]:.3f}" for m in cols))
        print("\n".join(out[-8:]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(OUT / f"eye_{model}{EYE_TAG}.tsv", sep="\t", index=False)
    (OUT / f"eye_{model}{EYE_TAG}.txt").write_text("\n".join(out))


def over(rules, seeds=T.SEEDS):
    """Every round's certificate vs the truth (found / all judge-yes): recallLower must never exceed it.
    Fast mode and exhaustive mode (every round until all judged), both sets."""
    lines = []
    for which in ("audit4", "ev16"):
        cases = T.audit4() if which == "audit4" else T.ev16()
        for s in rules:
            kind, r = parse(s)
            for exh in ((False, True) if kind == "timed" else (False,)):
                nr = no = 0
                for c in cases:
                    y = c["y"]; tot = int(y.sum())
                    if tot == 0:
                        continue
                    for sd in range(seeds):
                        if kind == "timed":
                            order, rounds = sim_timed(y, r, sd, exhaustive=exh)
                        else:
                            order, rounds = sim_old(y, r, sd)
                        for (nc, cert, k, last) in rounds:
                            f = int(y[order[:nc]].sum())
                            assert f == cert["found"]
                            lo = f / (f + cert["missed_upper"]) if f + cert["missed_upper"] > 0 else 0
                            nr += 1; no += lo > f / tot + 1e-9
                lines.append(f"{which:6s} {s:50s} {'exhaustive' if exh else 'fast':10s} rounds {nr:6d} overclaims {no}")
                print(lines[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overclaims.txt").write_text("\n".join(lines))


def overlist(rules, seeds=T.SEEDS):
    """Every overclaim of `over`, with how likely a sample that bad was: P(<= hits | hypergeometric draw of `sampled`
    from a tail holding `tail_yes` matches), next to the round's alpha. A bug shows as P far below alpha."""
    from scipy.stats import hypergeom
    for which in ("audit4", "ev16"):
        cases = T.audit4() if which == "audit4" else T.ev16()
        for s in rules:
            kind, r = parse(s)
            for c in cases:
                y = c["y"]; tot = int(y.sum())
                if tot == 0:
                    continue
                for sd in range(seeds):
                    for exh in (False, True):
                        order, rounds = sim_timed(y, r, sd, exhaustive=exh)
                        for (nc, cert, k, last) in rounds:
                            f = cert["found"]; up = cert["missed_upper"]
                            if f / (f + up) > f / tot + 1e-9:
                                nt = cert["n_tail"]; ty = int(y[len(y) - nt:].sum())
                                p = hypergeom.cdf(cert["tail_hits"], nt, ty, cert["tail_sampled"])
                                print(f"{which} {s} | {c['key']} n={len(y)} seed={sd} {'exh' if exh else 'fast'} round {k} "
                                      f"calls {nc}: tail {nt} holds {ty}, sample {cert['tail_sampled']} hits "
                                      f"{cert['tail_hits']}, bound {up:.1f} vs missing {tot - f}; P(<=hits) {p:.4f}, "
                                      f"alpha {alpha_k(k, max_rounds(len(y), r), r['a1']):.4f}", flush=True)


def uniform(rules, seeds=2000):
    """Is the random sample uniform over the unchecked rest? Per round, over `seeds` permutations: mean sample hits vs
    mean (sample size x the rest's yes rate), and overclaims (exhaustive runs, car and bicycle of audit4)."""
    from collections import defaultdict
    for s in rules:
        kind, r = parse(s)
        for c in T.audit4():
            if "car" not in c["key"] and "bicycle" not in c["key"]:
                continue
            y = c["y"]; acc = defaultdict(lambda: [0, 0.0, 0.0, 0])
            for sd in range(seeds):
                for (nc, cert, k, last) in sim_timed(y, r, sd, exhaustive=True)[1]:
                    if cert["n_tail"] == 0 or cert["tail_sampled"] == 0:
                        continue
                    a = acc[k]; a[0] += 1; a[1] += cert["tail_hits"]
                    a[2] += cert["tail_sampled"] * y[len(y) - cert["n_tail"]:].mean()
                    a[3] += cert["found"] / (cert["found"] + cert["missed_upper"]) > cert["found"] / y.sum() + 1e-9
            print(s, "|", c["key"], " ".join(f"r{k}: n={v[0]} hits {v[1] / v[0]:.3f} vs {v[2] / v[0]:.3f} over {v[3]}"
                                            for k, v in sorted(acc.items())), flush=True)


HEAT_MARKS = (3, 10, 20, 40)
EYE_TAG = ""


def _heat_one(args):
    """One rule on one set: per case and seed, the call sequence once, then every clock read off it."""
    which, s, models, marks = args
    cases = T.audit4() if which == "audit4" else T.ev16()
    kind, r = parse(s)
    rows = []
    for c in cases:
        y = c["y"]
        for sd in range(T.SEEDS):
            order, rounds = simulate(kind, y, r, sd)
            cum = np.cumsum(y[order])
            base = dict(set=which, rule=s, key=c["key"], seed=sd, kept=int(y.sum()), n=len(y), calls=len(order),
                        found=int(cum[-1]) if len(cum) else 0, first_bound_calls=rounds[0][0],
                        rounds=len(rounds), bound_last=rounds[-1][1]["missed_upper"])
            for model in models:
                row = dict(base, clock=model)
                for m in marks:
                    cm = calls_at(60 * m, model)
                    row[f"f{m}"] = int(cum[min(cm, len(cum)) - 1]) if cm > 0 and len(cum) else 0
                cw = calls_at(1e9, model)                                   # everything judged before the wall
                row["f_wall"] = int(cum[min(cw, len(cum)) - 1]) if len(cum) else 0
                row["first_bound_min"] = minutes_of(rounds[0][0], model)
                row["done_min"] = minutes_of(len(order), model)
                # the last bound SHOWN before the wall (inf = none)
                shown = [x for x in rounds if minutes_of(x[0], model) < np.inf]
                row["bound_wall"] = shown[-1][1]["missed_upper"] if shown else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def heat(rules, models=("m4", "m8", "fit4"), marks=HEAT_MARKS, tag="", procs=8):
    """Every rule under the MEASURED phone clock (RESULTS 44): judge-relative found / judge-yes at each minute mark,
    at the wall (68 min; 65 for m8), and when fast mode ends (inf = the wall comes first)."""
    from multiprocessing import Pool
    jobs = [(w, s, models, marks) for w in ("audit4", "ev16") for s in rules]
    with Pool(procs) as pool:
        D = pd.concat(pool.map(_heat_one, jobs), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / f"heat{tag}.tsv", sep="\t", index=False)
    heat_table(D, marks, tag)


def heat_table(D=None, marks=HEAT_MARKS, tag=""):
    if D is None:
        D = pd.read_csv(OUT / f"heat{tag}.tsv", sep="\t")
    lines = []
    for (w, model), G in D.groupby(["set", "clock"], sort=False):
        lines.append(f"== {w}, clock {model}: judge-relative found / judge-yes; pooled | per-search mean; minutes")
        lines.append(f"{'rule':58s} " + " ".join(f"{m:>11d}" for m in marks) + "        wall    unbounded"
                     " | bound@ med  done med   p90  done<wall  calls/search")
        for s, g in G.groupby("rule", sort=False):
            cells = []
            for col in [f"f{m}" for m in marks] + ["f_wall", "found"]:
                pool_ = g[col].sum() / g.kept.sum()
                mean_ = (g[col] / g.kept.clip(lower=1)).groupby(g.key).mean().mean()
                cells.append(f"{pool_:.3f}|{mean_:.3f}")
            dm = g.done_min.replace(np.inf, 1e9)
            fmt = lambda v: "wall" if v >= 1e8 else f"{v:5.1f}"
            lines.append(f"{s:58s} " + " ".join(cells) + f" | {g.first_bound_min.median():5.1f}"
                         f"  {fmt(dm.median()):>8s} {fmt(dm.quantile(0.9)):>5s}  {np.isfinite(g.done_min).mean():.3f}"
                         f"  {g.calls.mean():7.0f}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    (OUT / f"heat{tag}.txt").write_text(txt)


def fixture(rules, out="ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/stream_timed.json", n=3000):
    """Golden runs for the Swift test (StreamingTests.testTimedMatchesPythonReplay): 3 stream.json concepts cut to their
    top `n` positions, the permutation passed in (so this checks the round logic, not the RNG), fast and exhaustive."""
    cases = json.load(open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/stream.json"))
    runs = []
    for s in rules:
        kind, r = parse(s)
        runs.append(dict(rule=r, cases=fixture_cases(cases, r, n)))
    json.dump(dict(runs=runs), open(out, "w"))


def fixture_cases(cases, r, n):
    res = []
    for c in cases:
        if c["name"] not in ("christmas_tree", "guitar", "bicycle"):
            continue
        p = np.round(np.array(c["p"][:n]), 4); y = p >= 0.7
        perm = np.random.default_rng(7).permutation(n)
        row = dict(name=c["name"], p=[round(float(v), 4) for v in p], perm=perm.tolist())
        for mode, exh in (("fast", False), ("exhaustive", True)):
            order, rounds = sim_timed(y, r, 0, exhaustive=exh, perm=perm)
            row[mode] = dict(calls=[x[0] for x in rounds], hits=[x[1]["tail_hits"] for x in rounds],
                             sampled=[x[1]["tail_sampled"] for x in rounds], ntail=[x[1]["n_tail"] for x in rounds],
                             found=[x[1]["found"] for x in rounds],
                             upper=[round(x[1]["missed_upper"], 6) for x in rounds], order=order[:80].tolist())
        res.append(row)
        print(c["name"], {m: row[m]["calls"] for m in ("fast", "exhaustive")})
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    which = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--set=")), "audit4")
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--time=")), "const")
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "")
    mins = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--minutes=")), "")
    if mins:                       # e.g. --minutes=3:10:20:40:999 (999 = everything before the wall; ":" survives sbatch --export)
        MINUTES = tuple(int(x) for x in mins.replace(":", ",").split(","))
    if cmd == "sweep":
        sweep(which, args, tag)
    elif cmd == "eye":
        EYE_TAG = tag
        eye(args, model)
    elif cmd == "over":
        over(args)
    elif cmd == "uniform":
        uniform(args)
    elif cmd == "overlist":
        overlist(args)
    elif cmd == "table":
        pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
        print(table(which, tag).round(3).to_string())
    elif cmd == "fixture":
        fixture(args)
    elif cmd == "heat":
        heat(args, tag=tag)
    elif cmd == "heat_table":
        heat_table(tag=tag)
