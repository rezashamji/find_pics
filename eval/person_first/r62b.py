"""RESULTS 62b: a COVERAGE-FIRST walk for LOOK person searches (pre-registered in JOURNAL 10-14 ~21:10, f47db8d).

Mac M72(a): the owner's face set = 41,307 photos in 5,764 events (3 h gaps), median 3, largest 517, 1,689 singletons.
A pair's 480-call first answer judges 240 photos; the shipped random walk (cap 3) reaches few events. Walks:
  size   events by size descending (ties: hash of the event's first photo); pass 1 = photo #1 of every event, pass 2 =
         photos #2-#3 of every event in the same order, then the rest in global hash order
  walk3  shipped (person_first.walk_order, cap 3)
  walk1  random, cap 1
Within an event photos are in walk_key order (fmix64 of fnv1a64("seed|id")) for all three.

  python eval/person_first/r62b.py public    -> eval/person_first/r62b_public.txt  (S1 parametric answers, sizes, S2)
  python eval/person_first/r62b.py private   -> data/private/audits/r62b_private.txt (S1 bootstrap from his events, S3)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval/person_first"))
from findpics import person_first as PF  # noqa: E402

OUT = ROOT / "eval/person_first"
N_PHOTOS, N_EVENTS, N_SINGLE, MAX_EV, MEDIAN = 41307, 5764, 1689, 517, 3
BUDGETS = (480, 1000, 2000, 5000)
S3_BUDGETS = (120, 240, 480)
WALKS = ("size", "walk3", "walk1")
DRAWS, SEEDS = 5, 5


# ------------------------------------------------------------------------------------------------------------ walks
def size_walk(ids, event_of, seed=0):
    ids = [str(i) for i in ids]
    key = {i: PF.walk_key(seed, i) for i in ids}
    per = {}
    for k, i in enumerate(ids):
        per.setdefault(PF._event(event_of, i), []).append(k)
    for e in per:
        per[e].sort(key=lambda k: (key[ids[k]], ids[k]))
    evs = sorted(per, key=lambda e: (-len(per[e]), key[ids[per[e][0]]], ids[per[e][0]]))
    first = [per[e][0] for e in evs]
    second = [k for e in evs for k in per[e][1:3]]
    taken = set(first) | set(second)
    rest = [k for k in sorted(range(len(ids)), key=lambda k: (key[ids[k]], ids[k])) if k not in taken]
    return first + second + rest


def order_of(walk, ids, event_of, seed):
    if walk == "size":
        return size_walk(ids, event_of, seed)
    return PF.walk_order(ids, event_of, seed, cap=3 if walk == "walk3" else 1)


def coverage_curve(order, ids, event_of):
    """Covered share of photos after m walked photos, m = 0..n (label-free)."""
    size, first_at = {}, {}
    for i in ids:
        e = PF._event(event_of, i); size[e] = size.get(e, 0) + 1
    for m, k in enumerate(order):
        e = PF._event(event_of, ids[k])
        first_at.setdefault(e, m)
    inc = np.zeros(len(ids) + 1)
    for e, m in first_at.items():
        inc[m + 1] += size[e]
    return np.cumsum(inc) / len(ids)


def calls_to(cov, share, per_photo=2):
    k = int(np.searchsorted(cov, share - 1e-12))
    return k * per_photo


# ------------------------------------------------------------------------------------------------------------ his sizes
def lognormal_fits():
    """(mu, sigma) of s = 2 + floor(Y), Y lognormal, for the non-singleton events: mean 39,618 / 4,075 and the median-3
    constraint (n1 + n2 <= 2,881, n1 + n2 + n3 >= 2,883). Returns best (mean closest), lightest and heaviest tail."""
    from scipy.stats import lognorm
    rows = []
    k = np.arange(0, MAX_EV - 1)
    target = (N_PHOTOS - N_SINGLE) / (N_EVENTS - N_SINGLE)
    for mu in np.linspace(-1, 3, 161):
        for sg in np.linspace(0.3, 3, 109):
            Y = lognorm(sg, scale=np.exp(mu))
            pk = Y.cdf(k + 1) - Y.cdf(k); pk[-1] += 1 - Y.cdf(k[-1] + 1); pk /= pk.sum()
            m = float(((k + 2) * pk).sum()); n_ns = N_EVENTS - N_SINGLE
            n2, n3 = n_ns * pk[0], n_ns * pk[1]
            if N_SINGLE + n2 <= 2881 and N_SINGLE + n2 + n3 >= 2883:
                rows.append((abs(m - target), mu, sg, m))
    R = pd.DataFrame(rows, columns=["err", "mu", "sg", "mean"])
    near = R[R.err < 0.1]
    best = R.sort_values("err").iloc[0]
    return {"best": (best.mu, best.sg), "light": tuple(near.sort_values("sg").iloc[0][["mu", "sg"]]),
            "heavy": tuple(near.sort_values("sg").iloc[-1][["mu", "sg"]])}


def his_sizes(mu, sg):
    from scipy.stats import lognorm
    n_ns = N_EVENTS - N_SINGLE
    q = (np.arange(n_ns) + 0.5) / n_ns
    s = np.minimum(2 + np.floor(lognorm(sg, scale=np.exp(mu)).ppf(q)).astype(int), MAX_EV)
    s[-1] = MAX_EV
    need = (N_PHOTOS - N_SINGLE) - s.sum()          # adjust the total on the large events, never past MAX_EV
    j = n_ns - 2
    while need != 0 and j > 0:
        step = 1 if need > 0 else -1
        if 2 <= s[j] + step <= MAX_EV - 1:
            s[j] += step; need -= step
        j -= 1
        if j <= n_ns // 2:
            j = n_ns - 2
    sizes = np.r_[np.ones(N_SINGLE, int), np.sort(s)]
    assert sizes.sum() == N_PHOTOS and len(sizes) == N_EVENTS and sizes.max() == MAX_EV and np.median(sizes) == MEDIAN
    return sizes


def library(sizes, rng):
    perm = rng.permutation(len(sizes))           # event ids not ordered by size
    ids, ev = [], {}
    for e, s in zip(perm, sizes):
        for j in range(s):
            i = f"e{e:05d}p{j:03d}"; ids.append(i); ev[i] = f"E{e}"
    return ids, ev


# ------------------------------------------------------------------------------------------------------------ answers
def parametric_answers(ids, ev, rng):
    sig = lambda z: 1 / (1 + np.exp(-z))
    evs = sorted(set(ev.values()))
    era = {e: ("H" if rng.random() < 0.5 else "F") for e in evs}
    xe = {e: (1.5 if era[e] == "H" else -1.5) + rng.normal(0, 0.8) for e in evs}
    pa, pb, lab = {}, {}, {}
    for i in ids:
        e = ev[i]
        x = rng.normal(0, 2) if rng.random() < 0.1 else xe[e] + rng.standard_t(3)
        u = rng.normal(0, 1)
        pa[i] = float(sig(x / 2 + u)); pb[i] = float(sig(-x / 2 + u)); lab[i] = era[e]
    return pa, pb, lab


def bootstrap_answers(ids, ev, donors, rng):
    """donors: list of events, each a list of (pA, pB, label) from his sample."""
    pa, pb, lab = {}, {}, {}
    pick = {}
    for i in ids:
        e = ev[i]
        if e not in pick:
            pick[e] = donors[rng.integers(len(donors))]
        d = pick[e][rng.integers(len(pick[e]))]
        pa[i], pb[i], lab[i] = float(d[0]), float(d[1]), d[2]
    return pa, pb, lab


# ------------------------------------------------------------------------------------------------------------ scoring
def score(first, full, lab, n):
    srt = [i for i in first]
    corr = sum(first[i] == full.get(i, "x") for i in srt)
    r = dict(sorted=len(srt) / n, correct=corr / n, prec_sorted=corr / len(srt) if srt else np.nan)
    for a, nm in ((0, "A"), (1, "B")):
        fa = [i for i in srt if first[i] == a]; ga = sum(1 for v in full.values() if v == a)
        r[f"{nm}_prec"] = (sum(full.get(i) == a for i in fa) / len(fa)) if fa else np.nan
        r[f"{nm}_rec"] = (sum(full.get(i) == a for i in fa) / ga) if ga else np.nan
        if lab is not None:
            want, other = ("H", "F") if a == 0 else ("F", "H")
            r[f"{nm}_lab_ok"] = sum(lab.get(i) == want for i in fa); r[f"{nm}_lab_wrong"] = sum(lab.get(i) == other for i in fa)
    return r


def run_library(ids, ev, pa, pb, lab, budgets, seeds, extra=None):
    full = PF.pair_albums(ids, ev, pa, pb)
    rows = []
    for walk in WALKS:
        for sd in seeds:
            o = order_of(walk, ids, ev, sd)
            cov = coverage_curve(o, ids, ev)
            for b in budgets:
                k = min(len(ids), b // 2)
                done = [ids[q] for q in o[:k]]
                first = PF.pair_albums(ids, ev, {i: pa[i] for i in done}, {i: pb[i] for i in done})
                rows.append(dict(walk=walk, seed=sd, calls=b, **(extra or {}), **score(first, full, lab, len(ids)),
                                 to50=calls_to(cov, .5), to90=calls_to(cov, .9)))
    fullc = {k: sum(v == k for v in full.values()) for k in (0, 1, None)}
    return pd.DataFrame(rows), fullc


def table(R, budgets, lab=True):
    L = ["| walk | calls | sorted | correctly sorted (of all) | precision among sorted | heavier prec / rec vs full | "
         "fit prec / rec vs full |" + (" labels: heavier ok + wrong / fit ok + wrong |" if lab else ""),
         "|---|---|---|---|---|---|---|" + ("---|" if lab else "")]
    for w in WALKS:
        for b in budgets:
            z = R[(R.walk == w) & (R.calls == b)]
            if z.empty:
                continue
            c = (f"| {w} | {b:,} | {z.sorted.mean():.3f} | {z.correct.mean():.3f} | {z.prec_sorted.mean():.3f} | "
                 f"{z.A_prec.mean():.3f} / {z.A_rec.mean():.3f} | {z.B_prec.mean():.3f} / {z.B_rec.mean():.3f} |")
            if lab:
                c += (f" {z.A_lab_ok.mean():.1f} + {z.A_lab_wrong.mean():.1f} / {z.B_lab_ok.mean():.1f} + "
                      f"{z.B_lab_wrong.mean():.1f} |")
            L.append(c)
    L.append("calls to sort 50% / 90% of photos (pair, 2 calls a photo; median over runs): " + "; ".join(
        f"{w} {int(R[R.walk == w].to50.median()):,} / {int(R[R.walk == w].to90.median()):,}" for w in WALKS))
    return L


def coverage_lines(name, ids, ev, seeds=range(SEEDS)):
    rows = []
    for w in WALKS:
        for sd in seeds:
            cov = coverage_curve(order_of(w, ids, ev, sd), ids, ev)
            rows.append({"walk": w, **{b: cov[min(len(ids), b // 2)] for b in BUDGETS}, "to50": calls_to(cov, .5),
                         "to90": calls_to(cov, .9)})
    R = pd.DataFrame(rows)
    out = [f"{name}: share of photos sorted (event reached) at " + " / ".join(f"{b:,}" for b in BUDGETS)
           + " pair calls; calls to 50% / 90%"]
    for w in WALKS:
        z = R[R.walk == w]
        out.append(f"  {w}: " + " / ".join(f"{z[b].mean():.3f}" for b in BUDGETS)
                   + f"; {int(z.to50.median()):,} / {int(z.to90.median()):,}")
    return out


# ------------------------------------------------------------------------------------------------------------ commands
def cmd_public():
    fits = lognormal_fits()
    L = ["RESULTS 62b (public part): coverage-first walk for LOOK person searches. Pair = 2 judge calls per photo.", "",
         "S1 his event sizes (M72(a): 41,307 photos, 5,764 events, median 3, max 517, 1,689 singletons), stand-in = 1,689",
         "singletons + 4,075 events of size 2 + floor(Y), Y lognormal (deterministic quantiles, largest = 517, total exact):",
         "  fits keeping median 3 and mean 9.72: " + "; ".join(f"{k} mu {v[0]:.3f} sigma {v[1]:.3f}" for k, v in fits.items()), ""]
    for k, (mu, sg) in fits.items():
        sizes = his_sizes(mu, sg)
        top = np.sort(sizes)[::-1]
        L.append(f"  {k}: top-240 events hold {top[:240].sum() / N_PHOTOS:.3f} of photos, top-500 {top[:500].sum() / N_PHOTOS:.3f}, "
                 f"top-1000 {top[:1000].sum() / N_PHOTOS:.3f}; events of size >= 10: {(sizes >= 10).sum():,}")
        ids, ev = library(sizes, np.random.default_rng(0))
        L += ["  " + x for x in coverage_lines(f"S1-{k} coverage", ids, ev)]
    L.append("")
    # S1 parametric answers on the best-fit sizes
    mu, sg = fits["best"]; sizes = his_sizes(mu, sg)
    allR = []
    for d in range(DRAWS):
        rng = np.random.default_rng(620 + d)
        ids, ev = library(sizes, rng)
        pa, pb, lab = parametric_answers(ids, ev, rng)
        R, fullc = run_library(ids, ev, pa, pb, lab, BUDGETS, range(SEEDS), extra=dict(draw=d))
        allR.append(R)
        print("draw", d, "full albums", fullc, flush=True)
    R = pd.concat(allR)
    R.to_csv(OUT / "r62b_s1_parametric.tsv", sep="\t", index=False)
    L.append(f"S1 best-fit sizes x PUBLIC parametric answers ({DRAWS} draws x {SEEDS} walk seeds; labels = the event's era):")
    L += table(R, BUDGETS)
    L.append("")
    # S2 the 41k DISBench proxy (label-free)
    from findpics.agent import events
    it = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet", columns=["item_id", "path", "taken"])
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    users = sorted(it.user.unique()); rng = np.random.default_rng(6241)
    for d in range(3):
        us = rng.choice(users, size=21, replace=False)
        g = it[it.user.isin(us)]
        ev = {}
        for u in us:
            gu = g[g.user == u]
            ev.update({str(i): f"{u}:{int(e)}" for i, e in zip(gu.item_id, events(gu.taken.reset_index(drop=True)))})
        ids = list(ev)
        sz = pd.Series(ev).value_counts()
        L += coverage_lines(f"S2 41k DISBench proxy draw {d} ({len(ids):,} photos, {len(sz):,} events, median {int(sz.median())}, "
                            f"max {sz.max()}, singletons {(sz == 1).sum():,})", ids, ev, seeds=range(3))
    txt = "\n".join(L)
    (OUT / "r62b_public.txt").write_text(txt + "\n")
    print(txt)


def his_sample():
    sys.path.insert(0, str(ROOT / "scripts"))
    import replay_pairing as RP
    from findpics.agent import events
    lab0 = json.loads(RP.LABELS.read_text())
    items = pd.read_parquet(RP.INDEX / "items.parquet")
    dated = pd.to_datetime(items["taken"], utc=True, errors="coerce", format="ISO8601").notna().to_numpy()
    ev_all = {str(i): int(x) for i, x, ok in zip(items["item_id"], events(items["taken"]), dated) if ok}
    run = [r for r in RP.runs() if r.name == "mode_q3vl4b_real"][0]
    pl = RP.pools_of(run)
    A_, B_ = [j[j["where"] == "identity_match"] for _, j in pl]
    pa = dict(zip(A_.item_id.astype(str), A_.p_attr)); pb = dict(zip(B_.item_id.astype(str), B_.p_attr))
    ids = sorted(i for i in pa if i in pb)
    name_of = dict(zip(A_.item_id.astype(str), [Path(p).name for p in A_.path]))
    lab = {i: lab0.get(name_of[i]) for i in ids}
    ev = {i: str(ev_all[i]) for i in ids if i in ev_all}
    return ids, ev, pa, pb, lab


def cmd_private():
    ids, ev, pa, pb, lab = his_sample()
    L = ["RESULTS 62b (PRIVATE, numbers only): coverage-first walk. Phone judge run (q3vl4b_real) of his labelled sample.", ""]
    R3, fullc = run_library(ids, ev, pa, pb, lab, S3_BUDGETS, range(20))
    sz = pd.Series([PF._event(ev, i) for i in ids]).value_counts()
    L.append(f"S3 his sample: {len(ids)} photos, {len(sz)} events (median {int(sz.median())}, max {sz.max()}, singletons "
             f"{(sz == 1).sum()}); judge-everything albums heavier {fullc[0]} / fit {fullc[1]} / neither {fullc[None]}; 20 seeds:")
    L += table(R3, S3_BUDGETS)
    L.append("")
    donors = {}
    for i in ids:
        donors.setdefault(PF._event(ev, i), []).append((pa[i], pb[i], lab[i]))
    donors = list(donors.values())
    fits = lognormal_fits(); sizes = his_sizes(*fits["best"])
    allR = []
    for d in range(DRAWS):
        rng = np.random.default_rng(720 + d)
        lids, lev = library(sizes, rng)
        a, b, lb = bootstrap_answers(lids, lev, donors, rng)
        R, fullc = run_library(lids, lev, a, b, lb, BUDGETS, range(SEEDS), extra=dict(draw=d))
        allR.append(R)
        print("draw", d, "full", fullc, flush=True)
    R1 = pd.concat(allR)
    L.append(f"S1 best-fit his sizes x answers BOOTSTRAPPED from his sample's {len(donors)} events ({DRAWS} draws x {SEEDS} seeds):")
    L += table(R1, BUDGETS)
    L.append("")
    # decision (pre-registered)
    P = pd.read_csv(OUT / "r62b_s1_parametric.tsv", sep="\t")
    at = lambda R, w: R[(R.walk == w) & (R.calls == 480)]
    i1 = all(at(R, "size").correct.mean() > max(at(R, "walk3").correct.mean(), at(R, "walk1").correct.mean()) for R in (P, R1))
    i2 = all(at(R, "size").prec_sorted.mean() >= .95 for R in (P, R1))
    s, w3 = at(R3, "size"), at(R3, "walk3")
    i3 = (s.A_prec.mean() >= .95 and s.B_prec.mean() >= .95 and
          (s.A_lab_wrong + s.B_lab_wrong).mean() <= (w3.A_lab_wrong + w3.B_lab_wrong).mean() + 4)
    L.append(f"DECISION at 480 calls: (i) correctly sorted, size vs walk3 / walk1: parametric {at(P, 'size').correct.mean():.3f} vs "
             f"{at(P, 'walk3').correct.mean():.3f} / {at(P, 'walk1').correct.mean():.3f}, bootstrap {at(R1, 'size').correct.mean():.3f} vs "
             f"{at(R1, 'walk3').correct.mean():.3f} / {at(R1, 'walk1').correct.mean():.3f} -> {i1}; (ii) precision among sorted "
             f"{at(P, 'size').prec_sorted.mean():.3f} / {at(R1, 'size').prec_sorted.mean():.3f} -> {i2}; (iii) his sample: heavier / fit "
             f"precision {s.A_prec.mean():.3f} / {s.B_prec.mean():.3f}, wrong-album by labels {(s.A_lab_wrong + s.B_lab_wrong).mean():.1f} "
             f"vs walk3 {(w3.A_lab_wrong + w3.B_lab_wrong).mean():.1f} -> {i3}. SIZE WALK {'ADOPTED' if i1 and i2 and i3 else 'NOT adopted'}")
    priv = ROOT / "data/private/audits"
    R1.to_csv(priv / "r62b_s1_bootstrap.tsv", sep="\t", index=False); R3.to_csv(priv / "r62b_s3.tsv", sep="\t", index=False)
    txt = "\n".join(L)
    (priv / "r62b_private.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    {"public": cmd_public, "private": cmd_private}[sys.argv[1]]()
