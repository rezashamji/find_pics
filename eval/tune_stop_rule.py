"""Fast mode's round-1 stop rule, tuned OFFLINE on stored judge answers (RESULTS 41). No new judge calls.

Problem (RESULTS 40): the phone's fast mode (round 1 of FindPicsCore.streamRounds) keeps car 306 and bicycle 31 of the
410 / 48 photos the same judge keeps when it looks at everything. This replays streamRounds exactly (Streaming.swift:
head, +chunk while the last `window` judged head photos are >= `rate` yes, up to head_max; a uniform tail sample fixed
before any tail label; rounds k >= 2 double the head and sample tail_budget * 2^(k-1) of a permutation fixed in advance;
certificate = Completeness.certify at alpha/2 for round 1, alpha/2/nLater after) on recorded yes/no answers, with
candidate rules for when round 1 stops and whether fast mode runs more rounds by itself.

Data:
  audit4  the RESULTS 34/40 set: 4 DISBench libraries as ONE 7,886-photo library, 6 searches; judge = the real phone
          weights (eval/recall_audit/scores_q3vl4b_real, P(yes) on every photo), cut = judgeCutoff (0.99 / 0.7),
          ranking = PE-Core-B-16 (the phone's model, eval/apple_photos/looks_b16.parquet); `--rank=l` uses the
          server's PE-Core-L-14 look score stored with the judge answers instead.
  ev16    validation: the 16 DISBench libraries of RESULTS 27 (none of them in audit4), 12 searches each; yes = the
          exhaustive set of eval/everyday16_q3vl (Qwen3-VL-4B 16-bit judge on every in-scope photo, cut 0.7, after
          exclusions); ranking = engine.look_scores on data/public/index_disbench (PE-Core-L-14: B-16 vectors of these
          libraries do not exist). Prepared once by `prep16` -> eval/stop_rule/ev16_ranked.parquet (git-ignored).

  python eval/tune_stop_rule.py prep16            (GPU job: scripts/race_sbatch.sh; the login node takes >15 min to build PE-Core-L)
  python eval/tune_stop_rule.py sweep [--rank=l]  -> eval/stop_rule/sweep_<set>.tsv + printed tables
  python eval/tune_stop_rule.py eye RULE...       -> eye-label recall (RESULTS 34 truth, eval_recall.score_set)
  python eval/tune_stop_rule.py compare audit4|ev16 RULE...  -> per-search found / judge calls / auto-round rate
  python eval/tune_stop_rule.py sheets            -> contact sheets of the photos the adopted rule adds (viewed)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

sys.path.insert(0, "eval")
OUT = Path("eval/stop_rule")
SEEDS = 20
ALPHA = 0.05
SEC_PER_CALL = 1.0      # the phone judge, as Search.swift assumes (~1 photo/s)


def cp_upper(hits, n, alpha):
    return 1.0 if n == 0 or hits >= n else float(_beta.ppf(1 - alpha, hits + 1, n - hits))


def certify(found, n_tail, labels, alpha):
    """Completeness.certify (Swift) == audit.certify (Python)."""
    n, hits = len(labels), int(np.sum(labels))
    if n_tail == 0:
        return dict(found=found, n_tail=0, tail_sampled=0, tail_hits=0, missed_point=0.0, missed_upper=0.0)
    return dict(found=found, n_tail=n_tail, tail_sampled=n, tail_hits=hits,
                missed_point=n_tail * hits / n if n else float("nan"), missed_upper=n_tail * cp_upper(hits, n, alpha))


# ---------------------------------------------------------------------------------------------------------------- rules
# head/chunk/head_max/tail: Search.swift's StreamParams. window/rate: stop when fewer than rate*window of the last
# `window` judged head photos are yes (current: window = chunk = 50, rate 0.03 -> stop at 0 or 1 yes in the last 50).
# more: fast mode keeps running rounds k = 2, 3... while the round's certificate says so, at most max_rounds in all:
#   ("hits", h)   the round's random tail sample found >= h matches
#   ("point", m)  the point estimate of matches still hiding (n_tail * hits / sampled) >= m
#   ("upper", m)  the 95% bound shown to the person >= m
#   ("rel", f)    the point estimate of hiding matches >= f x found
CURRENT = dict(head=150, chunk=50, window=50, rate=0.03, head_max=1500, tail=150, more=None, max_rounds=1)


def rule(**kw):
    r = dict(CURRENT); r.update(kw); return r


def name(r):
    parts = []
    for k in ("head", "window", "rate", "head_max", "tail"):
        if r[k] != CURRENT[k]:
            parts.append(f"{k}={r[k]}")
    if r["more"]:
        parts.append(f"more={r['more'][0]}>={r['more'][1]},max{r['max_rounds']}")
    return "current" if not parts else " ".join(parts)


def keep_going(cert, more):
    kind, v = more
    if kind == "hits":
        return cert["tail_hits"] >= v
    if kind == "point":
        return cert["missed_point"] >= v
    if kind == "upper":
        return cert["missed_upper"] >= v
    if kind == "rel":
        return cert["missed_point"] >= v * max(cert["found"], 1)
    raise ValueError(kind)


def simulate(y, r, seed):
    """streamRounds on yes/no answers y (bool, best fast score first). Returns (found mask, judge calls, last shown
    certificate, rounds run, round-1 head)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    n_head = min(r["head"], n)
    while n_head < min(r["head_max"], n):
        last = y[max(0, n_head - r["window"]):n_head]
        if last.sum() / max(len(last), 1) < r["rate"]:
            break
        n_head = min(n_head + r["chunk"], n)
    head1 = n_head
    h, n_later = n_head, 0
    while True:
        h = min(n, 2 * h)
        if h < n:
            n_later += 1
        else:
            break
    perm = rng.permutation(n)
    judged = np.zeros(n, bool)
    k = 0
    while True:
        k += 1
        tail = n - n_head
        n_t = min(tail, r["tail"] * 2 ** (k - 1))
        ts = rng.permutation(np.arange(n_head, n))[:n_t] if k == 1 else perm[perm >= n_head][:n_t]
        judged[:n_head] = True; judged[ts] = True
        a_k = ALPHA / 2 if k == 1 else ALPHA / 2 / max(n_later, 1)
        cert = certify(int((judged & y).sum()), tail, y[ts], a_k)
        if n_head >= n or k >= r["max_rounds"] or not r["more"] or not keep_going(cert, r["more"]):
            break
        n_head = min(n, 2 * n_head)
    return judged & y, int(judged.sum()), cert, k, head1


def run(cases, r, seeds=SEEDS):
    """cases: list of dict(key, y). Per case: mean over seeds of found, calls, bound, rounds."""
    rows = []
    for c in cases:
        S = [simulate(c["y"], r, s) for s in range(seeds)]
        f = np.array([s[0].sum() for s in S]); j = np.array([s[1] for s in S])
        rows.append(dict(key=c["key"], kept=int(c["y"].sum()), n=len(c["y"]), found=f.mean(), found_min=f.min(),
                         found_max=f.max(), calls=j.mean(), calls_max=j.max(),
                         bound=np.mean([s[2]["missed_upper"] for s in S]), rounds=np.mean([s[3] for s in S]),
                         rounds_max=max(s[3] for s in S), head1=np.mean([s[4] for s in S])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------- data
def audit4(rank="b16"):
    from eval_recall import QUERIES
    from score_apple_photos import D, judge_cutoff
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores_q3vl4b_real").glob("part*.parquet"))])
    lk = pd.read_parquet(D / "looks_b16.parquet").set_index("item_id")
    cases = []
    for q in QUERIES:
        d = sc[sc["query"] == q].set_index("item_id")
        cut = judge_cutoff(d.question.iloc[0])
        s = lk.loc[d.index, q].to_numpy() if rank == "b16" else d.look.to_numpy()
        o = d.assign(s=s).sort_values("s", ascending=False, kind="stable")   # as score_apple_photos.findpics
        cases.append(dict(key=q, y=(o.p >= cut).to_numpy(), ids=o.index.to_numpy()))
    return cases


def prep16():
    from findpics import store
    from findpics.converse import Plan
    from findpics.engine import look_scores, scope_mask
    from findpics.models import ImageTextEncoder
    recs = [r for f in sorted(Path("eval/everyday16_q3vl").glob("part*.json")) for r in json.load(open(f))]
    print("loading index", flush=True)
    idx = store.load("data/public/index_disbench")
    print("loading text tower", idx.clip_model, flush=True)
    enc = ImageTextEncoder(idx.clip_model)
    print("ready", flush=True)
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    out = []
    for u in sorted({r["user"] for r in recs}):
        sub = store.subset(idx, np.where(user_of == u)[0])
        for r in [r for r in recs if r["user"] == u]:
            if r["error"]:
                continue
            a = Plan.model_validate(r["plan"]).albums
            if len(a) != 1:
                print("skip (several albums)", u, r["query"]); continue
            a = a[0]
            m = scope_mask(sub, a)
            s = look_scores(sub, enc, a.looks, a.avoid)
            ids = sub.items.item_id.astype(str).to_numpy()
            ex = set(map(str, r["exhaustive"]))
            out.append(pd.DataFrame(dict(user=u, query=r["query"], item_id=ids[m], score=s[m],
                                         yes=np.isin(ids[m], list(ex)))))
            print(u, r["query"], int(m.sum()), "in scope,", len(ex), "exhaustive,", int(out[-1].yes.sum()), "matched", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(OUT / "ev16_ranked.parquet")


def ev16():
    d = pd.read_parquet(OUT / "ev16_ranked.parquet")
    cases = []
    for (u, q), g in d.groupby(["user", "query"], sort=True):
        g = g.sort_values("score", ascending=False, kind="stable")
        cases.append(dict(key=f"{q}|{u}", query=q, user=u, y=g.yes.to_numpy(), ids=g.item_id.to_numpy()))
    return cases


# ---------------------------------------------------------------------------------------------------------------- sweep
def candidates():
    R = [CURRENT]
    for w in (50, 100, 150, 200, 300):                 # stop threshold and window
        for rt in (0.03, 0.02, 0.01, 0.005):
            if (w, rt) != (50, 0.03):
                R.append(rule(window=w, rate=rt))
    for w in (50, 100, 150, 200, 300):                 # head cap (the round-1 time budget)
        for rt in (0.03, 0.02, 0.015, 0.01):
            for hm in (1750, 2000, 2500, 3000, 4000, 7886):
                R.append(rule(window=w, rate=rt, head_max=hm))
    for h in (300, 500, 750, 1000):                    # minimum head
        R.append(rule(head=h))
        R.append(rule(head=h, window=200, rate=0.01))
    for tb in (300, 600):                              # bigger round-1 tail sample (tighter bound, finds more by chance)
        R.append(rule(tail=tb))
    for mr in (2, 3, 99):                              # automatic later rounds
        for more in (("hits", 1), ("hits", 2), ("hits", 3), ("point", 50), ("point", 100), ("upper", 200),
                     ("upper", 400), ("rel", 0.1), ("rel", 0.25)):
            R.append(rule(more=more, max_rounds=mr))
            R.append(rule(window=200, rate=0.01, more=more, max_rounds=mr))
            for w, rt, hm in ((100, 0.03, 2000), (200, 0.03, 2000), (150, 0.02, 2000), (300, 0.02, 2500)):
                R.append(rule(window=w, rate=rt, head_max=hm, more=more, max_rounds=mr))
    return R


def pooled(df, keys=None):
    d = df if keys is None else df[df.key.isin(keys)]
    return d.found.sum() / max(d.kept.sum(), 1), d.calls.sum()


def sweep(cases, tag):
    rows = []
    for r in candidates():
        df = run(cases, r)
        df.insert(0, "rule", name(r))
        rows.append(df)
        rec, calls = pooled(df)
        print(f"{name(r):60s} judge-relative recall {rec:.3f}  calls {calls:8.0f}", flush=True)
    A = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / f"sweep_{tag}.tsv", sep="\t", index=False, float_format="%.4g")
    return A


def table(A, rules, keys_label=None):
    """Per-query found/kept and calls for some rules."""
    lines = []
    for nm in rules:
        d = A[A.rule == nm]
        cells = [f"{k.split('|')[0][:14]:14s} {r.found:7.1f}/{r.kept:<5d} {r.calls:6.0f}c b{r.bound:5.0f} r{r.rounds:.1f}"
                 for k, r in zip(d.key, d.itertuples())]
        rec, calls = pooled(d)
        lines.append(f"{nm}\n   pooled {d.found.sum():.1f}/{d.kept.sum()} = {rec:.3f}, calls {calls:.0f} "
                     f"({calls * SEC_PER_CALL / 60:.0f} min)\n   " + "\n   ".join(cells))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------ eye
def parse_rule(s):
    if s == "current":
        return dict(CURRENT)
    r = dict(CURRENT)
    for p in s.split():
        k, v = p.split("=", 1)
        if k == "more":
            kind, rest = v.split(">=")
            m, mx = rest.split(",max")
            r["more"] = (kind, float(m)); r["max_rounds"] = int(mx)
        else:
            r[k] = float(v) if k == "rate" else int(v)
    return r


def eye(rules, seeds=SEEDS):
    """Recall/precision on the RESULTS 34 eye-label truth (eval_recall.score_set: same strata, labels, estimator as
    RESULTS 34/40). Each rule's returned sets differ by tail-sample seed (and auto rounds fire by chance), so every
    seed is scored; reported: mean [min, max] per query and pooled, plus exhaustive for reference."""
    from eval_recall import QUERIES, _labels_eye, load_scores, old_labels, score_set
    cases = audit4()
    df = load_scores(); old = old_labels(); eye_l = _labels_eye()
    lines, rows = [], []
    ex = score_set(df, old, eye_l, {c["key"]: set(c["ids"][c["y"]]) for c in cases}, B=10).set_index("query")
    lines.append("exhaustive: " + " | ".join(f"{q.split()[-1]} {ex.loc[q, 'recall']:.3f}" for q in QUERIES) +
                 f" | pooled {ex.real_returned_est.sum() / ex.real_est.sum():.3f}")
    for s in ["current"] + [r for r in rules if r != "current"]:
        r = parse_rule(s)
        per = []
        for sd in range(seeds):
            sims = {c["key"]: simulate(c["y"], r, sd) for c in cases}
            R = score_set(df, old, eye_l, {c["key"]: set(c["ids"][sims[c["key"]][0]]) for c in cases}, B=10).set_index("query")
            R["calls"] = [sims[q][1] for q in R.index]; R["rounds"] = [sims[q][3] for q in R.index]
            R["seed"] = sd; R["rule"] = s
            per.append(R.reset_index())
        P = pd.concat(per, ignore_index=True); rows.append(P)
        g = P.groupby("query", sort=False)
        pooled = P.groupby("seed").apply(lambda d: d.real_returned_est.sum() / d.real_est.sum(), include_groups=False)
        lines.append(f"-- {s}: pooled recall {pooled.mean():.3f} [{pooled.min():.3f}, {pooled.max():.3f}], "
                     f"calls {P.calls.sum() / seeds:.0f} ({P.calls.sum() / seeds * SEC_PER_CALL / 60:.0f} min)")
        for q in QUERIES:
            d = g.get_group(q)
            lines.append(f"  {q:22s} recall {d.recall.mean():.3f} [{d.recall.min():.3f}, {d.recall.max():.3f}]"
                         f"  precision {d.precision_est.mean():.2f}  calls {d.calls.mean():6.0f}"
                         f"  auto round 2 in {(d.rounds > 1).sum()}/{seeds} seeds")
    pd.concat(rows, ignore_index=True).to_csv(OUT / "eye_per_seed.tsv", sep="\t", index=False, float_format="%.4g")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "eye_report.txt").write_text(txt)


def compare(cases, rules, tag):
    """Per search (pooled over libraries for ev16): found / judge-yes, judge calls, auto round 2 rate, mean bound."""
    rows = []
    for s in rules:
        df = run(cases, parse_rule(s)); df.insert(0, "rule", s); rows.append(df)
    A = pd.concat(rows, ignore_index=True)
    A["query"] = A.key.str.split("|").str[0]
    A["fired"] = A.rounds - 1          # max_rounds 2: share of seeds with an automatic round 2
    A.to_csv(OUT / f"compare_{tag}.tsv", sep="\t", index=False, float_format="%.4g")
    lines = []
    for s, d in A.groupby("rule", sort=False):
        lines.append(f"-- {s}: found {d.found.sum():.1f} of {d.kept.sum()} judge-yes ({d.found.sum() / d.kept.sum():.3f}), "
                     f"calls {d.calls.sum():.0f} ({d.calls.sum() * SEC_PER_CALL / 60:.0f} min), "
                     f"{len(d)} searches, auto round 2 in {d.fired.mean():.1%} of search-seeds")
        for q, g in d.groupby("query", sort=False):
            lines.append(f"  {q:22s} found {g.found.sum():7.1f} / {g.kept.sum():5d} ({g.found.sum() / max(g.kept.sum(), 1):.3f})"
                         f"  calls {g.calls.sum():7.0f}  (median per library {g.calls.median():5.0f})"
                         f"  round 2 in {g.fired.mean():5.1%}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / f"compare_{tag}.txt").write_text(txt)
    return A


def gain_sheets(rule_s="window=100 head_max=2000 more=hits>=3,max2"):
    """Contact sheets (native pixels) of photos the rule adds over the old one: 12 random cars (from a seed where the
    auto round ran) and every bicycle -> eval/stop_rule/sheets/gain_*.jpg (git-ignored) + key.json."""
    import eval_recall as ER
    from PIL import ImageFont
    cases = {c["key"]: c for c in audit4()}
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores_q3vl4b_real").glob("part*.parquet"))])
    new = parse_rule(rule_s)
    ER.OUT = OUT; (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0); keys = []
    for q, short in (("photos with a car", "car"), ("photos with a bicycle", "bicycle")):
        c = cases[q]
        old = set(c["ids"][simulate(c["y"], CURRENT, 0)[0]])
        fire = [s for s in range(SEEDS) if simulate(c["y"], new, s)[3] > 1]
        sd = fire[0] if fire else 0
        gained = sorted(set(c["ids"][simulate(c["y"], new, sd)[0]]) - old)
        print(q, "seed", sd, "gained", len(gained))
        pick = list(rng.choice(gained, size=min(12, len(gained)), replace=False))
        d = sc[(sc["query"] == q) & sc.item_id.isin(pick)].drop_duplicates("item_id").assign(stratum="gain")
        keys += ER.sheets(list(d.itertuples()), f"gain_{short}", ImageFont.load_default())
    json.dump(keys, open(OUT / "sheets" / "key.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "prep16":
        prep16()
    elif cmd == "sweep":
        which = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--set=")), "audit4")
        rank = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rank=")), "b16")
        cases = audit4(rank) if which == "audit4" else ev16()
        sweep(cases, which + ("" if rank == "b16" or which != "audit4" else f"_{rank}"))
    elif cmd == "compare":   # python eval/tune_stop_rule.py compare audit4|ev16 RULE...
        compare(audit4() if sys.argv[2] == "audit4" else ev16(), sys.argv[3:], sys.argv[2])
    elif cmd == "sheets":
        gain_sheets()
    elif cmd == "eye":
        eye(sys.argv[2:])
    elif cmd == "show":
        A = pd.read_csv(OUT / f"sweep_{sys.argv[2]}.tsv", sep="\t")
        print(table(A, sys.argv[3:]))
