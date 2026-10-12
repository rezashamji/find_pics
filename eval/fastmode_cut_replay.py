"""Fast mode's FIRST ANSWER at two "Is this a photo of X?" cuts (eval/judge_cut_sweep/fastmode_replay.txt). No model runs.

Why: eval/judge_cut_sweep/report.txt recommends subject cut 0.99 -> 0.999. The same cut is fast mode's `accept`
(Search.swift: phoneFastParams(accept: judgeCutoff(...))), so the head rule (stop once < 3% of the last 100 head photos
are yes) sees fewer yeses and may stop earlier, and the auto round (>= 3 random hits) may fire less.

Replays the SHIPPED rule (FindPicsCore.phoneFastParams == tune_stop_time rule SHIPPED below, RESULTS 43/44) with
tune_stop_time.sim_timed on audit4 (tune_stop_rule.audit4: the RESULTS 34 set, real phone judge P, PE-Core-B-16 rank),
20 seeds, at each cut in CUTS (score_apple_photos.SUBJECT_CUT). Object searches keep 0.7 at both (sanity: identical).
Per search and cut: judge-yes in the library, where the head rule alone is satisfied (seed-free: inter=0), the first
answer's head, judge calls, rounds, found, bound (missed_upper, recall lower bound), minutes to the first answer on the
measured phone clocks (m4 / m8 / fit4, RESULTS 44), and eye precision / recall (RESULTS 34 estimator,
eval_recall.score_set, unsure = no match; and unsure = match). Also: exhaustive (every photo judged) eye numbers per
cut, the RESULTS 41 rule (tune_stop_rule.simulate, head-only) for reference, and the 44b continuation (keep looking
after the first answer at the m4 hot-floor pace, duty 1.0 / 0.5) eye recall at 40/60/90 min.

ev16 (the 16-library validation set) is NOT replayable here: it stores only yes sets at a uniform 0.7 cut, no P.

  python eval/fastmode_cut_replay.py   (Slurm CPU job: ~minutes; writes eval/judge_cut_sweep/fastmode_replay.{tsv,txt})
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
import score_apple_photos as S  # noqa: E402
import tune_stop_rule as T  # noqa: E402
import tune_stop_time as TT  # noqa: E402
from eval_recall import QUERIES, _labels_eye, load_scores, old_labels, score_set  # noqa: E402

OUT = Path("eval/judge_cut_sweep")
SHIPPED = "timed round_calls=120 inter=0 topup=30 head=150 a1=0.2 fast_stop=480 stop_topup=150"
CUTS = (0.99, 0.999)
CLOCKS = ("m4", "m8", "fit4")
PACE = ((1.0, (40, 60, 90)), (0.5, (40, 60, 90)))
SHORT = {q: q.split()[-1] if q.startswith("photos with") else q.split()[0] for q in QUERIES}


def head_rule_b(y, r):
    """Head size at which sim_timed's head_rule is first satisfied (inter=0: the head is judged in rank order, so this
    does not depend on the seed). The first answer may stop before it (fast_stop budget)."""
    n = len(y); cap = min(r["head_max"], n); b = r["head"]
    while b < cap:
        last = y[max(0, b - r["window"]):b]
        if last.sum() / max(len(last), 1) < r["rate"]:
            return b
        b += r["chunk"]
    return cap


def cases_at(cut):
    S.SUBJECT_CUT = cut
    return T.audit4()


def main():
    kind, r = TT.parse(SHIPPED)
    assert kind == "timed" and r["window"] == 100 and r["head_max"] == 2000 and r["rate"] == 0.03 and r["hits"] == 3
    df = load_scores(); old = old_labels(); eye_l = _labels_eye()
    rows, eye_rows, ex_rows, r41_rows, pace_rows = [], [], [], [], []
    for cut in CUTS:
        cases = cases_at(cut)
        # exhaustive reference (every photo judged at this cut)
        for um in (False, True):
            E = score_set(df, old, eye_l, {c["key"]: set(c["ids"][c["y"]]) for c in cases}, unsure_match=um, B=10)
            for e in E.itertuples():
                ex_rows.append(dict(cut=cut, query=e.query, unsure_match=um, returned=e.returned,
                                    precision=e.precision_est, recall=e.recall))
        # RESULTS 41 rule (head-only rounds), judge-relative, for reference
        R41 = T.run(cases, TT.OLD)
        for x in R41.itertuples():
            r41_rows.append(dict(cut=cut, query=x.key, kept=x.kept, found=x.found, calls=x.calls, head1=x.head1,
                                 bound=x.bound, rounds=x.rounds))
        sims = {}
        for c in cases:
            y = c["y"]; n = len(y); brule = head_rule_b(y, r)
            for sd in range(T.SEEDS):
                order, rounds = TT.sim_timed(y, r, sd)
                eo, _ = TT.sim_timed(y, r, sd, exhaustive=True)
                assert (eo[:len(order)] == order).all()
                sims[(c["key"], sd)] = (order, eo)
                cert = rounds[-1][1]
                b_end = n - cert["n_tail"]
                row = dict(cut=cut, query=c["key"], seed=sd, n=n, judge_yes=int(y.sum()), head_rule_b=brule,
                           head_end=b_end, head_rule_met=b_end >= brule, calls=len(order), rounds=len(rounds),
                           auto_round=int(b_end > brule), found=cert["found"], tail_sampled=cert["tail_sampled"],
                           tail_hits=cert["tail_hits"], missed_upper=cert["missed_upper"],
                           recall_lower=cert["found"] / (cert["found"] + cert["missed_upper"])
                           if cert["found"] + cert["missed_upper"] > 0 else 0.0,
                           judge_rel_recall=cert["found"] / max(int(y.sum()), 1))
                assert int(y[order].sum()) == cert["found"]
                for clk in CLOCKS:
                    row[f"min_{clk}"] = TT.minutes_of(len(order), clk)
                rows.append(row)
        # eye precision / recall of the first answer, per seed
        for sd in range(T.SEEDS):
            keep = {}
            for c in cases:
                o = sims[(c["key"], sd)][0]
                keep[c["key"]] = set(c["ids"][o[c["y"][o]]])
            for um in (False, True):
                E = score_set(df, old, eye_l, keep, unsure_match=um, B=10)
                for e in E.itertuples():
                    eye_rows.append(dict(cut=cut, query=e.query, seed=sd, unsure_match=um, returned=e.returned,
                                         precision=e.precision_est, recall=e.recall,
                                         real_returned=e.real_returned_est, real=e.real_est))
            # 44b: keep looking after the first answer (m4 measured curve, then hot floor x duty)
            for duty, marks in PACE:
                for m in marks:
                    keep = {}
                    for c in cases:
                        fo, eo = sims[(c["key"], sd)]
                        o = eo[:max(0, TT.paced_calls(m, "m4", duty, len(fo)))]
                        keep[c["key"]] = set(c["ids"][o[c["y"][o]]])
                    E = score_set(df, old, eye_l, keep, B=10)
                    for e in E.itertuples():
                        pace_rows.append(dict(cut=cut, query=e.query, seed=sd, duty=duty, minute=m, recall=e.recall,
                                              precision=e.precision_est, real_returned=e.real_returned_est,
                                              real=e.real_est))
        print("cut", cut, "done", flush=True)
    D = pd.DataFrame(rows); Y = pd.DataFrame(eye_rows); X = pd.DataFrame(ex_rows)
    R41 = pd.DataFrame(r41_rows); P = pd.DataFrame(pace_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "fastmode_replay_runs.tsv", sep="\t", index=False, float_format="%.5g")
    Y.to_csv(OUT / "fastmode_replay_eye.tsv", sep="\t", index=False, float_format="%.5g")
    X.to_csv(OUT / "fastmode_replay_exhaustive.tsv", sep="\t", index=False, float_format="%.5g")
    R41.to_csv(OUT / "fastmode_replay_r41.tsv", sep="\t", index=False, float_format="%.5g")
    P.to_csv(OUT / "fastmode_replay_pace.tsv", sep="\t", index=False, float_format="%.5g")
    report(D, Y, X, R41, P)


def f(v, d=1):
    return "wall" if not np.isfinite(v) else f"{v:.{d}f}"


def report(D, Y, X, R41, P):
    L = []
    L.append("FAST MODE FIRST ANSWER AT SUBJECT CUT 0.99 vs 0.999 (2026-10-11; public DISBench audit4; no model runs)")
    L.append(f"Code eval/fastmode_cut_replay.py. Rule = the shipped phoneFastParams: {SHIPPED}")
    L.append("  (head 150, +50 while >= 3% of the last 100 head photos are yes, head cap 2,000, rounds of 120 x 2^k calls,"
             " one extra round if its random sample has >= 3 hits, first answer ends at the round reaching 480 calls,"
             " sample topped up to 150).")
    L.append("Set = audit4 (4 DISBench libraries as one 7,886-photo library), judge = Qwen3-VL-4B real phone weights, stored P"
             " on every photo; rank = PE-Core-B-16. 20 tail-sample seeds; numbers are means over seeds [min-max] unless stated.")
    L.append("Eye truth = RESULTS 34 estimator (eval_recall.score_set; V1 labels, unsure = no match unless stated).")
    L.append("Object searches (dog/car/bicycle) keep 0.7 at both: they must come out identical (sanity check).")
    L.append("ev16 (RESULTS 41/43/44's 16-library validation) CANNOT be replayed at either subject cut: it stores only the"
             " judge's yes sets at a uniform 0.7 cut (no P), so its subject searches were never at 0.99 either.")
    L.append("")
    L.append("Columns: yes = judge-yes in the 7,886 at that cut | stop@ = head size where the head rule alone fires"
             " (seed-free) | head = head judged at the first answer | calls | min m4/m8 = minutes to the first answer on"
             " the measured clocks | found = judge-yes shown | bound = 'at most N matches missed' (95%) and the recall"
             " lower bound shown | eye P / R = eye precision / recall of what is shown (unsure = no match) | R(u) = recall"
             " with unsure = match | ex R = exhaustive eye recall at that cut (every photo judged).")
    hdr = (f"{'search':8s} {'cut':>5s} {'yes':>5s} {'stop@':>5s} {'head':>6s} {'calls':>6s} {'min m4':>6s} {'m8':>5s}"
           f" {'found':>6s} {'found range':>11s} {'bound':>6s} {'R>=':>5s} {'eye P':>5s} {'eye R [min-max]':>19s}"
           f" {'R(u)':>5s} {'ex P':>5s} {'ex R':>5s}")
    L.append(hdr)
    flags = []
    by = {}
    for q in QUERIES:
        for cut in CUTS:
            d = D[(D['query'] == q) & (D.cut == cut)]
            e = Y[(Y["query"] == q) & (Y.cut == cut) & (~Y.unsure_match)]
            eu = Y[(Y["query"] == q) & (Y.cut == cut) & (Y.unsure_match)]
            x = X[(X["query"] == q) & (X.cut == cut) & (~X.unsure_match)].iloc[0]
            s = dict(yes=d.judge_yes.iloc[0], stop=d.head_rule_b.iloc[0], head=d.head_end.mean(), calls=d.calls.mean(),
                     m4=d.min_m4.mean(), m8=d.min_m8.mean(), found=d.found.mean(), fmin=d.found.min(), fmax=d.found.max(),
                     bound=d.missed_upper.mean(), rl=d.recall_lower.mean(), P=e.precision.mean(), R=e.recall.mean(),
                     Rmin=e.recall.min(), Rmax=e.recall.max(), Ru=eu.recall.mean(), exP=x.precision, exR=x.recall,
                     auto=d.auto_round.mean(), met=d.head_rule_met.mean())
            by[(q, cut)] = s
            L.append(f"{SHORT[q]:8s} {cut:5g} {s['yes']:5d} {s['stop']:5d} {s['head']:6.0f} {s['calls']:6.0f}"
                     f" {f(s['m4']):>6s} {f(s['m8']):>5s} {s['found']:6.1f} {s['fmin']:5d}-{s['fmax']:<5d} {s['bound']:6.0f}"
                     f" {s['rl']:5.3f} {s['P']:5.2f} {s['R']:.3f} [{s['Rmin']:.3f}-{s['Rmax']:.3f}] {s['Ru']:5.3f}"
                     f" {s['exP']:5.2f} {s['exR']:5.3f}")
        a, b = by[(q, CUTS[0])], by[(q, CUTS[1])]
        dR = b["R"] - a["R"]
        why = []
        if dR < -0.03:
            why.append(f"first-answer eye recall {a['R']:.3f} -> {b['R']:.3f} ({dR:+.3f})")
        if b["stop"] < a["stop"]:
            why.append(f"head rule fires earlier: head {a['stop']} -> {b['stop']}")
        if b["head"] < a["head"] - 0.5:
            why.append(f"first answer's head shorter: {a['head']:.0f} -> {b['head']:.0f}")
        if why:
            flags.append(f"  {SHORT[q]}: " + "; ".join(why))
        L.append(f"{'':8s} delta: eye P {b['P'] - a['P']:+.2f}, eye R {dR:+.3f}, R(u) {b['Ru'] - a['Ru']:+.3f},"
                 f" calls {b['calls'] - a['calls']:+.0f}, min m4 {b['m4'] - a['m4']:+.1f}, found {b['found'] - a['found']:+.1f},"
                 f" auto round share {a['auto']:.2f} -> {b['auto']:.2f}, head rule met {a['met']:.2f} -> {b['met']:.2f}")
    L.append("")
    L.append("Pooled over the 3 subject searches (eye, unsure = no match; real = estimated real matches, same denominator"
             " at both cuts):")
    subj = ["beach photos", "sunset photos", "food photos"]
    for cut in CUTS:
        e = Y[(Y.cut == cut) & (~Y.unsure_match) & Y["query"].isin(subj)]
        g = e.groupby("seed").apply(lambda t: pd.Series(dict(rr=t.real_returned.sum(), real=t.real.sum(), ret=t.returned.sum())),
                                    include_groups=False)
        d = D[(D.cut == cut) & D['query'].isin(subj)]
        L.append(f"  cut {cut:5g}: shown {g.ret.mean():6.1f}, real shown {g.rr.mean():6.1f} of {g.real.mean():6.1f}"
                 f" -> recall {(g.rr / g.real).mean():.3f}, precision {(g.rr / g.ret).mean():.3f};"
                 f" judge calls {d.calls.sum() / T.SEEDS:.0f} (3 searches)")
    L.append("")
    L.append("FLAGS (first-answer eye recall drop > 0.03, or the stop rule fires earlier / shorter head):")
    L.extend(flags or ["  none"])
    L.append("")
    L.append("RESULTS 41 rule (head-only rounds, tune_stop_rule.simulate with window=100 head_max=2000 more=hits>=3,max2),"
             " judge-relative, for reference:")
    for q in QUERIES:
        cells = []
        for cut in CUTS:
            x = R41[(R41['query'] == q) & (R41.cut == cut)].iloc[0]
            cells.append(f"cut {cut:5g}: found {x.found:6.1f}/{x.kept:<5d} head1 {x.head1:5.0f} calls {x.calls:5.0f}"
                         f" bound {x.bound:5.0f} rounds {x.rounds:.2f}")
        L.append(f"  {SHORT[q]:8s} " + " | ".join(cells))
    L.append("")
    L.append("44b continuation (keep looking after the first answer; ASSUMED m4 hot-floor pace x duty; eye recall, unsure ="
             " no match, mean of 20 seeds):")
    for duty, marks in PACE:
        for q in QUERIES:
            cells = []
            for cut in CUTS:
                p = P[(P["query"] == q) & (P.cut == cut) & (P.duty == duty)].groupby("minute")[["recall", "precision"]].mean()
                cells.append(f"cut {cut:5g}: " + " ".join(f"{m}m R{p.loc[m, 'recall']:.3f}/P{p.loc[m, 'precision']:.2f}" for m in marks))
            L.append(f"  duty {duty:.1f} {SHORT[q]:8s} " + " | ".join(cells))
    txt = "\n".join(L)
    print(txt)
    (OUT / "fastmode_replay_tables.txt").write_text(txt)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:      # tables again from the saved TSVs
        rd = lambda n: pd.read_csv(OUT / f"fastmode_replay_{n}.tsv", sep="\t")
        report(rd("runs"), rd("eye"), rd("exhaustive"), rd("r41"), rd("pace"))
    else:
        main()
