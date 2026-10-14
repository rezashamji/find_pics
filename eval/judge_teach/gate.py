"""RESULTS 58 teacher gate + eye-set tables (rule pre-registered in JOURNAL 10-14 00:30).

Groups: D1 = the 6 everyday subject searches, D2 = person red-box facts (7 conditions), D3 = complex dev b (cb_b);
recall guards: OBJ = dog/car/bicycle/boat, FACT = night/selfies. Unsure rows are dropped.
(a) at the PRODUCT cut of each row: kept right / kept wrong, precision, recall per judge and group;
(b) EQUAL RECALL: per search s, R*_s = the reference judge's recall at the product cut; every judge (reference
    included) takes its highest threshold with recall >= R*_s on s; TP/FP summed per group -> precision;
(c) within-search AUC (pairs of match / no-match inside one search, pooled over the group's searches).
Paired bootstrap: rows resampled within each search (same draws for every judge), everything re-derived per draw.
Usage: python eval/judge_teach/gate.py [--ref=phone_v8] [--lib=heldout|ev16|all] [judge names...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "eval/judge_teach"
GROUPS = {"D1 subjects": lambda E: (E.set == "everyday") & (E.kind == "subject"),
          "D2 person facts": lambda E: (E.set == "person") & (E.kind == "person_fact"),
          "D3 cb_b attributes": lambda E: E.set == "cb_b",
          "OBJ objects": lambda E: (E.set == "everyday") & (E.kind == "object"),
          "FACT night/selfie": lambda E: (E.set == "everyday") & (E.kind == "fact")}
DISPUTED = ["D1 subjects", "D2 person facts", "D3 cb_b attributes"]
UNSURE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--unsure=")), "drop")


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def load(names, lib="heldout", eyeset=None):
    E = pd.read_parquet(eyeset or D / "eyeset.parquet")
    P = {}
    def one(n):
        x = pd.read_parquet(D / "eye_p" / f"{n}.parquet")
        assert len(x) == len(E) and (x.item_id.to_numpy() == E.item_id.to_numpy()).all(), n
        return x.p.to_numpy(np.float64)
    for n in names:
        if n.startswith("ens:"):            # mean logit gap of several judges
            ps = [np.clip(one(m), 1e-9, 1 - 1e-9) for m in n[4:].split("+")]
            P[n] = 1 / (1 + np.exp(-np.mean([np.log(p / (1 - p)) for p in ps], 0)))
        else:
            P[n] = one(n)
    if UNSURE == "no_match":                 # sensitivity: the labellers' default protocol (unsure = not a match)
        E = E.assign(label=E.label.replace("unsure", "no_match"))
    keep = (E.label != "unsure").to_numpy().copy()
    if lib != "all":
        keep &= (E.lib == lib).to_numpy()
    for n in names:
        keep &= ~np.isnan(P[n])
    E = E[keep].reset_index(drop=True)
    P = {n: p[keep] for n, p in P.items()}
    return E, P


def _search_blocks(E, mask):
    out = []
    for s, g in E[mask].groupby("search", sort=True):
        out.append((s, g.index.to_numpy(), (g.label == "match").to_numpy(), g.cut.to_numpy()))
    return out


def stats(blocks, P, names, ref, draws=None):
    """-> {name: dict(tp_cut, fp_cut, nm, tp_eq, fp_eq, auc_num, auc_den)} for one (re)sample of rows."""
    r = {n: dict(tp_cut=0, fp_cut=0, nm=0, tp_eq=0, fp_eq=0, an=0.0, ad=0.0) for n in names}
    for b, (s, ix, y, cut) in enumerate(blocks):
        if draws is not None:
            sel = draws[b]
            ix, y, cut = ix[sel], y[sel], cut[sel]
        nm = int(y.sum())
        pr = P[ref][ix]
        rstar = (pr[y] >= cut[y]).mean() if nm else 0.0
        k = int(np.ceil(rstar * nm - 1e-9))
        for n in names:
            p = P[n][ix]
            kept = p >= cut
            d = r[n]
            d["tp_cut"] += int((kept & y).sum()); d["fp_cut"] += int((kept & ~y).sum()); d["nm"] += nm
            if k > 0:
                t = np.sort(p[y])[::-1][k - 1]
                d["tp_eq"] += int((p[y] >= t).sum()); d["fp_eq"] += int((p[~y] >= t).sum())
            pm, pn = p[y], p[~y]
            if len(pm) and len(pn):
                # Mann-Whitney with ties = 0.5
                rk = rankdata(np.concatenate([pm, pn]))
                d["an"] += rk[:len(pm)].sum() - len(pm) * (len(pm) + 1) / 2; d["ad"] += len(pm) * len(pn)
    return r


def summ(d):
    pc = d["tp_cut"] / max(d["tp_cut"] + d["fp_cut"], 1)
    rc = d["tp_cut"] / max(d["nm"], 1)
    pe = d["tp_eq"] / max(d["tp_eq"] + d["fp_eq"], 1)
    auc = d["an"] / d["ad"] if d["ad"] else np.nan
    return dict(prec_cut=pc, rec_cut=rc, prec_eq=pe, auc=auc)


def merge(rs):
    out = {}
    for r in rs:
        for n, d in r.items():
            o = out.setdefault(n, dict.fromkeys(d, 0))
            for k, v in d.items():
                o[k] += v
    return out


def report(names, ref, lib="heldout", B=2000, seed=0, eyeset=None, out=print, compare=None, alpha=0.05):
    E, P = load(names, lib, eyeset)
    compare = compare or [n for n in names if n != ref]
    rng = np.random.default_rng(seed)
    gb = {g: _search_blocks(E, f(E).to_numpy()) for g, f in GROUPS.items()}
    point = {g: stats(bl, P, names, ref) for g, bl in gb.items()}
    point["D1+D2+D3"] = merge([point[g] for g in DISPUTED])
    boots = {g: [] for g in point}
    for _ in range(B):
        rr = {}
        for g, bl in gb.items():
            draws = [rng.integers(0, len(ix), len(ix)) for _, ix, _, _ in bl]
            rr[g] = stats(bl, P, names, ref, draws)
        rr["D1+D2+D3"] = merge([rr[g] for g in DISPUTED])
        for g in rr:
            boots[g].append({n: summ(d) for n, d in rr[g].items()})
    out(f"lib={lib}  ref={ref}  rows={len(E)} (unsure dropped)  bootstrap {B}x within search, {100 - 100 * alpha:.2f}% CIs")
    for g in point:
        n_s = sum(len(gb[x]) for x in (DISPUTED if g == "D1+D2+D3" else [g]))
        out(f"\n[{g}]  searches {n_s}, match rows {point[g][ref]['nm']}")
        out(f"  {'judge':22s} {'kept right/wrong @cut':>22s} {'prec@cut':>9s} {'rec@cut':>8s} {'prec@eqR':>9s} {'AUC':>6s}"
            f"  {'d prec@cut vs ref':>22s} {'d rec@cut':>22s} {'d prec@eqR':>22s} {'d AUC':>22s}")
        for n in names:
            d = point[g][n]; s = summ(d)
            line = (f"  {n:22s} {d['tp_cut']:>10d} / {d['fp_cut']:<10d} {s['prec_cut']:9.3f} {s['rec_cut']:8.3f} "
                    f"{s['prec_eq']:9.3f} {s['auc']:6.3f}")
            if n != ref:
                for k in ("prec_cut", "rec_cut", "prec_eq", "auc"):
                    v = np.array([b[n][k] - b[ref][k] for b in boots[g]])
                    lo, hi = np.nanpercentile(v, [100 * alpha / 2, 100 - 100 * alpha / 2])
                    line += f"  {s[k] - summ(point[g][ref])[k]:+.3f} [{lo:+.3f},{hi:+.3f}]"
            out(line)
        for a in compare:
            for b in compare:
                if a < b and a != ref and b != ref:
                    v = np.array([x[a]["prec_eq"] - x[b]["prec_eq"] for x in boots[g]])
                    lo, hi = np.nanpercentile(v, [100 * alpha / 2, 100 - 100 * alpha / 2])
                    out(f"    prec@eqR {a} - {b}: {summ(point[g][a])['prec_eq'] - summ(point[g][b])['prec_eq']:+.3f} "
                        f"[{lo:+.3f},{hi:+.3f}]")
    return point, boots


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    ref = arg("ref", "phone_v8")
    if ref not in names:
        names = [ref] + names
    report(names, ref, arg("lib", "heldout"), int(arg("B", "2000")), alpha=float(arg("alpha", "0.05")))
