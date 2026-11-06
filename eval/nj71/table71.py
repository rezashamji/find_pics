"""RESULTS 71 table + pre-registered decisions D1-D4 (eval/RESULTS.md "## 71" + ADDENDUM).
  python eval/nj71/table71.py [--out=eval/nj71/table.txt]
Runs: eval/nj71/replay/s1/<split>/{DRY_fp32 (control), NJ_fp32_T<xx> (grid)}; eval/nj71/replay/s2/<split>/NJ_fp32_T<t>_{F<k>,<cov>}.
AJ = the all-judge stack (eval/scope69/base/<split>/alljudge); R59d-DRY = RESULTS 69 DRY_fp32 (eval/scope69/replay/<split>/DRY_fp32).
ALL F1 = mean final-round F1 over requests (score.py rules, RESULTS-63 truth). Paired bootstrap 2,000 over requests, seed 0.
Recall / precision vs AJ = AJ's final answer taken as the truth (score.py matching): POOLED = photos in both / photos AJ
(NJ) returns, request-level bootstrap; MEAN = per-request recall (precision) averaged over requests where AJ (NJ) returns
>= 1 photo. t = eval/nj71/threshold.txt (chosen on TRAIN by choose71.py before any dev number was read)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scope69"))
from common71 import CATS, ROOT, SPLITS, boot, boot_ratio, f1s, fmt, truth, vs_ref  # noqa: E402
from table69 import returned, stats  # noqa: E402

GRID = [f"{x:02d}" for x in range(5, 100, 5)]
KS = ["k16", "k24", "k32", "k48", "k64"]
COVS = ["0", "10", "25", "50"]
NAMES = {"a": "objects/scenes", "b": "attributes", "c": "person+look", "d": "time/place/event", "e": "multi-hop",
         "f": "vague 1 photo", "g": "two albums", "h": "follow-ups"}


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def main():
    T = (ROOT / "eval/nj71/threshold.txt").read_text().split()[0]
    NJ = f"NJ_fp32_T{T}"
    D = {"aj": "eval/scope69/base/{s}/alljudge", "r59d": "eval/scope69/replay/{s}/DRY_fp32",
         "ctl": "eval/nj71/replay/s1/{s}/DRY_fp32", "nj": f"eval/nj71/replay/s1/{{s}}/{NJ}",
         "nj70": "eval/nj71/replay/s1/{s}/NJ_fp32_T70"}
    for t in GRID:
        D[f"g{t}"] = f"eval/nj71/replay/s1/{{s}}/NJ_fp32_T{t}"
    for k in KS:
        D[k] = f"eval/nj71/replay/s2/{{s}}/{NJ}_F{k}"
    for c in COVS:
        D[f"c{c}"] = f"eval/nj71/replay/s2/{{s}}/{NJ}_{c}"
    F, ST, missing = {}, {}, []
    qs = {s: sorted(truth(s)) for s in SPLITS}
    for s in SPLITS:
        for n, d in D.items():
            d = d.format(s=s)
            if not (ROOT / d).exists():
                missing.append(d); continue
            F[(s, n)] = f1s(d, s)
            if sorted(F[(s, n)]) != qs[s]:
                missing.append(f"{d} ({len(F[(s, n)])}/{len(qs[s])})"); del F[(s, n)]; continue
            if n not in ("aj",):
                st = stats(d); ST[(s, n)] = st
                if n not in ("r59d", "ctl"):
                    assert all(v["judge"] == 0 and v.get("judgeA", 0) == 0 and v.get("misses", 0) == 0 for v in st.values()), f"{d}: NJ judged"
    have = lambda s, n: (s, n) in F

    def diff(s, a, b, cats=None):
        q_ = [q for q in qs[s] if cats is None or truth(s)[q]["cat"] in cats]
        return boot([F[(s, a)][q] - F[(s, b)][q] for q in q_])

    def mean(s, n, cats=None):
        return np.mean([F[(s, n)][q] for q in qs[s] if cats is None or truth(s)[q]["cat"] in cats])

    def calls(s, n):
        v = np.array([ST[(s, n)][q]["judge"] + ST[(s, n)][q].get("judgeA", 0) for q in qs[s]], float)
        return f"{v.mean():.0f} / {np.median(v):.0f} / {np.percentile(v, 90):.0f}"
    L = ["RESULTS 71: judge-free answers (NJ). ALL F1 final round, RESULTS-63 truth; diffs [95%, paired bootstrap 2,000, seed 0]"]
    if missing:
        L.append("MISSING / INCOMPLETE: " + "; ".join(missing))
    # control
    for s in SPLITS:
        a, b = returned(D["ctl"].format(s=s)), returned(D["r59d"].format(s=s))
        same = sum(a[q] == b.get(q) for q in a)
        L.append(f"CONTROL {s}: DRY_fp32 rerun returned sets identical to RESULTS 69 DRY_fp32 on {same}/{len(a)} requests "
                 f"({'OK' if same == len(a) else 'DIFFERS'}); F1 diff {fmt(diff(s, 'ctl', 'r59d'))}")
        d2 = ROOT / f"eval/nj71/replay/s2/{s}/{NJ}"
        if d2.exists():
            a2, b2 = returned(d2.relative_to(ROOT)), returned(D["nj"].format(s=s))
            L.append(f"  {s}: stage-2 in-job NJ rerun identical to stage 1 on {sum(a2[q] == b2.get(q) for q in a2)}/{len(a2)} requests")
        nj_calls = sum(v["judge"] + v.get("judgeA", 0) for n in D if n.startswith(("nj", "g", "k", "c")) and n != "ctl"
                       and (s, n) in ST for v in ST[(s, n)].values())
        L.append(f"  {s}: judge calls summed over every NJ config (both passes) = {nj_calls}; reference-photo calls "
                 f"{sum(v['route'].get('nj-ref', 0) for n in D if (s, n) in ST and n.startswith(('g', 'nj')) for v in ST[(s, n)].values())}")
    # threshold grid
    L += ["", f"THRESHOLD (chosen on TRAIN only, eval/nj71/grid_train.txt): t = .{T}",
          "grid t | TRAIN ALL F1 | dev ALL F1 (not used for the choice)"]
    for t in GRID:
        if have("train", f"g{t}"):
            L.append(f"  .{t} | {mean('train', f'g{t}'):.4f} | {mean('dev', f'g{t}'):.4f}" if have("dev", f"g{t}") else f"  .{t} | {mean('train', f'g{t}'):.4f} | -")
    # main table
    L += ["", "MAIN: config | split | ALL F1 | NJ - this [95%] | judge calls mean / median / p90"]
    for s in SPLITS:
        for n, lab in (("aj", "all-judge (AJ)"), ("r59d", "R59d-DRY (R69 DRY_fp32)"), ("nj", f"NJ t=.{T}"), ("nj70", "NJ t=.70 (identity; info)")):
            if not have(s, n):
                continue
            L.append(f"  {lab} | {s} | {mean(s, n):.4f} | {fmt(diff(s, 'nj', n)) if n not in ('nj',) else '-'} | "
                     f"{calls(s, n) if (s, n) in ST else 'every in-scope photo'}")
    # per category F1
    L += ["", "PER CATEGORY ALL F1: cat | split | n | AJ | R59d | NJ | NJ - AJ [95%] | NJ - R59d [95%]"]
    for c in CATS:
        for s in SPLITS:
            n_ = sum(truth(s)[q]["cat"] == c for q in qs[s])
            L.append(f"  {c} {NAMES[c]} | {s} | {n_} | {mean(s, 'aj', c):.3f} | {mean(s, 'r59d', c):.3f} | {mean(s, 'nj', c):.3f} | "
                     f"{fmt(diff(s, 'nj', 'aj', c))} | {fmt(diff(s, 'nj', 'r59d', c))}")

    def rp(s, n, cats):
        V = vs_ref(D[n].format(s=s), D["aj"].format(s=s), s)
        rows = [v for q, v in V.items() if v[0] in cats]
        tp, ns, na = (np.array([r[i] for r in rows], float) for i in (1, 2, 3))
        R = boot_ratio(tp, na); P = boot_ratio(tp, ns)
        mr = boot([t / a for t, a in zip(tp, na) if a > 0]) if (na > 0).any() else (np.nan,) * 3
        mp = boot([t / x for t, x in zip(tp, ns) if x > 0]) if (ns > 0).any() else (np.nan,) * 3
        return dict(n=len(rows), na=int((na > 0).sum()), ns=int((ns > 0).sum()), A=int(na.sum()), S=int(ns.sum()), TP=int(tp.sum()),
                    R=R, P=P, mR=mr, mP=mp)
    REC = {}
    for s in ("dev", "train"):
        L += ["", f"NJ (t=.{T}) vs AJ's answer, {s}: cat | requests (AJ non-empty / NJ non-empty) | photos AJ / NJ / both | "
                  "RECALL pooled [95%] | recall mean per request [95%] | PRECISION pooled [95%] | precision mean [95%]"]
        for c in CATS + ["ALL"]:
            x = rp(s, "nj", CATS if c == "ALL" else [c]); REC[(s, "nj", c)] = x
            L.append(f"  {c} | {x['n']} ({x['na']} / {x['ns']}) | {x['A']} / {x['S']} / {x['TP']} | {fmt(x['R'], False)} | "
                     f"{fmt(x['mR'], False)} | {fmt(x['P'], False)} | {fmt(x['mP'], False)}")
    # K table
    if all(have(s, k) for s in SPLITS for k in KS):
        L += ["", "WRITER CAP K under NJ: K | split | ALL F1 | vs AJ [95%] | vs uncapped NJ [95%]"]
        q2 = {}
        for k in KS:
            lows = []
            for s in SPLITS:
                t_ = diff(s, k, "nj"); lows.append(t_[1])
                L.append(f"  {k} | {s} | {mean(s, k):.4f} | {fmt(diff(s, k, 'aj'))} | {fmt(t_)}")
            q2[k] = min(lows) >= -.01
            L.append(f"     {k}: lower bounds vs uncapped NJ TRAIN {lows[0]:+.4f}, dev {lows[1]:+.4f} -> {'QUALIFIES' if q2[k] else 'no'} (>= -.01 both)")
        for s in SPLITS:
            L.append(f"  uncapped | {s} | {mean(s, 'nj'):.4f} | {fmt(diff(s, 'nj', 'aj'))} | 0")
        ok = [k for k in ("k16", "k24", "k32", "k48") if q2[k]]
        L.append(f"D2: smallest qualifying K in {{16, 24, 32, 48}} = {ok[0][1:] if ok else 'none'} -> night-pass K = "
                 f"{ok[0][1:] if ok else '48 (stays)'}")
    # coverage table
    if all(have(s, f"c{c}") for s in SPLITS for c in COVS):
        L += ["", "COVERAGE under NJ (newest first, policy V): cov | split | ALL F1 | vs AJ [95%] | vs NJ 100% [95%] | recall vs AJ pooled ALL | recall vs AJ mean per request ALL"]
        for c in COVS + ["100"]:
            n = "nj" if c == "100" else f"c{c}"
            for s in SPLITS:
                x = rp(s, n, CATS); REC[(s, n, "ALL")] = x
                L.append(f"  {c}% | {s} | {mean(s, n):.4f} | {fmt(diff(s, n, 'aj'))} | {fmt(diff(s, n, 'nj')) if n != 'nj' else '0'} | {fmt(x['R'], False)} | {fmt(x['mR'], False)}")
        for s in ("dev", "train"):
            L += ["", f"COVERAGE: NJ recall vs AJ per category, {s} (pooled [95%]; mean per request in parentheses)",
                  "  cov | " + " | ".join(CATS)]
            for c in COVS + ["100"]:
                n = "nj" if c == "100" else f"c{c}"
                cells = []
                for ct in CATS:
                    x = rp(s, n, [ct]); REC[(s, n, ct)] = x
                    cells.append(f"{x['R'][0]:.2f} [{x['R'][1]:.2f}, {x['R'][2]:.2f}] ({x['mR'][0]:.2f})")
                L.append(f"  {c}% | " + " | ".join(cells))
    # decisions
    L += ["", "D1 (wording, dev, NJ t=.%s): per category the fast-mode note states NJ's recall vs judging everything:" % T]
    for c in CATS:
        x = REC[("dev", "nj", c)]
        L.append(f"  {c} {NAMES[c]}: pooled {fmt(x['R'], False)} ({x['TP']}/{x['A']} photos, {x['na']} requests); "
                 f"mean per request {fmt(x['mR'], False)}")
    low = [c for c in CATS if REC[("dev", "nj", c)]["R"][0] < .70]
    lowm = [c for c in CATS if REC[("dev", "nj", c)]["mR"][0] < .70]
    L.append(f"D3 condition 1 (dev NJ recall vs AJ < .70): pooled -> {low or 'none'}; mean per request -> {lowm or 'none'}")
    L.append(f"D5 (ADDENDUM 2; 'verify the shown answer' mode offered for a category only if dev NJ precision vs AJ < .90):")
    for c in CATS:
        x, y = REC[("dev", "nj", c)], REC[("train", "nj", c)]
        L.append(f"  {c} {NAMES[c]}: dev precision pooled {fmt(x['P'], False)} ({x['TP']}/{x['S']} photos, {x['ns']} requests), "
                 f"mean per request {fmt(x['mP'], False)} -> {'OFFER' if x['P'][0] < .90 else 'not offered'} (pooled) / "
                 f"{'OFFER' if x['mP'][0] < .90 else 'not offered'} (mean); TRAIN pooled {fmt(y['P'], False)}, mean {fmt(y['mP'], False)}")
    txt = "\n".join(L)
    print(txt)
    if arg("out"):
        Path(arg("out")).write_text(txt + "\n")


if __name__ == "__main__":
    main()
