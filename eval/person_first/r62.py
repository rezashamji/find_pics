"""RESULTS 62: a FIRST ANSWER for person searches (findpics.person_first; FindPicsCore PersonFirst.swift).

Problem (10-14 07:00): a person album with a condition judged EVERY face-matched photo before showing anything (41,541
for the owner on Reza's phone, ~55 h at ~4.8 s a photo). Pre-registered (JOURNAL 10-14, before any number below):

FACT questions (engine.is_look_question false; the judge's yes alone, RESULTS 49c)
  rule   the person's photos ranked by the look vector (COS) or its learned-head blend (BLEND, RESULTS 48), then the
         phone's timed rounds exactly (tune_stop_time.sim_timed with fastmode_cut_replay.SHIPPED = phoneFastParams);
         first answer = where fast mode stops; the search keeps looking until everything is judged.
  sets   eval/vector_first_person (32 DISBench people, phone judge on every photo, red box), 7 facts:
         F1 each person alone (224 searches), F2 all 32 pooled as one 4,010-photo scope (7), F3 random pools of 8 and
         16 people (20 draws each, rng 62; 280). 20 tail-sample seeds (T.SEEDS). RANDOM order = ablation.
  report first-answer calls, judge-yes found / judge-yes in scope (judge-relative recall; precision vs judging
         everything is 1 by construction), calls to the first screen (min(20, yes)), minutes (m8 clock, 4.8 s/call),
         overclaims in EVERY round of every seed (exhaustive replay), eye precision / recall estimated with the RESULTS
         49c strata rates (blind labels, unsure = no match).
  D1     ship only if 0 overclaims over every round of every seed on F1+F2+F3.
  D2     order = BLEND iff its first-answer judge-yes found, summed over F2+F3 searches (mean over seeds), >= COS's;
         else COS.

LOOK questions (relative to the person's own photos)
  rule   person_first.walk_order (fnv1a64 random order, <= 3 judged photos per event first, events = 3 h gaps),
         pairs: both questions per photo (2 calls); pair_partial / single_partial (event median of the judged photos,
         split fitted on judged photos weighted by the photos they stand for); unsorted = events not reached.
  sets   L1 Reza's 393-photo labelled sample, replayed from data/private/sample_runs judge caches (q3vl4b_real = phone
         judge PRIMARY, 9b_prob secondary, all 33 runs at 480 calls), numbers only, data/private/audits/r62_*.txt;
         L2 IMDB-WIKI eras (Pratt / Hill / Rogen, 239 face-filtered crops, cached 9B P, no dates = each photo its own
         event); L3 the 32 public people, single look "tired" (phone judge, DISBench dates).
  budgets 200 / 480 / 1,000 judge calls + the curve every 20 calls; 20 walk seeds.
  report vs the judge-everything albums: precision, recall, wrong-album photos, share of photos sorted, agreement
         (same label incl. "neither"/unsorted); vs labels (era): H found / F leak in heavier, F found / H leak in fit.
  D3     ship the walk (cap 3, events propagated) iff at 480 calls on L1 primary (pair), mean over seeds, BOTH albums
         have precision vs the judge-everything albums >= 0.95. Else: judged-only (no propagation) if it passes; else
         say so. Ablations (cap 1, no cap, judged-only) are reported, not decided on.
  check  with every photo judged the partial rules equal the full rules (0 photo differences).

COVERAGE (label-free): share of a person's / library's photos in events reached after m walked photos, DISBench.

  python eval/person_first/r62.py texts      (Slurm GPU/CPU job: PE-Core-B-16 text vectors of the plan questions)
  python eval/person_first/r62.py facts      -> eval/person_first/facts.txt
  python eval/person_first/r62.py looks      -> eval/person_first/looks_public.txt (+ curves_public.tsv)
  python eval/person_first/r62.py private    -> data/private/audits/r62_looks_private.txt (numbers only; never committed)
  python eval/person_first/r62.py coverage   -> eval/person_first/coverage.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "eval/learned_head")); sys.path.insert(0, str(ROOT / "src"))
import tune_stop_rule as T            # noqa: E402
import tune_stop_time as TT           # noqa: E402
from findpics import person_first as PF   # noqa: E402

OUT = ROOT / "eval/person_first"
VFP = ROOT / "eval/vector_first_person"
HEAD = ROOT / "ios/FindPicsApp.swiftpm/Sources/Models/learned_head.fphead"
SHIPPED = "timed round_calls=120 inter=0 topup=30 head=150 a1=0.2 fast_stop=480 stop_topup=150"   # = phoneFastParams
ACC = 0.7
FACTS = ["beach", "eating", "hat", "outdoors", "smiling", "snow", "sunglasses"]
SEEDS = list(range(T.SEEDS))
BUDGETS = (200, 480, 1000)
SEC_PER_CALL = 4.8          # the person path as measured on the phone 10-14 (judge 2.4 s + full-res read 2.4 s)
POOL_SIZES = (8, 16)
POOL_DRAWS = 20


def _arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


# ------------------------------------------------------------------------------------------------------------ texts
def plan_questions():
    from eval_vector_first_person import COND
    return {c: q.replace("the person in the red box", "the person") for c, (_, q, _) in COND.items()}


def cmd_texts():
    from findpics.models import ImageTextEncoder
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    Q = plan_questions()
    V = enc.texts(list(Q.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "plan_q_texts.npz", **{c: v for c, v in zip(Q, V)})
    print("TEXTS", {c: q for c, q in Q.items()})


# ------------------------------------------------------------------------------------------------------------ data
def people():
    """All 32 people x 8 conditions with s (look cosine), h (head logit for the plan question), y, events per person,
    and the 49c stratum of each fact row."""
    import eval_vector_first_person as VF
    import person_rule_49c as R
    from head_ship import HeadFile
    from findpics.agent import events
    D = VF.load()
    ids, V = [], []
    for f in sorted(VFP.glob("vec*.npz")):
        z = np.load(f); ids += list(z["ids"]); V.append(z["V"].astype(np.float32))
    V = np.concatenate(V); row = {i: r for r, i in enumerate(ids)}
    X = V[[row[i] for i in D.item_id]].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    H = HeadFile(HEAD); TX = np.load(OUT / "plan_q_texts.npz")
    D["h"] = np.nan
    for c in D.cond.unique():
        m = (D.cond == c).to_numpy()
        D.loc[m, "h"] = H.logits(X[m], H.fold(TX[c]))
    it = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet", columns=["item_id", "taken"])
    taken = dict(zip(it.item_id.astype(str), it.taken))
    D["taken"] = D.item_id.map(taken)
    D["event"] = ""
    for per, g in D[D.cond == D.cond.iloc[0]].groupby("person"):
        ev = events(g.taken.reset_index(drop=True))
        emap = dict(zip(g.item_id, [f"{per}:{e}" for e in ev]))
        m = D.person == per
        D.loc[m, "event"] = D.loc[m, "item_id"].map(emap)
    S = R.strata()[["person", "cond", "item_id", "st"]]
    D = D.merge(S, on=["person", "cond", "item_id"], how="left")
    return D


def strata_rates():
    import person_rule_49c as R
    key, lab = R.labels()
    rates = {}
    for c in FACTS:
        for h in R.STRATA:
            v = [lab[(k["cond"], k["person"], k["item_id"])] for k in key
                 if k["cond"] == c and k["stratum"] == h and (k["cond"], k["person"], k["item_id"]) in lab]
            if v:
                rates[(c, h)] = float(np.mean([x == "match" for x in v]))
    return rates


# ------------------------------------------------------------------------------------------------------------ FACT
def fact_searches(D):
    rng = np.random.default_rng(62)
    persons = sorted(D.person.unique())
    S = []
    for c in FACTS:
        g = D[D.cond == c]
        for p in persons:
            S.append(("F1", f"{p}", c, g[g.person == p]))
        S.append(("F2", "all32", c, g))
    for size in POOL_SIZES:
        for d in range(POOL_DRAWS):
            ps = set(rng.choice(persons, size=size, replace=False))
            for c in FACTS:
                g = D[(D.cond == c) & D.person.isin(ps)]
                S.append((f"F3_{size}", f"draw{d}", c, g))
    return S


def run_fact(g, order_kind, seed_r=0):
    _, r = TT.parse(SHIPPED)
    s = g.s.to_numpy(); h = g.h.to_numpy()
    if order_kind == "COS":
        o = PF.fact_order(s)
    elif order_kind == "BLEND":
        o = PF.fact_order(s, h)
    else:
        o = np.random.default_rng(1000 + seed_r).permutation(len(g))
    y = (g.p.to_numpy() >= ACC)[o]
    st = g.st.to_numpy()[o]
    Y = int(y.sum())
    out = []
    for sd in SEEDS:
        calls, rounds = TT.sim_timed(y, r, sd)
        _, rx = TT.sim_timed(y, r, sd, exhaustive=True)
        n_over = sum(1 for (nc, cert, k, last) in rx if (Y - cert["found"]) > cert["missed_upper"] + 1e-9)
        nc, cert, k, last = rounds[-1]
        judged = calls[:nc]
        found = judged[y[judged]]
        cum = np.cumsum(y[calls]); need = min(20, Y)
        first_screen = int(np.searchsorted(cum, need) + 1) if need else 0
        out.append(dict(seed=sd, n=len(y), Y=Y, calls=int(nc), found=int(len(found)), rounds=len(rx), over=n_over,
                        rounds_first=k, first_screen=first_screen, missed_upper=float(cert["missed_upper"]),
                        st_found=list(st[found])))
    return out


def eye_est(rows, rates, cond):
    """Estimated eye-right photos among rows (their 49c strata), unknown strata = no information (counted 0)."""
    return float(sum(rates.get((cond, h), 0.0) for h in rows))


def cmd_facts():
    D = people()
    rates = strata_rates()
    S = fact_searches(D)
    res = []
    for setname, pool, c, g in S:
        g = g.reset_index(drop=True)
        tot_eye = eye_est(g.st, rates, c)
        yes_eye = eye_est(g.st[g.p >= ACC], rates, c)
        for ok in ("COS", "BLEND", "RANDOM"):
            for o in run_fact(g, ok):
                res.append(dict(set=setname, pool=pool, cond=c, order=ok, tot_eye=tot_eye, yes_eye=yes_eye,
                                eye_found=eye_est(o.pop("st_found"), rates, c), **o))
    R = pd.DataFrame(res)
    R.to_parquet(OUT / "facts_runs.parquet")
    L = ["RESULTS 62 (A): FACT person searches, first answer = the phone's timed fast mode inside the person's photos.",
         f"Rule {SHIPPED} (phoneFastParams). Judge = Qwen3-VL-4B phone weights, red box, yes at P >= {ACC}.",
         f"{len(SEEDS)} tail-sample seeds per search; means over seeds unless stated. Minutes: m8 = the 8-bit tower's measured",
         f"clock (RESULTS 44), 4.8 s = {SEC_PER_CALL} s per call (the person path as measured 10-14).", ""]
    L.append("| set | order | searches | photos (median) | judge-yes in scope | first answer: calls (median / max) | found "
             "(judge-rel. recall) | first screen calls (median) | min m8 / 4.8 s (median) | overclaims / rounds |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for setname in ["F1", "F2", "F3_8", "F3_16"]:
        for ok in ("COS", "BLEND", "RANDOM"):
            x = R[(R.set == setname) & (R.order == ok)]
            per = x.groupby(["pool", "cond"]).agg(n=("n", "first"), Y=("Y", "first"), calls=("calls", "mean"),
                                                  found=("found", "mean"), fs=("first_screen", "median"))
            m8 = np.median([TT.minutes_of(cc, "m8") for cc in x.calls])
            mc = np.median(x.calls) * SEC_PER_CALL / 60
            L.append(f"| {setname} | {ok} | {len(per)} | {int(per.n.median())} | {int(per.Y.sum()):,} | "
                     f"{int(np.median(x.calls))} / {int(x.calls.max())} | {per.found.sum():,.0f} "
                     f"({per.found.sum() / max(per.Y.sum(), 1):.3f}) | {int(np.median(x.first_screen))} | "
                     f"{m8:.1f} / {mc:.1f} | {int(x.over.sum())} / {int(x.rounds.sum()):,} |")
    L.append("")
    L.append("Per condition, F2 (all 32 people as one 4,010-photo scope; judge everything = 4,010 calls = "
             f"{4010 * SEC_PER_CALL / 3600:.1f} h at 4.8 s):")
    L.append("| condition | judge-yes | COS: calls, found (recall) | BLEND: calls, found (recall) | RANDOM: calls, found |")
    L.append("|---|---|---|---|---|")
    for c in FACTS:
        x = R[(R.set == "F2") & (R.cond == c)]
        cells = []
        for ok in ("COS", "BLEND", "RANDOM"):
            z = x[x.order == ok]
            cells.append(f"{z.calls.mean():.0f}, {z.found.mean():.0f} ({z.found.mean() / max(z.Y.iloc[0], 1):.3f})")
        L.append(f"| {c} | {int(x.Y.iloc[0])} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("EYE (RESULTS 49c strata rates; blind labels, unsure = no match), pooled over F2's 7 searches, means over seeds:")
    for ok in ("COS", "BLEND"):
        x = R[(R.set == "F2") & (R.order == ok)]
        g = x.groupby("cond").agg(found=("found", "mean"), eye=("eye_found", "mean"), tot=("tot_eye", "first"),
                                  yes_eye=("yes_eye", "first"), Y=("Y", "first"))
        L.append(f"  {ok}: first answer {g.found.sum():.0f} photos, est. {g.eye.sum():.0f} right by eye (precision "
                 f"{g.eye.sum() / g.found.sum():.3f}, eye recall {g.eye.sum() / g.tot.sum():.3f}); judge everything: "
                 f"{g.Y.sum():.0f} photos, est. {g.yes_eye.sum():.0f} right (precision {g.yes_eye.sum() / g.Y.sum():.3f}, "
                 f"eye recall {g.yes_eye.sum() / g.tot.sum():.3f})")
    L.append("")
    a = R[R.set.isin(["F2", "F3_8", "F3_16"])].groupby(["set", "pool", "cond", "order"]).found.mean().unstack()
    d2 = "BLEND" if a.BLEND.sum() >= a.COS.sum() else "COS"
    paired = (a.BLEND - a.COS)
    L.append(f"D1: overclaims over every round of every seed, F1+F2+F3, COS+BLEND: {int(R[R.order != 'RANDOM'].over.sum())} "
             f"of {int(R[R.order != 'RANDOM'].rounds.sum()):,} rounds -> {'PASS' if R[R.order != 'RANDOM'].over.sum() == 0 else 'FAIL'}")
    L.append(f"D2: first-answer judge-yes found summed over F2+F3 ({len(a)} searches): BLEND {a.BLEND.sum():,.0f} vs COS "
             f"{a.COS.sum():,.0f} (paired per search: BLEND better {int((paired > 0).sum())}, worse {int((paired < 0).sum())}, "
             f"same {int((paired == 0).sum())}) -> order = {d2}")
    txt = "\n".join(L)
    (OUT / "facts.txt").write_text(txt + "\n")
    print(txt)


# ------------------------------------------------------------------------------------------------------------ LOOK
VARIANTS = {"walk3": dict(cap=3, propagate=True), "walk1": dict(cap=1, propagate=True),
            "uniform": dict(cap=None, propagate=True), "judged_only": dict(cap=3, propagate=False)}


def full_pair(ids, event_of, pa, pb):
    return PF.pair_albums(ids, event_of, pa, pb)


def partial_pair(ids, event_of, pa, pb, propagate):
    return PF.pair_albums(ids, event_of, pa, pb, propagate=propagate)


def pair_curve(ids, event_of, pa_all, pb_all, lab, variant, seed, marks):
    """Per call budget in marks: per photo label (0 / 1 / None / 'u' = unsorted) of the first answer."""
    v = VARIANTS[variant]
    pos = PF.walk_order(ids, event_of, seed, cap=v["cap"])
    out = {}
    for m in marks:
        k = min(len(ids), m // 2)
        done = [ids[q] for q in pos[:k]]
        pa = {i: pa_all[i] for i in done}; pb = {i: pb_all[i] for i in done}
        a = partial_pair(ids, event_of, pa, pb, v["propagate"]) if k >= 1 else {}
        out[m] = {i: a.get(i, "u") if i in a else "u" for i in ids}
    return out


def single_curve(ids, event_of, p_all, variant, seed, marks):
    v = VARIANTS[variant]
    pos = PF.walk_order(ids, event_of, seed, cap=v["cap"])
    out = {}
    for m in marks:
        done = [ids[q] for q in pos[:min(len(ids), m)]]
        keep = PF.single_partial(ids, event_of, {i: p_all[i] for i in done}, propagate=v["propagate"])
        out[m] = {i: (keep[i] if i in keep else "u") for i in ids}
    return out


def score_pair(first, full, lab=None):
    """first/full: {id: 0|1|None|'u'}. Precision/recall of each album vs the full albums, wrong-album photos, share
    sorted, agreement; vs labels: (found, leak) per album."""
    ids = list(full)
    r = {}
    for a, name in ((0, "A"), (1, "B")):
        fa = {i for i in ids if first[i] == a}; ga = {i for i in ids if full[i] == a}
        r[f"{name}_n"] = len(fa); r[f"{name}_full"] = len(ga)
        r[f"{name}_prec"] = len(fa & ga) / len(fa) if fa else np.nan
        r[f"{name}_rec"] = len(fa & ga) / len(ga) if ga else np.nan
        r[f"{name}_wrong"] = len({i for i in fa if full[i] == 1 - a})
        if lab is not None:
            want, other = ("H", "F") if a == 0 else ("F", "H")
            r[f"{name}_lab_found"] = sum(lab.get(i) == want for i in fa); r[f"{name}_lab_leak"] = sum(lab.get(i) == other for i in fa)
    r["sorted"] = np.mean([first[i] != "u" for i in ids])
    r["agree"] = np.mean([(first[i] if first[i] != "u" else None) == full[i] for i in ids])
    return r


def score_single(first, full, lab=None, want="H"):
    ids = list(full)
    fa = {i for i in ids if first[i] is True}; ga = {i for i in ids if full[i] is True}
    r = dict(n=len(fa), full=len(ga), prec=len(fa & ga) / len(fa) if fa else np.nan,
             rec=len(fa & ga) / len(ga) if ga else np.nan, sorted=np.mean([first[i] != "u" for i in ids]),
             agree=np.mean([(first[i] is True) == (full[i] is True) for i in ids]))
    if lab is not None:
        r["lab_found"] = sum(lab.get(i) == want for i in fa); r["lab_leak"] = sum(lab.get(i) not in (want, None) for i in fa)
    return r


def marks_for(total_calls):
    return sorted(set(list(range(20, total_calls + 20, 20)) + [b for b in BUDGETS] + [total_calls]))


def _pair_table(name, sets, lab_of, total_note):
    """sets: list of (label, ids, event_of, pa, pb). Returns lines + curve rows."""
    L, curves = [], []
    for variant in VARIANTS:
        for label, ids, event_of, pa, pb in sets:
            full = full_pair(ids, event_of, pa, pb)
            tot = 2 * len(ids)
            marks = marks_for(tot)
            lab = lab_of(label) if lab_of else None
            S = {m: [] for m in marks}
            for sd in SEEDS:
                cur = pair_curve(ids, event_of, pa, pb, lab, variant, sd, marks)
                for m in marks:
                    S[m].append(score_pair(cur[m], full, lab))
            for m in marks:
                d = pd.DataFrame(S[m]).mean(numeric_only=True).to_dict()
                curves.append(dict(set=name, person=label, variant=variant, calls=m, total=tot, **d))
    return L, pd.DataFrame(curves)


def imdb_sets():
    Tdir = ROOT / "data/public/transformation"
    s = pd.read_csv(Tdir / "scored_fit.csv", index_col=0)
    fs = pd.read_csv(Tdir / "face_sims.csv").set_index("key")
    s["sim"] = fs.loc[s.index, "sim_expanded"].to_numpy()
    s = s[s.sim >= 0.40].copy(); s["item_id"] = s.index.astype(str)
    out = []
    for name, g in s.groupby("name"):
        ids = list(g.item_id)
        out.append((name, ids, None, dict(zip(ids, g.p_heavier)), dict(zip(ids, g.p_fit)),
                    {i: ("H" if e == "heavy" else "F") for i, e in zip(ids, g.era)}))
    return out


def tired_sets(D):
    out = []
    for per, g in D[D.cond == "tired"].groupby("person"):
        ids = list(g.item_id)
        out.append((per, ids, dict(zip(ids, g.event)), dict(zip(ids, g.p))))
    return out


def summarize_curves(C, lines, title, budgets=BUDGETS, fracs=(0.1, 0.25, 0.5, 1.0), lab=True, pair=True):
    lines.append(title)
    if pair:
        hdr = ("| variant | calls | sorted | agreement | heavier: prec / rec vs full, wrong | fit: prec / rec vs full, wrong |"
               + (" heavier H found + F leak / fit F found + H leak (labels) |" if lab else ""))
    else:
        hdr = "| variant | calls | sorted | agreement | album: prec / rec vs full |" + (" H found + other leak |" if lab else "")
    lines.append(hdr); lines.append("|" + "---|" * (hdr.count("|") - 1))
    for variant in VARIANTS:
        x = C[C.variant == variant]
        rows = [("abs", b) for b in budgets] + [("frac", f) for f in fracs]
        for kind, b in rows:
            if kind == "abs":
                z = x[x.calls == b]
                if z.empty:
                    z = x.sort_values("calls").groupby("person").tail(1)
                    if (z.total < b).all() and b != budgets[0]:
                        pass
                tag = f"{b}"
            else:
                z = x.assign(d=(x.calls - b * x.total).abs()).sort_values("d").groupby("person").head(1)
                tag = f"{int(b * 100)}% of all"
            if z.empty:
                continue
            if pair:
                cell = (f"| {variant} | {tag} | {z.sorted.mean():.3f} | {z.agree.mean():.3f} | "
                        f"{z.A_prec.mean():.3f} / {z.A_rec.mean():.3f}, {z.A_wrong.sum():.1f} | "
                        f"{z.B_prec.mean():.3f} / {z.B_rec.mean():.3f}, {z.B_wrong.sum():.1f} |")
                if lab:
                    cell += (f" {z.A_lab_found.sum():.1f} + {z.A_lab_leak.sum():.1f} / {z.B_lab_found.sum():.1f} + "
                             f"{z.B_lab_leak.sum():.1f} |")
            else:
                cell = f"| {variant} | {tag} | {z.sorted.mean():.3f} | {z.agree.mean():.3f} | {z.prec.mean():.3f} / {z.rec.mean():.3f} |"
            lines.append(cell)
    lines.append("")


def cmd_looks():
    lines = ["RESULTS 62 (B): LOOK person searches, public sets. Means over 20 walk seeds; 'sorted' = share of the person's",
             "photos in the first answer's albums or 'neither' (the rest: events not reached yet); agreement = same label as",
             "the judge-everything albums (unsorted counts as neither); prec / rec = the first answer's album vs the same album",
             "from judging everything; wrong = photos in the OTHER album than the full answer puts them (sum over people).", ""]
    # L2 IMDB pair + singles
    sets = imdb_sets()
    curves = []
    for variant in VARIANTS:
        for name, ids, ev, pa, pb, lab in sets:
            full = full_pair(ids, ev, pa, pb)
            tot = 2 * len(ids); marks = marks_for(tot)
            S = {m: [] for m in marks}
            for sd in SEEDS:
                cur = pair_curve(ids, ev, pa, pb, lab, variant, sd, marks)
                for m in marks:
                    S[m].append(score_pair(cur[m], full, lab))
            for m in marks:
                curves.append(dict(set="imdb_pair", person=name, variant=variant, calls=m, total=tot,
                                   **pd.DataFrame(S[m]).mean(numeric_only=True).to_dict()))
            for alb, pp, want in (("heavier", pa, "H"), ("fit", pb, "F")):
                fullk = PF.single_partial(ids, ev, pp)
                tot1 = len(ids); marks1 = marks_for(tot1)
                S1 = {m: [] for m in marks1}
                for sd in SEEDS:
                    cur = single_curve(ids, ev, pp, variant, sd, marks1)
                    for m in marks1:
                        S1[m].append(score_single(cur[m], fullk, lab, want))
                for m in marks1:
                    curves.append(dict(set=f"imdb_{alb}", person=name, variant=variant, calls=m, total=tot1,
                                       **pd.DataFrame(S1[m]).mean(numeric_only=True).to_dict()))
    C = pd.DataFrame(curves)
    # exactness at completion
    exact = []
    for name, ids, ev, pa, pb, lab in sets:
        from findpics import engine as E
        from types import SimpleNamespace as NS
        full = full_pair(ids, ev, pa, pb)
        sp = E._split_pair([pd.DataFrame(dict(item_id=ids, p_attr=[pa[i] for i in ids])),
                            pd.DataFrame(dict(item_id=ids, p_attr=[pb[i] for i in ids]))], None)
        ref = {i: (v[0] if v else None) for i, v in sp.items()} if sp is not None else PF.rank_margin_pair(pa, pb)
        exact.append(sum(full[i] != ref[i] for i in ids))
    lines.append(f"Check: partial rule with every photo judged vs engine._split_pair / rank margin (IMDB, 3 people): "
                 f"{sum(exact)} photo differences of {sum(len(s[1]) for s in sets)}.")
    lines.append("")
    summarize_curves(C[C.set == "imdb_pair"], lines,
                     "L2 IMDB-WIKI eras, PAIR heavier vs fit (cached 9B; Pratt 99 / Hill 74 / Rogen 66 photos = 198 / 148 / 132 "
                     "calls for everything; no dates, so each photo is its own event and propagation does nothing). Budgets "
                     "above a person's total = that person complete.", budgets=(50, 100, 200))
    for alb in ("heavier", "fit"):
        summarize_curves(C[C.set == f"imdb_{alb}"], lines, f"L2 IMDB single '{alb}' album (1 call per photo):",
                         budgets=(25, 50, 100), pair=False)
    # L3 tired
    D = people()
    curves3 = []
    for variant in VARIANTS:
        for per, ids, ev, p in tired_sets(D):
            fullk = PF.single_partial(ids, ev, p)
            tot1 = len(ids); marks1 = marks_for(tot1)
            S1 = {m: [] for m in marks1}
            for sd in SEEDS:
                cur = single_curve(ids, ev, p, variant, sd, marks1)
                for m in marks1:
                    S1[m].append(score_single(cur[m], fullk))
            for m in marks1:
                curves3.append(dict(set="tired", person=per, variant=variant, calls=m, total=tot1,
                                    **pd.DataFrame(S1[m]).mean(numeric_only=True).to_dict()))
    C3 = pd.DataFrame(curves3)
    ev_stats = D[D.cond == "tired"].groupby("person").agg(n=("item_id", "size"), ev=("event", "nunique"))
    summarize_curves(C3, lines, f"L3 public 'tired' (single look, phone judge; 32 people, {ev_stats.n.sum():,} photos in "
                     f"{ev_stats.ev.sum():,} events; judge-relative only: eye found 0/16 judge-yes clearly tired, M45):",
                     budgets=(20, 50, 100), pair=False, lab=False)
    pd.concat([C, C3]).to_csv(OUT / "curves_public.tsv", sep="\t", index=False)
    txt = "\n".join(lines)
    (OUT / "looks_public.txt").write_text(txt + "\n")
    print(txt)


def cmd_private():
    """L1: Reza's labelled sample, replayed from the saved runs' pools (judged.parquet p_attr = the judge cache's P,
    RESULTS 39). Numbers only, to data/private/audits/r62_looks_private.txt."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import replay_pairing as RP           # noqa: E402
    from findpics.agent import events
    lab0 = json.loads(RP.LABELS.read_text())
    items = pd.read_parquet(RP.INDEX / "items.parquet")
    dated = pd.to_datetime(items["taken"], utc=True, errors="coerce", format="ISO8601").notna().to_numpy()
    ev_all = {str(i): int(x) for i, x, ok in zip(items["item_id"], events(items["taken"]), dated) if ok}
    out_lines, curves, run_rows = [], [], []
    runs = RP.runs()
    primary = {"mode_q3vl4b_real", "mode_9b_prob"}
    for run in runs:
        if run.name.endswith("_bi"):
            continue                       # bipolar runs score one combined question; not this rule's input
        pl = RP.pools_of(run)
        A_, B_ = [j[j["where"] == "identity_match"] for _, j in pl]
        pa = dict(zip(A_.item_id.astype(str), A_.p_attr)); pb = dict(zip(B_.item_id.astype(str), B_.p_attr))
        ids = sorted(i for i in pa if i in pb)
        name_of = dict(zip(A_.item_id.astype(str), [Path(p).name for p in A_.path]))
        lab = {i: lab0.get(name_of[i]) for i in ids}
        ev = {i: ev_all[i] for i in ids if i in ev_all}
        full = full_pair(ids, ev, pa, pb)
        variants = list(VARIANTS) if run.name in primary else ["walk3"]
        tot = 2 * len(ids)
        marks = marks_for(tot) if run.name in primary else [480]
        for variant in variants:
            S = {m: [] for m in marks}
            for sd in SEEDS:
                cur = pair_curve(ids, ev, pa, pb, lab, variant, sd, marks)
                for m in marks:
                    S[m].append(score_pair(cur[m], full, lab))
            for m in marks:
                curves.append(dict(run=run.name, variant=variant, calls=m, total=tot,
                                   **pd.DataFrame(S[m]).mean(numeric_only=True).to_dict(),
                                   A_prec_min=pd.DataFrame(S[m]).A_prec.min(), B_prec_min=pd.DataFrame(S[m]).B_prec.min()))
        if run.name in primary:
            for alb, pp, want in (("heavier", pa, "H"), ("fit", pb, "F")):
                fullk = PF.single_partial(ids, ev, pp)
                for variant in VARIANTS:
                    marks1 = marks_for(len(ids))
                    S1 = {m: [] for m in marks1}
                    for sd in SEEDS:
                        cur = single_curve(ids, ev, pp, variant, sd, marks1)
                        for m in marks1:
                            S1[m].append(score_single(cur[m], fullk, lab, want))
                    for m in marks1:
                        curves.append(dict(run=f"{run.name}:{alb}", variant=variant, calls=m, total=len(ids),
                                           **pd.DataFrame(S1[m]).mean(numeric_only=True).to_dict()))
        # exactness at completion vs the engine's split
        from findpics import engine as E
        sp = E._split_pair([pd.DataFrame(dict(item_id=list(pa), p_attr=list(pa.values()))),
                            pd.DataFrame(dict(item_id=list(pb), p_attr=list(pb.values())))], ev)
        ref = {i: (v[0] if v else None) for i, v in sp.items()} if sp is not None else PF.rank_margin_pair(pa, pb)
        fullcnt = {k: sum(full[i] == k for i in ids) for k in (0, 1, None)}
        run_rows.append(dict(run=run.name, n=len(ids), events=len(set(ev.values())) + sum(i not in ev for i in ids),
                             diff=sum(full[i] != ref.get(i) for i in ids), split=sp is not None,
                             full_H=sum(full[i] == 0 and lab[i] == "H" for i in ids), full_HF=sum(full[i] == 0 and lab[i] == "F" for i in ids),
                             full_F=sum(full[i] == 1 and lab[i] == "F" for i in ids), full_FH=sum(full[i] == 1 and lab[i] == "H" for i in ids),
                             full_neither=fullcnt[None]))
    C = pd.DataFrame(curves); Rr = pd.DataFrame(run_rows)
    priv = ROOT / "data/private/audits"
    C.to_csv(priv / "r62_curves_private.tsv", sep="\t", index=False)
    L = ["RESULTS 62 L1 (PRIVATE, numbers only): Reza's labelled sample, pair 'me looking heavier' vs 'me looking fit',",
         "replayed from the saved runs' judged pools. Labels: 267 H / 124 F (+1 not him, 1 neutral). 20 walk seeds.", ""]
    L.append("Check (every photo judged): partial rule vs engine._split_pair, photo differences per run: "
             + ", ".join(f"{r.run[5:]} {r.diff}" for r in Rr.itertuples()))
    L.append(f"Events among the shared photos (primary run): " + ", ".join(f"{r.run[5:]} {r.events}" for r in Rr.itertuples() if r.run in primary))
    L.append("Full albums (judge everything): " + "; ".join(f"{r.run[5:]} {r.full_H}+{r.full_HF} / {r.full_F}+{r.full_FH} / {r.full_neither}"
                                                        for r in Rr.itertuples() if r.run in primary))
    L.append("")
    for run in sorted(primary):
        x = C[C.run == run]
        L.append(f"{run} PAIR (total {int(x.total.iloc[0])} calls = every photo, both questions):")
        L.append("| variant | calls | sorted | agreement | heavier prec / rec vs full (min prec), wrong | fit prec / rec vs full (min prec), wrong | heavier H + F leak / fit F + H leak (labels) |")
        L.append("|---|---|---|---|---|---|---|")
        for variant in VARIANTS:
            for b in list(BUDGETS) + [int(x.total.iloc[0])]:
                z = x[(x.variant == variant) & (x.calls == b)]
                if z.empty:
                    continue
                z = z.iloc[0]
                L.append(f"| {variant} | {b} | {z.sorted:.3f} | {z.agree:.3f} | {z.A_prec:.3f} / {z.A_rec:.3f} ({z.A_prec_min:.3f}), {z.A_wrong:.1f} | "
                         f"{z.B_prec:.3f} / {z.B_rec:.3f} ({z.B_prec_min:.3f}), {z.B_wrong:.1f} | {z.A_lab_found:.1f} + {z.A_lab_leak:.1f} / "
                         f"{z.B_lab_found:.1f} + {z.B_lab_leak:.1f} |")
        L.append("")
        L.append(f"{run} PAIR curve (walk3): calls -> sorted, agreement, heavier prec/rec, fit prec/rec")
        x3 = x[x.variant == "walk3"].sort_values("calls")
        L.append("  " + "; ".join(f"{int(z.calls)}: {z.sorted:.2f} {z.agree:.2f} {z.A_prec:.2f}/{z.A_rec:.2f} {z.B_prec:.2f}/{z.B_rec:.2f}"
                                  for z in x3.itertuples() if z.calls % 80 == 0 or z.calls in BUDGETS))
        L.append("")
        for alb in ("heavier", "fit"):
            xs = C[C.run == f"{run}:{alb}"]
            L.append(f"{run} SINGLE '{alb}' (1 call per photo): variant, calls -> sorted, agreement, prec / rec vs full, label found + leak")
            for variant in VARIANTS:
                L.append("  " + variant + ": " + "; ".join(
                    f"{b}: {z.sorted:.2f} {z.agree:.3f} {z.prec:.3f}/{z.rec:.3f} {z.lab_found:.1f}+{z.lab_leak:.1f}"
                    for b in (100, 200, 300, len(ids)) for z in xs[(xs.variant == variant) & (xs.calls == b)].itertuples()))
            L.append("")
    x = C[(C.calls == 480) & (C.variant == "walk3") & C.run.str.startswith("mode_") & ~C.run.str.contains(":")]
    L.append(f"ALL {len(x)} runs (bipolar runs excluded), walk3 at 480 calls, means over seeds: heavier prec vs full median "
             f"{x.A_prec.median():.3f} (min {x.A_prec.min():.3f}), fit prec median {x.B_prec.median():.3f} (min {x.B_prec.min():.3f}); "
             f"agreement median {x.agree.median():.3f} (min {x.agree.min():.3f}); sorted median {x.sorted.median():.3f}")
    z = C[(C.run == "mode_q3vl4b_real") & (C.variant == "walk3") & (C.calls == 480)].iloc[0]
    d3 = z.A_prec >= 0.95 and z.B_prec >= 0.95
    L.append(f"D3 (q3vl4b_real, walk3, 480 calls): heavier prec {z.A_prec:.3f}, fit prec {z.B_prec:.3f} -> {'PASS' if d3 else 'FAIL'}")
    if not d3:
        zj = C[(C.run == "mode_q3vl4b_real") & (C.variant == "judged_only") & (C.calls == 480)].iloc[0]
        L.append(f"   judged_only: heavier prec {zj.A_prec:.3f}, fit prec {zj.B_prec:.3f} -> "
                 f"{'PASS' if zj.A_prec >= 0.95 and zj.B_prec >= 0.95 else 'FAIL'}")
    txt = "\n".join(L)
    (priv / "r62_looks_private.txt").write_text(txt + "\n")
    print(txt)


# ------------------------------------------------------------------------------------------------------------ coverage
def coverage_curve(ids, event_of, seed, marks, cap=PF.LOOK_CAP):
    pos = PF.walk_order(ids, event_of, seed, cap=cap)
    size = {}
    for i in ids:
        size[event_of[i]] = size.get(event_of[i], 0) + 1
    out = {}
    for m in marks:
        evs = {event_of[ids[q]] for q in pos[:m]}
        out[m] = sum(size[e] for e in evs) / len(ids)
    return out


def cmd_coverage():
    from findpics.agent import events
    it = pd.read_parquet(ROOT / "data/public/index_disbench/items.parquet", columns=["item_id", "path", "taken"])
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    marks = (50, 100, 240, 480, 1000)
    L = ["RESULTS 62 COVERAGE (label-free): share of photos in events with >= 1 judged photo after m photos of the walk",
         "(cap 3; pair = 2 calls per photo, so a pair's 480-call first answer = m 240). Events = 3 h gaps. 5 walk seeds.", ""]
    rows = []
    for u, g in it.groupby("user"):
        ids = list(g.item_id.astype(str)); ev = dict(zip(ids, map(int, events(g.taken.reset_index(drop=True)))))
        sizes = pd.Series(ev).value_counts()
        cov = pd.DataFrame([coverage_curve(ids, ev, sd, marks) for sd in range(5)]).mean()
        rows.append(dict(user=u, n=len(ids), events=len(sizes), single=(sizes == 1).sum(), top=sizes.max(), **cov.to_dict()))
    R = pd.DataFrame(rows)
    L.append(f"DISBench whole libraries ({len(R)} users; photos per library median {int(R.n.median())}, events median "
             f"{int(R.events.median())}, photos per event median {R.n.sum() / R.events.sum():.1f}):")
    L.append("| m photos judged | " + " | ".join(str(m) for m in marks) + " |")
    L.append("|---|" + "---|" * len(marks))
    L.append("| covered, median over libraries [p10, p90] | " + " | ".join(
        f"{R[m].median():.3f} [{R[m].quantile(.1):.3f}, {R[m].quantile(.9):.3f}]" for m in marks) + " |")
    L.append("")
    D = people()
    rows = []
    for per, g in D[D.cond == "tired"].groupby("person"):
        ids = list(g.item_id); ev = dict(zip(ids, g.event))
        cov = pd.DataFrame([coverage_curve(ids, ev, sd, (10, 25, 50, 100)) for sd in range(5)]).mean()
        rows.append(dict(person=per, n=len(ids), events=len(set(ev.values())), **cov.to_dict()))
    P = pd.DataFrame(rows)
    L.append(f"The 32 public person sets (photos median {int(P.n.median())}, events median {int(P.events.median())}):")
    L.append("| m photos judged | 10 | 25 | 50 | 100 |")
    L.append("|---|---|---|---|---|")
    L.append("| covered, median [p10, p90] | " + " | ".join(f"{P[m].median():.3f} [{P[m].quantile(.1):.3f}, {P[m].quantile(.9):.3f}]"
                                                 for m in (10, 25, 50, 100)) + " |")
    txt = "\n".join(L)
    (OUT / "coverage.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    {"texts": cmd_texts, "facts": cmd_facts, "looks": cmd_looks, "private": cmd_private,
     "coverage": cmd_coverage}[sys.argv[1]]()
