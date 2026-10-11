"""WHERE IS THE FAST STAGE GOOD ENOUGH ON ITS OWN? (MAC_INBOX M38 -> RESULTS 45). No new judge calls.

On Reza's phone the judge runs ~0.3 photos/s sustained: "food photos" took 149 min (4,305 calls). The phone already
holds one PE-Core-B-16 image vector per photo, and ranking by it takes milliseconds. Question: for which searches is
the vector ranking's FIRST SCREEN (top 20 / 50) already as right as the judge's, and can the phone tell, before any
judge call, which searches those are?

Sets
  audit4  the RESULTS 34/40-44 set: 4 DISBench libraries (7,886 photos), 6 searches; judge = real phone weights
          (eval/recall_audit/scores_q3vl4b_real, P(yes) on every photo, cut judgeCutoff 0.99 / 0.7); ranking =
          PE-Core-B-16 (eval/apple_photos/looks_b16.parquet). Truth = eye labels: RESULTS 34's stratified estimator
          (eval_recall.score_systems) for recall and for precision at N >= 100, and a CENSUS of every photo on the first
          screens (combined library top 50; each library's top 20, vector and judge-verified) labelled blind here
          (labels_*.txt; photos already eye-labelled keep their label).
  ev16    the 16 unseen DISBench libraries of RESULTS 27/41/42 (30,273 photos, 12 searches each); judge = Qwen3-VL-4B
          16-bit on every in-scope photo, cut 0.7 (eval/stop_rule/ev16_ranked.parquet); ranking = the phone simulation
          of B-16 (eval/tiles_phone/ev16_vectors.npy, whole vector, plan looks minus avoid as val16). Eye truth = a
          stratified blind sample of the vector top 50 (judge-yes / judge-no strata, uniform inside each).

  python eval/eval_vector_enough.py sheets      -> eval/vector_enough/sheets/*.jpg + key.json (key NOT read before labels)
  python eval/eval_vector_enough.py analyze     -> report_audit4.txt, report_ev16.txt, searches.tsv
  python eval/eval_vector_enough.py rule        -> rule.txt (leave-one-library-out)
DISBench images are public Flickr data; sheets are git-ignored (*.jpg).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/vector_enough")
NS = (20, 50, 100, 200, 500)
BUDGETS = (0, 25, 50, 100, 200)
SCREENS = (20, 50)
FAST = "timed round_calls=120 inter=0 topup=30 head=150 a1=0.2 fast_stop=480 stop_topup=150"   # phoneFastParams
EV16_SAMPLE = 10            # per (query, judge stratum) of the ev16 vector top 50
SHORT = {"photos with a dog": "dog", "photos with a car": "car", "photos with a bicycle": "bicycle",
         "beach photos": "beach", "sunset photos": "sunset", "food photos": "food", "photos of a cat": "cat",
         "photos of a church": "church", "photos of flowers": "flowers", "photos with a boat": "boat",
         "photos taken at night": "night", "selfies": "selfies"}


# ------------------------------------------------------------------------------------------------------------- data
def audit4():
    """Per query: the 7,886 photos in B-16 order with P(yes), cut, user, path."""
    from eval_recall import QUERIES
    from score_apple_photos import judge_cutoff
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores_q3vl4b_real").glob("part*.parquet"))])
    lk = pd.read_parquet("eval/apple_photos/looks_b16.parquet").set_index("item_id")
    out = {}
    for q in QUERIES:
        d = sc[sc["query"] == q].set_index("item_id")
        cut = judge_cutoff(d.question.iloc[0])
        d = d.assign(s=lk.loc[d.index, q].to_numpy(), y=d.p >= cut, cut=cut)
        out[q] = d.sort_values("s", ascending=False, kind="stable")
    return out


def ev16():
    """Per (query, user): in-scope photos in phone-sim B-16 order (whole vector, looks minus avoid; = val16) + judge yes.
    Searches whose plan has no looks (selfies: the app's default look is not in these texts) are returned with s = 0."""
    d = pd.read_parquet("eval/stop_rule/ev16_ranked.parquet")
    d["item_id"] = d.item_id.astype(str)
    V = np.load("eval/tiles_phone/ev16_vectors.npy", mmap_mode="r")
    ids = Path("eval/tiles_phone/ev16_ids.txt").read_text().split()
    row = {i: k for k, i in enumerate(ids)}
    T = np.load("eval/tiles_phone/ev16_texts.npz")
    out = {}
    for (u, q), g in d.groupby(["user", "query"], sort=True):
        k = f"{q}|{u}".replace("/", "_")
        r = np.array([row[i] for i in g.item_id])
        L = T[k + "|looks"] if k + "|looks" in T.files else np.zeros((0, 1024), np.float32)
        A = T[k + "|avoid"] if k + "|avoid" in T.files else np.zeros((0, 1024), np.float32)
        W = np.asarray(V[r, 0], np.float32)
        w = (W @ L.T).mean(1) if len(L) else np.zeros(len(r))
        av = (W @ A.T).mean(1) if len(A) else 0
        g = g.assign(s=w - av, has_looks=len(L) > 0)
        out[(q, u)] = g.sort_values("s", ascending=False, kind="stable").reset_index(drop=True)
    return out


def truth_audit4(q, i, stratum, old, eye, new):
    """One photo's eye label on the RESULTS 34 truth (kept photos: the earlier label first, as score_systems), else the
    census label written here."""
    if stratum == "kept" and (q, i) in old:
        return old[(q, i)]
    return eye.get((q, i)) or old.get((q, i)) or new.get((q, i))


def screen_sets(d, S):
    """First screens of one search (d in vector order): vector top S, judge-verified top S (first S judge-yes in the
    order the head is judged = vector order)."""
    return list(d.index[:S]), list(d.index[d.y.to_numpy()][:S])


# ----------------------------------------------------------------------------------------------------------- sheets
def sheets():
    from PIL import Image, ImageDraw, ImageFont
    from findpics import store
    from findpics.media import load_image
    from eval_recall import load_scores, old_labels, _labels_eye
    rng = np.random.default_rng(45)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    st = load_scores().set_index(["query", "item_id"]).stratum
    old, eye = old_labels(), _labels_eye()
    A = audit4()
    items = []                                    # (query, item_id, path, set)
    for q, d in A.items():
        need = set()
        for S, per_lib in ((50, False), (20, True)):
            groups = d.groupby("user") if per_lib else [("all", d)]
            for _, g in groups:
                v, j = screen_sets(g, S)
                need |= set(v) | set(j)
        for i in sorted(need):
            if truth_audit4(q, i, st[(q, i)], old, eye, {}) is None:
                items.append((q, i, d.path[i], "audit4"))
    idx = store.load("data/public/index_disbench")
    path = dict(zip(idx.items.item_id.astype(str), idx.items.path))
    E = ev16()
    for q in sorted({q for q, _ in E}):
        if q == "selfies":
            continue                              # no looks in these plans: no vector ranking to test
        pool = pd.concat([g.head(50) for (qq, _), g in E.items() if qq == q])
        for yv in (True, False):
            p = pool[pool.yes == yv]
            for i in p.item_id.iloc[rng.choice(len(p), size=min(EV16_SAMPLE, len(p)), replace=False)]:
                items.append((q, i, path[i], "ev16"))
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    key = []
    for q in sorted({it[0] for it in items}):
        its = [it for it in items if it[0] == q]
        its = [its[k] for k in rng.permutation(len(its))]          # audit4 census + ev16 sample mixed: blind to source
        ims = [(it, load_image(it[2], max_side=896)) for it in its]
        s = 0
        for orient in ("land", "port"):
            o = [(it, im) for it, im in ims if (im.width >= im.height) == (orient == "land")]
            cols, rows = (2, 3) if orient == "land" else (3, 2)
            for b in range(0, len(o), 6):
                chunk = o[b:b + 6]
                W = max(im.width for _, im in chunk); H = max(im.height for _, im in chunk)
                sheet = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * H + (rows - 1) * 6), (255, 255, 255))
                dr = ImageDraw.Draw(sheet)
                s += 1; name = f"{SHORT[q]}_{s:02d}.jpg"
                for k, (it, im) in enumerate(chunk):
                    x, y = (k % cols) * (W + 6), (k // cols) * (H + 6)
                    sheet.paste(im, (x, y))
                    dr.rectangle([x, y, x + 26, y + 28], fill=(255, 255, 0))
                    dr.text((x + 6, y + 1), str(k + 1), fill=(0, 0, 0), font=font)
                    key.append(dict(sheet=name, number=k + 1, query=it[0], item_id=it[1], source=it[3]))
                sheet.save(OUT / "sheets" / name, quality=95)
        print(q, len(its), "photos,", s, "sheets", flush=True)
    json.dump(key, open(OUT / "key.json", "w"), indent=1)


def new_labels():
    """labels_<short>.txt: '<sheet> <number> <match|no_match|unsure> <note>' written blind from the sheets."""
    recs = json.load(open(OUT / "key.json"))
    key = {(r["sheet"], r["number"]): r for r in recs}
    out = {}
    for f in sorted(OUT.glob("labels_*.txt")):
        for line in open(f):
            if not line.strip() or line.startswith("#"):
                continue
            sh, n, v = line.split()[:3]
            assert v in ("match", "no_match", "unsure"), line
            sh = sh if sh.endswith(".jpg") else sh + ".jpg"          # some labellers wrote the sheet name bare
            r = key[(sh, int(n))]
            out[(r["query"], r["item_id"])] = v
    return out, recs


# ---------------------------------------------------------------------------------------------------------- analyze
def _cp(m, n):
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if m == 0 else beta.ppf(0.025, m, n - m + 1)
    hi = 1.0 if m == n else beta.ppf(0.975, m + 1, n - m)
    return lo, hi


def _fmt(m, n):
    if n == 0:
        return "-"
    lo, hi = _cp(m, n)
    return f"{m:g}/{n} = {m / n:.2f} [{lo:.2f}-{hi:.2f}]"


def hybrid(ids, y, S, c):
    """First screen of S after c judge calls on the vector top c: the judge-yes among them (vector order), then the
    unchecked vector ranks c+1... Returns (screen ids, n checked-yes on it)."""
    yes = [i for i, v in zip(ids[:c], y[:c]) if v][:S]
    return yes + list(ids[c:c + S - len(yes)]), len(yes)


def features(s, k=20):
    """What the phone can compute in milliseconds from the in-scope fast scores (no labels)."""
    s = np.sort(np.asarray(s, float))[::-1]
    n = len(s)
    med = float(np.median(s)) if n else 0.0
    sk = float(s[min(k, n) - 1]) if n else 0.0
    return dict(n=n, s1=float(s[0]) if n else 0.0, s20=sk, s50=float(s[min(50, n) - 1]) if n else 0.0, med=med,
                gap20=sk - med)


def searches():
    """One row per (set, library, query) search: judge-relative and eye first-screen precision + features."""
    from eval_recall import load_scores, old_labels, _labels_eye
    import tune_stop_time as TT
    st = load_scores().set_index(["query", "item_id"]).stratum
    old, eye = old_labels(), _labels_eye()
    new, key = new_labels()
    rates = ev16_rates(new, key)
    _, r = TT.parse(FAST)
    rows, photos = [], []
    A = audit4()
    for q, d in A.items():
        for u, g in d.groupby("user"):
            ids, y = g.index.to_numpy(), g.y.to_numpy()
            lab = lambda i: truth_audit4(q, i, st[(q, i)], old, eye, new)         # noqa: E731
            row = dict(set="audit4", library=u, query=q, kind=kind_of(q), K=int(y.sum()), **features(g.s))
            order, rounds = TT.sim_timed(y, r, 0)
            row.update(_screen_stats(ids, y, order, rounds, lab, None))
            rows.append(row)
            L = [lab(i) for i in ids[:50]]
            photos.append(pd.DataFrame(dict(set="audit4", library=u, query=q, rank=np.arange(len(L)), s=g.s.to_numpy()[:50],
                                            y=y[:50], eye_m=[float(v == "match") if v else np.nan for v in L])))
    for (q, u), g in ev16().items():
        ids, y = g.item_id.to_numpy(), g.yes.to_numpy()
        row = dict(set="ev16", library=u, query=q, kind=kind_of(q), K=int(y.sum()), has_looks=bool(g.has_looks.iloc[0]),
                   **features(g.s))
        order, rounds = TT.sim_timed(y, r, 0)
        row.update(_screen_stats(ids, y, order, rounds, None, rates.get(q)))
        rows.append(row)
        rt = rates.get(q)
        em = (np.where(y[:50], rt[True][0] / rt[True][1], rt[False][0] / max(rt[False][1], 1)) if rt
              else np.full(min(50, len(y)), np.nan))
        photos.append(pd.DataFrame(dict(set="ev16", library=u, query=q, rank=np.arange(min(50, len(y))),
                                        s=g.s.to_numpy()[:50], y=y[:50], eye_m=em)))
    R = pd.DataFrame(rows)
    R["has_looks"] = R.has_looks.fillna(True).astype(bool)
    pd.concat(photos, ignore_index=True).to_parquet(OUT / "photos.parquet")      # top 50 of every search (git-ignored)
    return R


def kind_of(q):
    return {"photos with a dog": "object", "photos with a car": "object", "photos with a bicycle": "object",
            "photos with a boat": "object", "photos taken at night": "fact", "selfies": "no-looks"}.get(q, "subject")


def ev16_rates(new, key):
    """Eye match rate of the ev16 vector-top-50 sample per (query, judge stratum)."""
    yes_of = {}
    d = pd.read_parquet("eval/stop_rule/ev16_ranked.parquet")
    d["item_id"] = d.item_id.astype(str)
    for q, i, v in zip(d["query"], d.item_id, d.yes):
        yes_of[(q, i)] = bool(v)
    out = {}
    for k in key:
        if k["source"] != "ev16" or (k["query"], k["item_id"]) not in new:
            continue
        v = new[(k["query"], k["item_id"])]
        o = out.setdefault(k["query"], {True: [0, 0, 0], False: [0, 0, 0]})[yes_of[(k["query"], k["item_id"])]]
        o[0] += v == "match"; o[1] += 1; o[2] += v == "unsure"
    return out


def _screen_stats(ids, y, order, rounds, lab, rate):
    """First-screen numbers of one search. lab(i) -> eye label (audit4 census) or None; rate: ev16 eye match rates per
    judge stratum {True: [match, n, unsure], False: [...]} (P = yes-share x rate_yes + no-share x rate_no)."""
    out = {}
    yes_pos = order[y[order]]                     # judge-yes in the order fast mode finds them
    for S in SCREENS:
        v = list(ids[:S])
        jv = list(ids[yes_pos[:S]])
        out[f"vec{S}_jrel"] = float(np.mean(y[:S])) if len(v) else float("nan")
        out[f"vec{S}_n"] = len(v)
        out[f"jud{S}_n"] = len(jv)
        out[f"jud{S}_calls"] = int(np.where(np.isin(order, yes_pos[:S]))[0].max() + 1) if len(jv) else 0
        for c in BUDGETS[1:]:
            scr, ny = hybrid(ids, y, S, c)
            out[f"hyb{S}_{c}_jrel"] = float(np.mean([yy for yy in y[np.isin(ids, scr)]])) if scr else float("nan")
            out[f"hyb{S}_{c}_checked"] = ny
        if lab is not None:
            def eye_p(sc):
                L = [lab(i) for i in sc]
                return sum(x == "match" for x in L), sum(x is not None for x in L), sum(x is None for x in L)
            for name, sc in [("vec", v), ("jud", jv)] + [(f"hyb{S}_{c}_", hybrid(ids, y, S, c)[0]) for c in BUDGETS[1:]]:
                m, n, miss = eye_p(sc)
                pre = f"{name}{S}" if name in ("vec", "jud") else name.rstrip("_")
                out[f"{pre}_eye_m"], out[f"{pre}_eye_n"], out[f"{pre}_eye_missing"] = m, n, miss
        elif rate is not None:
            ry = rate[True][0] / max(rate[True][1], 1); rn = rate[False][0] / max(rate[False][1], 1)
            out[f"vec{S}_eye_est"] = float(np.mean(np.where(y[:S], ry, rn))) if len(v) else float("nan")
            out[f"jud{S}_eye_est"] = ry if len(jv) else float("nan")
    # completeness: first round's bound and where the head rule is satisfied (fast mode's own first answer)
    out["calls_first_bound"] = rounds[0][0]
    out["calls_first_answer"] = rounds[-1][0]
    out["found_first_answer"] = int(y[order].sum())
    return out


def analyze():
    from eval_recall import QUERIES, load_scores, old_labels, _labels_eye, score_systems
    import tune_stop_time as TT
    df = load_scores()
    st = df.set_index(["query", "item_id"]).stratum
    old, eye = old_labels(), _labels_eye()
    new, key = new_labels()
    _, r = TT.parse(FAST)
    A = audit4()
    L = ["RESULTS 45 / audit4: 4 DISBench libraries as ONE 7,886-photo library (as RESULTS 40-44), 6 searches.",
         "Truth = eye labels. N >= 100: RESULTS 34's stratified estimator (score_systems, unsure = no match, 95% bootstrap).",
         "First screens (top 20 / 50): census, every photo eye-labelled (earlier label if any, else the blind census",
         "labels of this section); 95% Clopper-Pearson over the screen's photos. Judge = real phone weights at judgeCutoff;",
         f"fast mode = phoneFastParams ({FAST}), seed 0; its results in the order it finds them.", ""]
    # ---- N >= 50: estimator (vector top N, judge-verified top N, the fast first answer)
    systems, info = {}, []
    for q, d in A.items():
        ids, y = d.index.to_numpy(), d.y.to_numpy()
        order, rounds = TT.sim_timed(y, r, 0)
        yes_pos = order[y[order]]
        for N in NS:
            systems.setdefault(f"vector top {N}", {})[q] = set(ids[:N])
            systems.setdefault(f"judged top {N}", {})[q] = set(ids[yes_pos[:N]])
        systems.setdefault("fast first answer (all)", {})[q] = set(ids[yes_pos])
        calls_n = {N: (int(np.where(np.isin(order, yes_pos[:N]))[0].max() + 1) if len(yes_pos) >= N else None)
                   for N in NS}
        info.append(dict(query=q, judge_yes=int(y.sum()), first_answer_calls=len(order), first_answer_found=len(yes_pos),
                         **{f"calls_to_{N}": calls_n[N] for N in NS}))
    R = score_systems(df, old, eye, systems)
    R.to_csv(OUT / "audit4_estimator.tsv", sep="\t", index=False)
    I = pd.DataFrame(info)
    L.append("Judge calls fast mode needs before N judge-verified photos are on screen (minutes on the m4 / m8 clocks):")
    for x in info:
        cells = []
        for N in NS:
            c = x[f"calls_to_{N}"]
            cells.append(f"{N}: " + ("never in the first answer" if c is None else
                                      f"{c} ({TT.minutes_of(c, 'm4'):.1f} / {TT.minutes_of(c, 'm8'):.1f} min)"))
        L.append(f"  {SHORT[x['query']]:8s} judge-yes {x['judge_yes']:5d}; first answer {x['first_answer_found']} found in "
                 f"{x['first_answer_calls']} calls; " + "; ".join(cells))
    L.append("")
    L.append("| search | N | vector: precision [95%] | vector: recall [95%] | judged: precision [95%] | judged: recall [95%] |")
    L.append("|---|---|---|---|---|---|")
    g = R.set_index(["system", "query"])
    for q in QUERIES:
        for N in NS:
            a, b = g.loc[(f"vector top {N}", q)], g.loc[(f"judged top {N}", q)]
            L.append(f"| {SHORT[q]} | {N} | {a.precision_est:.2f} [{a.precision_boot_lo:.2f}-{a.precision_boot_hi:.2f}] | "
                     f"{a.recall:.3f} [{a.recall_boot_lo:.3f}-{a.recall_boot_hi:.3f}] | "
                     f"{b.precision_est:.2f} [{b.precision_boot_lo:.2f}-{b.precision_boot_hi:.2f}] ({b.returned_in_library}) | "
                     f"{b.recall:.3f} [{b.recall_boot_lo:.3f}-{b.recall_boot_hi:.3f}] |")
        a = g.loc[("fast first answer (all)", q)]
        L.append(f"| {SHORT[q]} | fast first answer: {a.returned_in_library} | | | {a.precision_est:.2f} "
                 f"[{a.precision_boot_lo:.2f}-{a.precision_boot_hi:.2f}] | {a.recall:.3f} [{a.recall_boot_lo:.3f}-{a.recall_boot_hi:.3f}] |")
    # ---- first screens: census
    L += ["", "FIRST SCREEN, census by eye (combined library). Hybrid = vector top S shown at once; the judge checks the",
          "vector top c; the screen = its yes photos, then unchecked vector ranks c+1... (c = 0 is vector only).",
          "| search | S | vector only (c=0) | c=25 | c=50 | c=100 | c=200 | judge-verified (fast mode) |", "|---|---|---|---|---|---|---|---|"]
    tot = {}
    for q, d in A.items():
        ids, y = d.index.to_numpy(), d.y.to_numpy()
        lab = lambda i: truth_audit4(q, i, st[(q, i)], old, eye, new)     # noqa: E731
        for S in SCREENS:
            cells = []
            sets = [hybrid(ids, y, S, c)[0] for c in BUDGETS] + [list(ids[y][:S])]
            for k, sc in enumerate(sets):
                Lb = [lab(i) for i in sc]
                m, n, miss = sum(v == "match" for v in Lb), sum(v is not None for v in Lb), sum(v is None for v in Lb)
                t = tot.setdefault((S, k), [0, 0, 0]); t[0] += m; t[1] += n; t[2] += miss
                cells.append(_fmt(m, n) + (f" ({miss} unlabelled)" if miss else ""))
            L.append(f"| {SHORT[q]} | {S} | " + " | ".join(cells) + " |")
    for S in SCREENS:
        L.append(f"| all 6 | {S} | " + " | ".join(_fmt(*tot[(S, k)][:2]) for k in range(len(BUDGETS) + 1)) + " |")
    txt = "\n".join(L)
    (OUT / "report_audit4.txt").write_text(txt)
    print(txt)
    # ---- per-library searches (audit4 24 + ev16 192)
    S_ = searches()
    S_.to_csv(OUT / "searches.tsv", sep="\t", index=False, float_format="%.4g")
    rep_ev16(S_, new, key)


def rep_ev16(S_, new, key):
    rates = ev16_rates(new, key)
    L = ["RESULTS 45 / ev16: 16 unseen DISBench libraries, 12 searches each; judge = Qwen3-VL-4B 16-bit on every in-scope",
         "photo at 0.7 (looser than the phone's 0.99 for 'Is this a photo of X?'). Vector = phone-sim B-16 whole vector.",
         "Judge-relative = share of the vector top S the judge says yes to (pooled over libraries).",
         "Eye = blind labels of a uniform sample of the pooled vector top 50, split by the judge's answer (10 + 10 per",
         "search); eye-estimated precision = yes-share x eye rate(yes) + no-share x eye rate(no); 95% from Jeffreys draws.",
         "", "| search | judge-yes per library (median) | vector top 20: judge-rel. | top 20: eye est. [95%] | "
         "top 50: judge-rel. | top 50: eye est. [95%] | eye rate in judge-yes / judge-no (unsure) |",
         "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    E = S_[S_["set"] == "ev16"]
    for q, g in E.groupby("query", sort=True):
        rt = rates.get(q)
        cells = []
        for S in SCREENS:
            jr = (g[f"vec{S}_jrel"] * g[f"vec{S}_n"]).sum() / g[f"vec{S}_n"].sum()
            if rt:
                ry = rng.beta(rt[True][0] + .5, rt[True][1] - rt[True][0] + .5, 4000)
                rn = rng.beta(rt[False][0] + .5, rt[False][1] - rt[False][0] + .5, 4000)
                pt = jr * rt[True][0] / rt[True][1] + (1 - jr) * rt[False][0] / rt[False][1]
                lo, hi = np.percentile(jr * ry + (1 - jr) * rn, [2.5, 97.5])
                cells += [f"{jr:.2f}", f"{pt:.2f} [{lo:.2f}-{hi:.2f}]"]
            else:
                cells += [f"{jr:.2f}", "-"]
        rr = (f"{rt[True][0]}/{rt[True][1]} ({rt[True][2]}) / {rt[False][0]}/{rt[False][1]} ({rt[False][2]})" if rt else "-")
        L.append(f"| {SHORT[q]} | {g.K.median():.0f} | " + " | ".join(cells) + f" | {rr} |")
    txt = "\n".join(L)
    (OUT / "report_ev16.txt").write_text(txt)
    print(txt)


# ------------------------------------------------------------------------------------------------------------- rule
ENOUGH = 0.9          # "vectors are enough": >= 18 of the vector top 20 are judge-yes (the judge's own first screen = 20/20)


def _fit_threshold(x, t, safe=None):
    """Threshold T on feature x (predict enough iff x >= T) maximizing training accuracy (ties -> higher T). safe: the
    lowest T whose training 'enough' predictions are >= safe truly enough (with >= 5 such predictions)."""
    cand = np.unique(np.concatenate([x, [np.inf]]))
    best, bt = -1, np.inf
    for T in cand:
        p = x >= T
        if safe is not None:
            if p.sum() >= 5 and t[p].mean() >= safe:
                return T
            continue
        a = (p == t).mean()
        if a > best or (a == best and T > bt):
            best, bt = a, T
    return bt if safe is None else np.inf


def lolo(R, feat, by_kind=False, safe=None):
    """Leave-one-library-out predictions: thresholds fitted on the other libraries only."""
    pred = np.zeros(len(R), bool)
    Ts = {}
    for lib in R.library.unique():
        tr, te = R.library != lib, R.library == lib
        groups = R.kind.unique() if by_kind else [None]
        for k in groups:
            mtr = tr & (R.kind == k) if k else tr
            mte = te & (R.kind == k) if k else te
            if not mte.any():
                continue
            T = _fit_threshold(R.loc[mtr, feat].to_numpy(), R.loc[mtr, "enough"].to_numpy(), safe) if mtr.any() else np.inf
            pred[mte.to_numpy()] = R.loc[mte, feat].to_numpy() >= T
            Ts.setdefault(k, []).append(T)
    return pred, Ts


def rule():
    import tune_stop_time as TT
    R = pd.read_csv(OUT / "searches.tsv", sep="\t", dtype={"library": str})
    R["enough"] = R.vec20_jrel >= ENOUGH
    elig = R.has_looks.astype(bool)
    L = [f"RESULTS 45 rule: 'vectors are enough for the first screen' = >= {ENOUGH:.0%} of the vector top 20 are judge-yes.",
         f"{len(R)} searches on 20 libraries (audit4 4 x 6 + ev16 16 x 12); base rate enough {R.enough.sum()}/{len(R)}.",
         "Searches whose plan has no looks (selfies here) cannot be ranked by vectors: always 'judge' (not fitted).",
         "Leave-one-library-out: every threshold is fitted on the other 19 libraries only.", ""]
    rows = []
    variants = {"always judge": None, "always vectors (looks)": "all",
                "s20 >= T": ("s20", False, None), "s20 >= T per kind": ("s20", True, None),
                "gap20 >= T": ("gap20", False, None), "s1 >= T": ("s1", False, None),
                "s20 >= T, safe (train >= 90% enough)": ("s20", False, 0.9),
                "gap20 >= T, safe (train >= 90% enough)": ("gap20", False, 0.9)}
    preds = {}
    for name, v in variants.items():
        if v is None:
            p = np.zeros(len(R), bool); Ts = {}
        elif v == "all":
            p = elig.to_numpy().copy(); Ts = {}
        else:
            p = np.zeros(len(R), bool)
            pe, Ts = lolo(R[elig].reset_index(drop=True), *v)
            p[np.where(elig)[0]] = pe
        preds[name] = p
        t = R.enough.to_numpy()
        tp, fp, fn, tn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()
        vp = R.vec20_jrel[p].mean() if p.any() else float("nan")
        rows.append(dict(rule=name, accuracy=f"{tp + tn}/{len(R)} = {(tp + tn) / len(R):.3f}",
                         said_vectors=int(p.sum()), of_which_enough=f"{tp}/{p.sum()}" if p.any() else "-",
                         missed_enough=f"{fn}/{t.sum()}", vec20_jrel_when_vectors=round(vp, 3),
                         thresholds=" ".join(f"{k or 'all'}:{np.median(x):.4f}[{np.min(x):.4f}-{np.max(x):.4f}]"
                                             for k, x in Ts.items())))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    L.append(T.to_string(index=False))
    # chosen rule: in-sample threshold for the phone + its LOLO behaviour per query
    chosen = "s20 >= T, safe (train >= 90% enough)"
    p = preds[chosen]
    R["pred_vectors"] = p
    T_all = _fit_threshold(R.loc[elig, "s20"].to_numpy(), R.loc[elig, "enough"].to_numpy(), 0.9)
    T_acc = _fit_threshold(R.loc[elig, "s20"].to_numpy(), R.loc[elig, "enough"].to_numpy())
    L += ["", f"CHOSEN: {chosen}. Fitted on all 20 libraries (for the phone): T = {T_all:.4f}"
              f" (accuracy-max T would be {T_acc:.4f}).", "",
          "Per search kind (LOLO decisions): said vectors / searches; vector top-20 judge-rel. precision when vectors / when judge",
          "| set | search | said vectors | truly enough | vec P@20 (said vectors) | vec P@20 (said judge) |", "|---|---|---|---|---|---|"]
    for (s, q), g in R.groupby(["set", "query"], sort=True):
        a, b = g[g.pred_vectors], g[~g.pred_vectors]
        L.append(f"| {s} | {SHORT[q]} | {len(a)}/{len(g)} | {g.enough.sum()}/{len(g)} | "
                 f"{a.vec20_jrel.mean():.2f} | {b.vec20_jrel.mean():.2f} |".replace("nan", "-"))
    # eye consequences on audit4 (census) and ev16 (estimate)
    L += ["", "Eye precision of the first screen (20) under the LOLO decision: vectors when it says vectors, else the",
          "judge-verified screen; vs always judge-verified."]
    a4 = R[R["set"] == "audit4"]
    for nm, m in (("said vectors", a4.pred_vectors), ("said judge", ~a4.pred_vectors)):
        g = a4[m]
        if len(g):
            L.append(f"  audit4 {nm}: {len(g)} searches; vector screen {_fmt(g.vec20_eye_m.sum(), g.vec20_eye_n.sum())}; "
                     f"judge-verified screen {_fmt(g.jud20_eye_m.sum(), g.jud20_eye_n.sum())}")
    e = R[R["set"] == "ev16"]
    for c in ("vec20_eye_est", "jud20_eye_est", "vec20_eye_m", "vec20_eye_n", "jud20_eye_m", "jud20_eye_n"):
        if c not in R:
            R[c] = np.nan; e = R[R["set"] == "ev16"]; a4 = R[R["set"] == "audit4"]
    for nm, m in (("said vectors", e.pred_vectors), ("said judge", ~e.pred_vectors)):
        g = e[m]
        if len(g):
            L.append(f"  ev16 {nm}: {len(g)} searches; vector screen eye est. "
                     f"{(g.vec20_eye_est * g.vec20_n).sum() / g.vec20_n.sum():.2f}; judge-verified eye est. "
                     f"{(g.jud20_eye_est * g.jud20_n).sum() / max(g.jud20_n.sum(), 1):.2f} "
                     f"(judge-verified screens hold {g.jud20_n.sum()} photos vs {g.vec20_n.sum()})")
    # ---- the same threshold per PHOTO: show at once every vector top-20 photo whose own score >= t
    P = pd.read_parquet(OUT / "photos.parquet")
    P = P[(P["rank"] < 20) & (P["query"] != "selfies")]
    L += ["", "PER PHOTO (the same score, applied to each photo): the first screen shows at once only the vector top-20",
          "photos whose own score >= t; t = the lowest threshold whose training photos above it are >= 90% judge-yes,",
          "fitted leave-one-library-out. Held-out results pooled over all 20 libraries:"]
    shown = []
    ts = []
    for lib in P.library.unique():
        tr, te = P[P.library != lib], P[P.library == lib]
        cand = np.sort(tr.s.unique())
        t = np.inf
        for c in cand:                                   # lowest c with >= 90% judge-yes above it (and >= 50 photos)
            m = tr.s >= c
            if m.sum() >= 50 and tr.y[m].mean() >= 0.9:
                t = c; break
        ts.append(t)
        shown.append(te[te.s >= t].assign(t=t))
    Sh = pd.concat(shown)
    a4, e6 = Sh[Sh["set"] == "audit4"], Sh[Sh["set"] == "ev16"]
    L.append(f"  t (LOLO) median {np.median(ts):.4f} [{np.min(ts):.4f}-{np.max(ts):.4f}]; photos shown {len(Sh)} of "
             f"{len(P)} top-20 slots; searches with >= 1 shown {Sh.groupby(['set', 'library', 'query']).ngroups} of "
             f"{P.groupby(['set', 'library', 'query']).ngroups}; with all 20 shown "
             f"{(Sh.groupby(['set', 'library', 'query']).size() >= 20).sum()}")
    L.append(f"  judge-relative precision of what is shown: {_fmt(int(Sh.y.sum()), len(Sh))}")
    L.append(f"  eye, audit4 (census): {_fmt(int(a4.eye_m.sum()), int(a4.eye_m.notna().sum()))}; ev16 (estimated from "
             f"the stratified sample): {e6.eye_m.mean():.2f} over {len(e6)} photos")
    L.append("  per search kind: shown / slots, judge-rel. precision, eye")
    for q, g in Sh.groupby("query"):
        n_slots = len(P[P["query"] == q])
        eye_s = (f"{g.eye_m.mean():.2f}" if g.eye_m.notna().any() else "-")
        L.append(f"    {SHORT[q]:8s} {len(g):4d}/{n_slots:4d}  {g.y.mean():.2f}  eye {eye_s}")
    # ---- budget for the searches that need the judge
    J = R[~R.pred_vectors & elig]
    L += ["", f"BUDGET for the {len(J)} searches the rule sends to the judge (LOLO). Hybrid screen of 20 after c calls on",
          "the vector top c (yes photos first, then unchecked vector ranks), judge-relative precision, mean over searches:"]
    for c in BUDGETS[1:]:
        L.append(f"  c = {c:3d} ({TT.minutes_of(c, 'm4'):.1f} min m4 / {TT.minutes_of(c, 'm8'):.1f} m8): precision "
                 f"{J[f'hyb20_{c}_jrel'].mean():.3f}; checked-yes on screen median {J[f'hyb20_{c}_checked'].median():.0f}; "
                 f"screens >= 0.9 judge-rel. {(J[f'hyb20_{c}_jrel'] >= 0.9).sum()}/{len(J)}; share of the judge's own "
                 f"first screen (min(20, judge-yes)) already checked: mean "
                 f"{(J[f'hyb20_{c}_checked'] / J.K.clip(upper=20).clip(lower=1)).mean():.3f}, complete in "
                 f"{(J[f'hyb20_{c}_checked'] >= J.K.clip(upper=20)).sum()}/{len(J)}")
    L.append(f"  c = 0 (vectors only): {J.vec20_jrel.mean():.3f}")
    full = J[J.jud20_n >= 20]
    L.append(f"  calls until 20 judge-verified photos (searches with >= 20 found: {len(full)}/{len(J)}): median "
             f"{full.jud20_calls.median():.0f}, p90 {full.jud20_calls.quantile(.9):.0f}, max {full.jud20_calls.max():.0f}")
    L.append(f"  calls until the first bound (round 1): median {J.calls_first_bound.median():.0f}; until the first "
             f"answer is complete: median {J.calls_first_answer.median():.0f}, p90 {J.calls_first_answer.quantile(.9):.0f}")
    # stop rule: first screen final when 20 verified are found OR the first bound is out (whichever first), capped
    for cap in (50, 150, 200):
        stop = np.minimum(np.where(J.jud20_n >= 20, J.jud20_calls, np.inf), J.calls_first_bound).clip(upper=cap)
        L.append(f"  screen final at min(20 verified, first bound) capped at {cap}: median {np.median(stop):.0f} calls "
                 f"({TT.minutes_of(np.median(stop), 'm4'):.1f} min m4), p90 {np.quantile(stop, .9):.0f}; share ending "
                 f"before the cap {(stop < cap).mean():.2f}")
    txt = "\n".join(L)
    (OUT / "rule.txt").write_text(txt)
    R.to_csv(OUT / "searches_rule.tsv", sep="\t", index=False, float_format="%.4g")
    print(txt)


if __name__ == "__main__":
    {"sheets": sheets, "analyze": analyze, "rule": rule}[sys.argv[1]]()
