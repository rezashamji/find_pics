"""DOES VECTOR-FIRST HOLD INSIDE A FACE-FILTERED SET? (MAC_INBOX M45 -> eval/vector_first_person/report.txt)

RESULTS 45 fitted firstScreenDecision (20th-best PE-Core-B-16 look score >= 0.2074 -> the vector top 20 is the first
screen) and likelyAtOnce (per photo >= 0.2036) on plain object/scene searches over whole libraries. Person searches
are excluded only because they were never measured. Here: one person's photos (the product's face filter), ranked by
the look vector, against the phone judge asked about the person in the red box.

Sets (public data only)
  disbench  57 DISBench Flickr libraries (~1,900 photos each). Per library the most frequent face group(s)
            (people.face_groups, buffalo_l, product cuts: expand 3 rounds, accept 0.40, 'closer to someone else'),
            kept if >= MIN_PHOTOS photos. 8 person conditions (COND). Judge = Qwen3-VL-4B phone weights
            (models/q3vl4b_real) on EVERY photo of the person, person crop + red box (engine crop_person="box").
            Vector = phone-sim B-16 (480 px q80 rendition, whole-photo vector), look texts as the planner writes them.
  imdb      Pratt / Hill / Rogen era sets of RESULTS 10/18 (IMDB-WIKI face crops; identity checked by eye; era from
            published weight losses). Looks 'looking heavier' / 'looking fit'; truth = era (a proxy).

  gen  --shard=k/K   (vLLM env, GPU)  -> eval/vector_first_person/part{k}.parquet + vec{k}.npz
  imdb               (vLLM env, GPU)  -> eval/vector_first_person/imdb.parquet
  analyze            (CPU)            -> eval/vector_first_person/report.txt (+ searches.tsv)
  sheets / eye       (CPU)            blind eye-label sheets (native pixels, red box) / score them
DISBench / IMDB-WIKI images are public; sheets are git-ignored (*.jpg).
"""
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/vector_first_person")
MODEL = os.path.abspath("models/q3vl4b_real")
MIN_PHOTOS = 60
ACCEPT = 0.7                       # phone judgeCutoff for a non-"Is this a photo of X?" question
T_SCREEN, T_PHOTO = 0.2074, 0.2036  # VectorFirst.swift
# key: (look as the planner writes it, judge question about the person in the red box, alt look text)
COND = {
    "smiling": ("smiling", "Is the person in the red box smiling?", "a photo of a person smiling"),
    "beach": ("at the beach", "Is the person in the red box at the beach?", "a photo of a person at the beach"),
    "outdoors": ("outdoors", "Is the person in the red box outdoors?", "a photo of a person outdoors"),
    "sunglasses": ("wearing sunglasses", "Is the person in the red box wearing sunglasses?",
                   "a photo of a person wearing sunglasses"),
    "hat": ("wearing a hat", "Is the person in the red box wearing a hat?", "a photo of a person wearing a hat"),
    "snow": ("in the snow", "Is the person in the red box in the snow?", "a photo of a person in the snow"),
    "tired": ("looking tired", "Does the person in the red box look tired?", "a photo of a person looking tired"),
    "eating": ("eating", "Is the person in the red box eating?", "a photo of a person eating"),
}
IMDB_LOOKS = {"heavier": ("looking heavier", "a photo of a person looking heavier"),
              "fit": ("looking fit", "a photo of a person looking fit")}


def _arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def _texts(enc):
    keys, txt = [], []
    for c, (lk, _, alt) in COND.items():
        keys += [f"{c}|look", f"{c}|alt"]; txt += [lk, alt]
    for c, (lk, alt) in IMDB_LOOKS.items():
        keys += [f"{c}|look", f"{c}|alt"]; txt += [lk, alt]
    return dict(zip(keys, enc.texts(txt)))


def people_of(sub, top=3):
    """The library's most frequent people as the product finds them: face groups -> refs -> expand -> item scores."""
    from findpics.people import face_groups, expand_refs, item_person_scores
    from findpics.face_profiles import index_profile
    prof = index_profile(sub)
    G = face_groups(sub, top=top)
    out, used = [], []
    for gi, g in enumerate(G):
        refs = sub.face_emb[np.array(g["faces"], int)]
        if any(float((refs.astype(np.float32) @ u.astype(np.float32).T).max(1).mean()) >= prof["accept"] for u in used):
            continue                                       # a second group of someone already taken
        refs = expand_refs(sub, refs, rounds=3)
        others = [sub.face_emb[np.array(h["faces"], int)] for hj, h in enumerate(G) if hj != gi]
        others = [o for o in others if float((o.astype(np.float32) @ refs.astype(np.float32).T).max(1).mean()) < prof["other"]]
        O = np.concatenate(others) if others else np.zeros((0, sub.face_emb.shape[1]), np.float32)
        score, best = item_person_scores(sub, refs, O)
        rows = np.where(score >= prof["accept"])[0]
        used.append(refs)
        out.append(dict(group=gi, rows=rows, best=best[rows], pscore=score[rows], group_items=len(g["items"])))
    return out


def gen():
    os.environ["FP_VLM_MODEL"] = MODEL
    from PIL import Image
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.models import ImageTextEncoder
    from eval_tiles_phone import rendition
    k, K = map(int, _arg("shard", "0/1").split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    users = sorted(set(user_of))[k::K]
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    if k == 0:
        np.savez(OUT / "texts.npz", **_texts(enc))
    persons, vecs = [], {}
    for u in users:
        sub = store.subset(idx, np.where(user_of == u)[0])
        for p in people_of(sub):
            print(f"{u} group {p['group']}: {len(p['rows'])} photos (group items {p['group_items']})", flush=True)
            if len(p["rows"]) < MIN_PHOTOS:
                continue
            f = sub.faces.iloc[p["best"]]
            d = pd.DataFrame(dict(user=u, person=f"{u}#{p['group']}", item_id=sub.items.item_id.astype(str).to_numpy()[p["rows"]],
                                  path=sub.items.path.to_numpy()[p["rows"]], pscore=p["pscore"],
                                  x1=f.x1.to_numpy(), y1=f.y1.to_numpy(), x2=f.x2.to_numpy(), y2=f.y2.to_numpy(),
                                  img_w=f.img_w.to_numpy(), img_h=f.img_h.to_numpy(), face_px=f.face_px.to_numpy(),
                                  n_faces=sub.faces.groupby("item_row").size().reindex(p["rows"]).fillna(0).astype(int).to_numpy()))
            V = []
            for b in range(0, len(d), 64):
                V.append(enc.images([rendition(Image.open(x)) for x in d.path[b:b + 64]]))
            for i, v in zip(d.item_id, np.concatenate(V)):
                vecs[i] = v.astype(np.float16)
            persons.append((sub, p, d))
    np.savez(OUT / f"vec{k}.npz", ids=np.array(list(vecs)), V=np.stack(list(vecs.values())) if vecs else np.zeros((0, 1024)))
    del enc
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    from findpics.vlm import VLLMJudge
    J = VLLMJudge(model=MODEL, gpu_mem=0.75)
    pp = OUT / "pp"; pp.mkdir(exist_ok=True)
    for sub, p, d in persons:                      # one file per person: a timed-out job loses at most one person
        f = pp / (d.person.iloc[0].replace("@", "_").replace("#", "_g") + ".parquet")
        if f.exists() or (OUT / f"part{k}.parquet").exists():
            continue
        out = []
        for c, (_, q, _) in COND.items():
            pr = _judge_rows(sub, J, p["rows"], p["best"], q, crop_person="box")
            out.append(d.assign(cond=c, question=q, p=pr))
            print(f"{d.person.iloc[0]} {c}: yes {(pr >= ACCEPT).sum()}/{len(pr)}", flush=True)
        pd.concat(out, ignore_index=True).to_parquet(f)
    print("GEN_DONE", k, len(persons), flush=True)


def imdb():
    """B-16 vectors of the RESULTS 10/18 era crops (face-filtered as RESULTS 18: sim_expanded >= 0.40)."""
    from PIL import Image
    from findpics.models import ImageTextEncoder
    from eval_tiles_phone import rendition
    OUT.mkdir(parents=True, exist_ok=True)
    T = Path("data/public/transformation")
    s = pd.read_csv(T / "scored_fit.csv", index_col=0)
    fs = pd.read_csv(T / "face_sims.csv").set_index("key")
    s["sim_expanded"] = fs.loc[s.index, "sim_expanded"].to_numpy()
    s["path"] = [str(T / f"{i}.jpg") for i in s.index]
    s = s[[Path(x).exists() for x in s.path]]
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    V = enc.images([rendition(Image.open(x)) for x in s.path])
    TX = _texts(enc)
    for c in IMDB_LOOKS:
        s[f"s_{c}"] = V @ TX[f"{c}|look"]; s[f"salt_{c}"] = V @ TX[f"{c}|alt"]
    s.to_parquet(OUT / "imdb.parquet")
    print("IMDB_DONE", len(s))


# ------------------------------------------------------------------------------------------------------- analysis
def load():
    fs = sorted(OUT.glob("part*.parquet")) + sorted((OUT / "pp").glob("*.parquet"))
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(["person", "cond", "item_id"])
    ids, V = [], []
    for f in sorted(OUT.glob("vec*.npz")):
        z = np.load(f); ids += list(z["ids"]); V.append(z["V"].astype(np.float32))
    V = np.concatenate(V); row = {i: r for r, i in enumerate(ids)}
    D = D[D.item_id.isin(row)].reset_index(drop=True)
    TX = np.load(OUT / "texts.npz")
    vi = V[[row[i] for i in D.item_id]]
    D["s"] = np.nan; D["s_alt"] = np.nan
    for c in COND:
        m = (D.cond == c).to_numpy()
        D.loc[m, "s"] = vi[m] @ TX[f"{c}|look"]; D.loc[m, "s_alt"] = vi[m] @ TX[f"{c}|alt"]
    D["y"] = D.p >= ACCEPT
    return D


def _auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cp(m, n):
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    return (0.0 if m == 0 else beta.ppf(0.025, m, n - m + 1)), (1.0 if m == n else beta.ppf(0.975, m + 1, n - m))


def _fmt(m, n):
    if n == 0:
        return "-"
    lo, hi = _cp(m, n)
    return f"{int(m)}/{n} = {m / n:.2f} [{lo:.2f}-{hi:.2f}]"


def searches(D, col="s"):
    rows = []
    for (per, c), g in D.groupby(["person", "cond"], sort=True):
        g = g.sort_values(col, ascending=False, kind="stable")
        y, s = g.y.to_numpy(), g[col].to_numpy()
        jy = g.sort_values("p", ascending=False).y.to_numpy()
        r = dict(person=per, cond=c, n=len(g), K=int(y.sum()), s1=s[0], s20=s[min(20, len(s)) - 1],
                 s50=s[min(50, len(s)) - 1], med=float(np.median(s)))
        for S in (20, 50):
            r[f"vec{S}"] = int(y[:S].sum()); r[f"jud{S}"] = int(jy[:S].sum()); r[f"cap{S}"] = min(S, len(g))
        r["likely_n"] = int((s[:20] >= T_PHOTO).sum()); r["likely_yes"] = int(y[:20][s[:20] >= T_PHOTO].sum())
        rows.append(r)
    R = pd.DataFrame(rows)
    R["enough"] = R.vec20 >= 18
    R["rule_vectors"] = R.s20 >= T_SCREEN
    return R


def analyze():
    from eval_vector_enough import _fit_threshold
    D = load()
    R = searches(D)
    R.to_csv(OUT / "searches.tsv", sep="\t", index=False, float_format="%.4f")
    Ralt = searches(D, "s_alt")
    G = pd.read_csv("eval/vector_enough/searches_rule.tsv", sep="\t", dtype={"library": str})
    L = ["M45: does RESULTS 45's vector-first rule hold INSIDE a face-filtered set? (eval/eval_vector_first_person.py)",
         f"DISBench: {D.person.nunique()} people (largest face groups of {D.user.nunique()} of 57 libraries with >= "
         f"{MIN_PHOTOS} face-matched photos), {D.groupby('person').item_id.nunique().sum()} photos, {len(COND)} conditions"
         f" = {len(R)} searches.",
         "Truth here = JUDGE-RELATIVE: Qwen3-VL-4B phone weights, person crop + red box, P(yes) >= 0.7, on every photo of",
         "the person. Eye labels (if any) further below. Vector = phone-sim PE-Core-B-16 whole-photo vector vs the look text.",
         "'judged top S' = the judge's own best S (by P) among the person's photos = what judging every photo would show.",
         "'enough' = >= 18 of the vector top 20 judge-yes (RESULTS 45's target).", ""]
    # per condition
    L += ["Per condition, pooled over people. precision = judge-yes share of the screen (photos), [95% Clopper-Pearson].",
          "| condition | people | photos per person (median) | judge-yes per person (median) | vector top 20 | judge's top 20 by P |"
          " vector top 50 | judge's top 50 by P | enough (>=18/20) | s20 median [min-max] | rule says vectors | of those enough |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in COND:
        g = R[R.cond == c]
        rv = g[g.rule_vectors]
        L.append(f"| {c} | {len(g)} | {g.n.median():.0f} | {g.K.median():.0f} | {_fmt(g.vec20.sum(), g.cap20.sum())} | "
                 f"{_fmt(g.jud20.sum(), g.cap20.sum())} | {_fmt(g.vec50.sum(), g.cap50.sum())} | "
                 f"{_fmt(g.jud50.sum(), g.cap50.sum())} | {g.enough.sum()}/{len(g)} | {g.s20.median():.4f} "
                 f"[{g.s20.min():.4f}-{g.s20.max():.4f}] | {len(rv)}/{len(g)} | {rv.enough.sum()}/{len(rv)} |")
    a = R
    L.append(f"| ALL | {len(a)} | {a.n.median():.0f} | {a.K.median():.0f} | {_fmt(a.vec20.sum(), a.cap20.sum())} | "
             f"{_fmt(a.jud20.sum(), a.cap20.sum())} | {_fmt(a.vec50.sum(), a.cap50.sum())} | {_fmt(a.jud50.sum(), a.cap50.sum())} | "
             f"{a.enough.sum()}/{len(a)} | {a.s20.median():.4f} [{a.s20.min():.4f}-{a.s20.max():.4f}] | "
             f"{a.rule_vectors.sum()}/{len(a)} | {a[a.rule_vectors].enough.sum()}/{a.rule_vectors.sum()} |")
    big = R[R.K >= 20]
    L += ["", f"Searches where the person has >= 20 judge-yes photos (a full right screen exists): {len(big)}/{len(R)}; vector top 20 "
          f"{_fmt(big.vec20.sum(), big.cap20.sum())}; enough {big.enough.sum()}/{len(big)}; by condition: "
          + ", ".join(f"{c} {len(g)} ({g.vec20.sum()}/{g.cap20.sum()})" for c, g in big.groupby('cond'))]
    # the rule as shipped
    t, p = R.enough.to_numpy(), R.rule_vectors.to_numpy()
    L += ["", f"THE SHIPPED RULE (s20 >= {T_SCREEN}) on these {len(R)} person searches:",
          f"  says vectors {p.sum()}/{len(R)}; of those truly enough {(p & t).sum()}/{p.sum()}; misses enough {(~p & t).sum()}/{t.sum()}; "
          f"accuracy {((p == t).sum())}/{len(R)}",
          f"  vector top-20 judge-rel. precision when it says vectors: {_fmt(R[R.rule_vectors].vec20.sum(), R[R.rule_vectors].cap20.sum())}"]
    # score distribution vs RESULTS 45
    L += ["", "SCORE SCALE: s20 (20th-best look score in scope) and s1, person sets vs RESULTS 45 whole-library searches",
          f"  RESULTS 45 (216 searches): s20 median {G.s20.median():.4f} [p10 {G.s20.quantile(.1):.4f}, p90 {G.s20.quantile(.9):.4f}]"
          f"; among 'enough' searches median {G[G.vec20_jrel >= 0.9].s20.median():.4f}; s1 median {G.s1.median():.4f}",
          f"  person sets ({len(R)}): s20 median {R.s20.median():.4f} [p10 {R.s20.quantile(.1):.4f}, p90 {R.s20.quantile(.9):.4f}]"
          f"; among 'enough' median {R[R.enough].s20.median() if R.enough.any() else float('nan'):.4f}; s1 median {R.s1.median():.4f}",
          f"  share of person searches above {T_SCREEN}: {(R.s20 >= T_SCREEN).sum()}/{len(R)}; per-photo scores above {T_PHOTO}: "
          f"{(D.s >= T_PHOTO).sum()}/{len(D)} photo-condition pairs"]
    # does s20 separate enough at all, inside person sets? (AUC + best threshold, leave-one-library-out)
    from scipy.stats import mannwhitneyu
    if R.enough.any() and (~R.enough).any():
        U = mannwhitneyu(R[R.enough].s20, R[~R.enough].s20).statistic / (R.enough.sum() * (~R.enough).sum())
        L.append(f"  s20 as a predictor of 'enough' inside person sets: AUC {U:.3f} ({R.enough.sum()} enough vs {(~R.enough).sum()} not)")
        R["library"] = R.person.str.split("#").str[0]
        pred = np.zeros(len(R), bool); Ts = []
        for lib in R.library.unique():
            tr, te = R.library != lib, R.library == lib
            T = _fit_threshold(R.loc[tr, "s20"].to_numpy(), R.loc[tr, "enough"].to_numpy(), 0.9)
            Ts.append(T); pred[te.to_numpy()] = R.loc[te, "s20"].to_numpy() >= T
        fin = [x for x in Ts if np.isfinite(x)]
        L.append(f"  refitted 'safe' threshold leave-one-library-out: {len(fin)}/{len(Ts)} folds find one "
                 f"({(np.median(fin) if fin else float('nan')):.4f} [{(min(fin) if fin else float('nan')):.4f}-"
                 f"{(max(fin) if fin else float('nan')):.4f}]); says vectors {pred.sum()}, of which enough "
                 f"{(pred & R.enough.to_numpy()).sum()}")
        Tall = _fit_threshold(R.s20.to_numpy(), R.enough.to_numpy(), 0.9)
        L.append(f"  refitted on all person searches: T = {Tall:.4f}")
    # per photo: likelyAtOnce
    L += ["", f"PER PHOTO (likelyAtOnce, own score >= {T_PHOTO}, of the vector top 20):"]
    for c in COND:
        g = R[R.cond == c]
        L.append(f"  {c:10s} shown {g.likely_n.sum():4d} of {g.cap20.sum()} slots; judge-yes {_fmt(g.likely_yes.sum(), g.likely_n.sum())}")
    L.append(f"  ALL        shown {R.likely_n.sum()} of {R.cap20.sum()}; judge-yes {_fmt(R.likely_yes.sum(), R.likely_n.sum())}")
    # within-person ranking quality (AUC of the look score vs judge-yes) + alternative phrasing
    L += ["", "Within-person ranking: AUC of the look score for judge-yes, median over people (people with >= 3 yes and >= 3 no):"]
    for c in COND:
        a1, a2 = [], []
        for per, g in D[D.cond == c].groupby("person"):
            if g.y.sum() >= 3 and (~g.y).sum() >= 3:
                a1.append(_auc(g.s.to_numpy(), g.y.to_numpy())); a2.append(_auc(g.s_alt.to_numpy(), g.y.to_numpy()))
        ga = Ralt[Ralt.cond == c]
        L.append(f"  {c:10s} planner look '{COND[c][0]}': AUC {np.median(a1) if a1 else float('nan'):.3f} ({len(a1)} people); "
                 f"alt '{COND[c][2]}': AUC {np.median(a2) if a2 else float('nan'):.3f}, vector top 20 {_fmt(ga.vec20.sum(), ga.cap20.sum())}, "
                 f"s20 median {ga.s20.median():.4f}")
    L += eye_section(D, R)
    # IMDB eras
    ip = OUT / "imdb.parquet"
    if ip.exists():
        S = pd.read_parquet(ip)
        S = S[S.sim_expanded >= 0.40]
        L += ["", "IMDB-WIKI eras (RESULTS 10/18; face-filtered as RESULTS 18; truth = era, a proxy; judge = stored P of",
              "RESULTS 10 judge, 'face heavier/fuller' and 'fit'):",
              "| person | look | photos (heavy/lean) | vector top 20: right era | judge top 20: right era | AUC vector | AUC judge | s20 |",
              "|---|---|---|---|---|---|---|---|"]
        for name, g in S.groupby("name"):
            for c, want, pj in (("heavier", "heavy", "p_heavier"), ("fit", "lean", "p_fit")):
                y = (g.era == want).to_numpy()
                v = g.sort_values(f"s_{c}", ascending=False); j = g.sort_values(pj, ascending=False)
                n = min(20, len(g))
                L.append(f"| {name} | {c} | {len(g)} ({(g.era == 'heavy').sum()}/{(g.era == 'lean').sum()}) | "
                         f"{_fmt((v.era == want).head(20).sum(), n)} | {_fmt((j.era == want).head(20).sum(), n)} | "
                         f"{_auc(g[f's_{c}'].to_numpy(), y):.3f} | {_auc(g[pj].to_numpy(), y):.3f} | "
                         f"{np.sort(g[f's_{c}'].to_numpy())[::-1][n - 1]:.4f} |")
    txt = "\n".join(L)
    (OUT / "report_auto.txt").write_text(txt)
    print(txt)


# ---------------------------------------------------------------------------------------------------------- eye
def eye_labels():
    """labels_<cond>.txt written blind from the sheets -> {(person, cond, item_id): match|no_match|unsure} + key rows."""
    kp = OUT / "key.json"
    if not kp.exists():
        return {}, []
    key = json.load(open(kp)); by = {(k["sheet"], k["number"]): k for k in key}
    out = {}
    for f in sorted(OUT.glob("labels_*.txt")):
        for line in open(f):
            if not line.strip() or line.startswith("#"):
                continue
            sh, n, v = line.split()[:3]
            assert v in ("match", "no_match", "unsure"), line
            k = by[(sh if sh.endswith(".jpg") else sh + ".jpg", int(n))]
            out[(k["person"], k["cond"], k["item_id"])] = (v, line.split(None, 3)[3].strip() if len(line.split()) > 3 else "")
    return out, key


def eye_section(D, R):
    """Eye precision of the vector-only screen vs the judge's screen, from the blind stratified sample: strata
    vec_yes / vec_no (uniform draws from the pooled vector top 20, split by the judge) and judge_only_yes (judge-yes
    photos of the judge's top 20 that are NOT in the vector top 20). unsure = not a match (RESULTS 34 convention)."""
    lab, key = eye_labels()
    if not lab:
        return ["", "EYE: no labels yet."]
    L = ["", f"EYE (blind, native pixels; {len(lab)} photos; protocol LABEL_PROTOCOL.txt; unsure counted as not a match;",
         " sample drawn from the people available when the sheets were made, see report header):",
         "| condition | eye rate: vec top-20 & judge-yes | vec top-20 & judge-no | judge-only yes | vector top 20, eye est. |"
         " judge's screen (its yes photos, up to 20), eye est. | photos on judge's screen / on vector screen |",
         "|---|---|---|---|---|---|---|"]
    st = {}
    for k in key:
        v = lab.get((k["person"], k["cond"], k["item_id"]))
        if v is None:
            continue
        t = st.setdefault((k["cond"], k["stratum"]), [0, 0, 0])
        t[0] += v[0] == "match"; t[1] += 1; t[2] += v[0] == "unsure"
    tot_v, tot_j = [0.0, 0], [0.0, 0]
    for c in COND:
        r = {s_: st.get((c, s_), [0, 0, 0]) for s_ in ("vec_yes", "vec_no", "judge_only_yes")}
        rate = {s_: (x[0] / x[1] if x[1] else float("nan")) for s_, x in r.items()}
        g = R[R.cond == c]
        ny, nn = g.vec20.sum(), g.cap20.sum() - g.vec20.sum()
        ev = (ny * rate["vec_yes"] + nn * (rate["vec_no"] if nn else 0)) / max(ny + nn, 1)
        # the judge's screen: its first min(20, K) yes photos by P; those also in the vector top 20 take vec_yes's rate
        Dc = D[D.cond == c]
        top = Dc.sort_values("s", ascending=False).groupby("person").head(20)
        vk = set(zip(top.person, top.item_id))
        js = Dc[Dc.y].sort_values("p", ascending=False).groupby("person").head(20)
        inv = np.array([(a, b) in vk for a, b in zip(js.person, js.item_id)], bool)
        jo_rate = rate["judge_only_yes"] if not np.isnan(rate["judge_only_yes"]) else rate["vec_yes"]
        ej = (inv.sum() * rate["vec_yes"] + (~inv).sum() * jo_rate) / max(len(js), 1)
        tot_v[0] += ev * (ny + nn); tot_v[1] += ny + nn; tot_j[0] += ej * len(js); tot_j[1] += len(js)
        f = lambda x: f"{x[0]}/{x[1]}" + (f" ({x[2]} unsure)" if x[2] else "")      # noqa: E731
        L.append(f"| {c} | {f(r['vec_yes'])} | {f(r['vec_no'])} | {f(r['judge_only_yes'])} | {ev:.2f} "
                 f"(~{ev * (ny + nn):.0f} of {ny + nn}) | {ej:.2f} (~{ej * len(js):.0f} of {len(js)}) | {len(js)} / {ny + nn} |")
    L.append(f"| ALL | | | | {tot_v[0] / tot_v[1]:.2f} (~{tot_v[0]:.0f} of {tot_v[1]}) | {tot_j[0] / tot_j[1]:.2f} "
             f"(~{tot_j[0]:.0f} of {tot_j[1]}) | |")
    notes = [(k, v) for k, v in lab.items() if re.search(r"\bdog\b", v[1], re.I)]
    L.append(f"  Face-filter purity seen on the way: {len(notes)}/{len(lab)} sampled boxes are a DOG's face, not a person "
             f"({', '.join(sorted({k[0] for k, _ in notes}))}).")
    return L


def sheets():
    """Blind sheets for eye labels: per condition, uniform draws from the pooled vector top 20, split by the judge's
    answer (N_EYE per stratum), shown as the full photo at native pixels with a red box on the person's face."""
    from PIL import Image, ImageDraw, ImageFont
    from findpics.vlm import draw_box
    n_eye = int(_arg("n", "10"))
    conds = (_arg("conds") or ",".join(COND)).split(",")
    D = load()
    rng = np.random.default_rng(45)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    top = D.sort_values("s", ascending=False).groupby(["person", "cond"]).head(20)
    jtop = D.sort_values("p", ascending=False).groupby(["person", "cond"]).head(20)
    vkey = set(zip(top.person, top.cond, top.item_id))
    jonly = jtop[jtop.y & ~pd.Series([k in vkey for k in zip(jtop.person, jtop.cond, jtop.item_id)], index=jtop.index)]
    items = []
    for c in conds:
        g = top[top.cond == c]
        for st, h, n in (("vec_yes", g[g.y], n_eye), ("vec_no", g[~g.y], n_eye),
                         ("judge_only_yes", jonly[jonly.cond == c], int(_arg("nj", "6")))):
            for _, r in h.iloc[rng.choice(len(h), size=min(n, len(h)), replace=False)].iterrows():
                r = r.copy(); r["stratum"] = st
                items.append(r)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    key = []
    for c in conds:
        its = [r for r in items if r.cond == c]
        its = [its[k] for k in rng.permutation(len(its))]
        ims = []
        for r in its:
            im = Image.open(r.path).convert("RGB")
            sx, sy = im.width / r.img_w, im.height / r.img_h
            ims.append((r, draw_box(im, (r.x1 * sx, r.y1 * sy, r.x2 * sx, r.y2 * sy))))
        s = 0
        for orient in ("land", "port"):
            o = [(r, im) for r, im in ims if (im.width >= im.height) == (orient == "land")]
            cols, rows = (2, 3) if orient == "land" else (3, 2)
            for b in range(0, len(o), 6):
                chunk = o[b:b + 6]
                W = max(im.width for _, im in chunk); H = max(im.height for _, im in chunk)
                sheet = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * H + (rows - 1) * 6), (255, 255, 255))
                dr = ImageDraw.Draw(sheet)
                s += 1; name = f"{c}_{s:02d}.jpg"
                for k, (r, im) in enumerate(chunk):
                    x, y = (k % cols) * (W + 6), (k // cols) * (H + 6)
                    sheet.paste(im, (x, y))
                    dr.rectangle([x, y, x + 24, y + 26], fill=(255, 255, 0))
                    dr.text((x + 6, y + 1), str(k + 1), fill=(0, 0, 0), font=font)
                    key.append(dict(sheet=name, number=k + 1, cond=c, person=r.person, item_id=r.item_id, stratum=r.stratum))
                sheet.save(OUT / "sheets" / name, quality=95)
        print(c, len(its), "photos,", s, "sheets", flush=True)
    json.dump(key, open(OUT / "key.json", "w"), indent=1)


if __name__ == "__main__":
    {"gen": gen, "imdb": imdb, "analyze": analyze, "sheets": sheets}[sys.argv[1]]()
