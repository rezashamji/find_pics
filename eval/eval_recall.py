"""RECALL of the phone's photo judge on real libraries: of all real matches in a library, how many does exhaustive mode
keep? (Precision was measured by eye in RESULTS 26-28/32; recall never was.)

Libraries: the 4 DISBench users (section 26's 8 libraries) with the most existing eye labels on these 6 queries, so the
kept side reuses those labels. Queries/questions: the planner's plans from the everyday runs (eval/everyday16_q3vl).
Judge: Qwen3-VL-4B at the phone's 4-bit weights (models/qwen3vl_4b_mlx4sim), P(yes) >= 0.7 (the app's cut), on EVERY
in-scope photo (= the last round of exhaustive mode). Also stores each photo's fast score (image-vector similarity to the
plan's looks, engine.look_scores) for stratified sampling of the rejected side.

  gen   (vLLM env, GPU): python eval/eval_recall.py gen --shard=k/K      -> eval/recall_audit/scores/part{k}.parquet
  sample (CPU): python eval/eval_recall.py sample   -> blind 2x2 sheets (native pixels) + key (judge scores hidden)
  report (CPU): python eval/eval_recall.py report   -> per-query and pooled recall with 95% intervals
DISBench images are public Flickr data: the sheets are never committed (eval/recall_audit/*.jpg is git-ignored).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

USERS = ["47642109@N04", "22736462@N07", "10299779@N03", "28495173@N00"]
QUERIES = ["photos with a dog", "photos with a car", "photos with a bicycle", "beach photos", "sunset photos",
           "food photos"]
PLANS = "eval/everyday16_q3vl"
MODEL = os.path.abspath("models/qwen3vl_4b_mlx4sim")
ACCEPT = 0.7
OUT = Path("eval/recall_audit")


def plans():
    from findpics.converse import Plan
    import glob
    ref = [r for f in sorted(glob.glob(f"{PLANS}/part*.json")) for r in json.load(open(f))]
    return {q: Plan.model_validate(next(r["plan"] for r in ref if r["query"] == q)) for q in QUERIES}


def _arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def gen():
    """--model=<dir> --tag=<name>: another judge on the same photos/questions -> eval/recall_audit/scores_<tag>/ (the
    strata stay those of the default judge's scores/, so its labels score any judge; see score_set)."""
    model = os.path.abspath(_arg("model", MODEL))
    tag = _arg("tag")
    os.environ["FP_VLM_MODEL"] = model
    from findpics import store
    from findpics.engine import _judge_rows, look_scores, scope_mask
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    P = plans()
    pairs = [(u, q) for u in USERS for q in QUERIES][k::K]
    J = VLLMJudge(model=model, gpu_mem=0.8)
    out = []
    for u, q in pairs:
        sub = store.subset(idx, np.where(user_of == u)[0])
        spec = P[q].albums[0].model_copy(deep=True); spec.person = None
        rows = np.where(scope_mask(sub, spec))[0]
        look = look_scores(sub, enc, spec.looks, spec.avoid)
        p = _judge_rows(sub, J, rows, np.full(len(rows), -1), spec.judge_question)
        out.append(pd.DataFrame(dict(user=u, query=q, question=spec.judge_question, library=sub.n_items,
                                     item_id=sub.items.item_id.astype(str).to_numpy()[rows],
                                     path=sub.items.path.to_numpy()[rows], look=look[rows], p=p)))
        print(f"{u} {q!r}: in scope {len(rows)}/{sub.n_items}, kept {(p >= ACCEPT).sum()}", flush=True)
    d = OUT / ("scores" if tag is None else f"scores_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(d / f"part{k}.parquet")


# ---- strata (per query, pooled over the 4 libraries; sampling uniform WITHIN a stratum) ----
# kept: P >= 0.7. Labeled = existing eye label (exact); the unlabeled rest -> a uniform draw (precision of the rest).
# A "doubt":   rejected, and 0.05 <= P < 0.7, or another judge's exhaustive set kept it (9B 16-bit everyday_v2, 9B/4B
#              4-bit everyday_q9b/q4b on the same libraries): where misses should concentrate.
# B "similar": rejected, P < 0.05, not in A, image-vector similarity to the plan's looks in the top 5% of the library.
# C "rest":    everything else rejected (the bulk, ~7,000 per query): uniform draw.
N_DRAW = {"kept": 20, "A": 30, "B": 30, "C": 80}
AUX = ["eval/everyday_v2", "eval/everyday_q9b", "eval/everyday_q4b"]


def strata(df):
    import glob
    aux = {(r["query"], str(i)) for d in AUX for f in glob.glob(f"{d}/part*.json") for r in json.load(open(f))
           for i in (r["exhaustive"] or [])}
    df = df.copy()
    df["aux"] = [(q, i) in aux for q, i in zip(df["query"], df.item_id)]
    df["look_pct"] = df.groupby(["user", "query"]).look.rank(pct=True, ascending=False)
    df["stratum"] = np.select([df.p >= ACCEPT, (df.p >= 0.05) | df.aux, df.look_pct <= 0.05], ["kept", "A", "B"], "C")
    return df


def load_scores():
    return strata(pd.concat([pd.read_parquet(f) for f in sorted((OUT / "scores").glob("part*.parquet"))],
                            ignore_index=True))


def sheets(items, prefix, font):
    """Native pixels (DISBench ~500 px, which the judge also sees: vlm._data_url only shrinks above 896 px), 6 per sheet:
    landscape 2 cols x 3 rows, portrait 3 x 2 (~1.1 MP, under the viewer's downscale limit). Tiny number top-left."""
    from PIL import Image, ImageDraw
    from findpics.media import load_image
    key, s = [], 0
    for orient in ("land", "port"):
        its = [(r, load_image(r.path, max_side=896)) for r in items]
        its = [(r, im) for r, im in its if (im.width >= im.height) == (orient == "land")]
        cols, rows = (2, 3) if orient == "land" else (3, 2)
        for b in range(0, len(its), 6):
            chunk = its[b:b + 6]
            W = max(im.width for _, im in chunk); H = max(im.height for _, im in chunk)
            sheet = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * H + (rows - 1) * 6), (255, 255, 255))
            d = ImageDraw.Draw(sheet)
            s += 1; name = f"{prefix}_{s:02d}.jpg"
            for j, (r, im) in enumerate(chunk):
                x, y = (j % cols) * (W + 6), (j // cols) * (H + 6)
                sheet.paste(im, (x, y))
                d.rectangle([x, y, x + 26, y + 28], fill=(255, 255, 0)); d.text((x + 6, y + 1), str(j + 1), fill=(0, 0, 0), font=font)
                key.append(dict(sheet=name, number=j + 1, item_id=r.item_id, user=r.user, query=r.query,
                                stratum=r.stratum, p=float(r.p)))
            sheet.save(OUT / "sheets" / name, quality=95)
    return key


def sample():
    from PIL import ImageFont
    sys.path.insert(0, "eval")
    from eval_question_variants import labels
    df = load_scores()
    lab = labels()
    rng = np.random.default_rng(7)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    key, counts = [], []
    for q in QUERIES:
        d = df[df["query"] == q].copy()
        d.loc[(d.stratum == "kept") & d.item_id.map(lambda i: (q, i) in lab), "stratum"] = "kept_labeled"
        picks = []
        for s, n in N_DRAW.items():
            pool = d[d.stratum == s]
            picks.append(pool.iloc[rng.choice(len(pool), size=min(n, len(pool)), replace=False)])
        counts.append(dict(query=q, **{f"N_{s}": int((d.stratum == s).sum()) for s in ["kept_labeled", "kept", "A", "B", "C"]},
                           **{f"n_{s}": len(p) for s, p in zip(N_DRAW, picks)}))
        allp = pd.concat(picks, ignore_index=True)
        allp = allp.iloc[rng.permutation(len(allp))]        # kept and rejected mixed: the labeler cannot tell which
        key += sheets(list(allp.itertuples()), q.replace(" ", "_"), font)
    json.dump(key, open(OUT / "key.json", "w"), indent=1)   # holds stratum + P: NOT read before labels.txt is written
    pd.DataFrame(counts).to_csv(OUT / "sample_counts.csv", index=False)
    print(pd.DataFrame(counts).to_string())


# round 2 (after round 1 showed the bulk stratum C sets the interval width: 80 per query cannot bound a rare miss
# rate in ~7,000 photos): more C, more of car's A (10/30 real in round 1), a few kept/B so the sheets stay mixed (blind).
N_DRAW2 = {"kept": 20, "A": 20, "B": 10, "C": 120}
N_DRAW2_A = {"photos with a car": 40}


def sample2():
    from PIL import ImageFont
    sys.path.insert(0, "eval")
    from eval_question_variants import labels
    df = load_scores()
    lab = labels()
    done = {(r["query"], r["item_id"]) for r in json.load(open(OUT / "key.json"))}
    rng = np.random.default_rng(11)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    key, counts = [], []
    for q in QUERIES:
        d = df[df["query"] == q].copy()
        d = d[~d.item_id.map(lambda i: (q, i) in done or (q, i) in lab)]
        picks = []
        for s, n in N_DRAW2.items():
            n = N_DRAW2_A.get(q, n) if s == "A" else n
            pool = d[d.stratum == s]
            picks.append(pool.iloc[rng.choice(len(pool), size=min(n, len(pool)), replace=False)])
        counts.append(dict(query=q, **{f"n_{s}": len(p) for s, p in zip(N_DRAW2, picks)}))
        allp = pd.concat(picks, ignore_index=True)
        allp = allp.iloc[rng.permutation(len(allp))]
        key += sheets(list(allp.itertuples()), "r2_" + q.replace(" ", "_"), font)
    json.dump(key, open(OUT / "key2.json", "w"), indent=1)   # NOT read before the round-2 labels are written
    print(pd.DataFrame(counts).to_string())


def _labels_eye():
    """eval/recall_audit/labels.txt: '<sheet> <number> <match|no_match|unsure> <note>', written blind."""
    key = {(r["sheet"], r["number"]): r for f in sorted(OUT.glob("key*.json")) for r in json.load(open(f))}
    out = {}
    for line in open(OUT / "labels.txt"):
        if not line.strip() or line.startswith("#"):
            continue
        sh, n, v = line.split()[:3]
        r = key[(sh, int(n))]
        out[(r["query"], r["item_id"])] = v
    return out


def _cp(m, n, a):
    from scipy.stats import beta
    lo = 0.0 if m == 0 else beta.ppf(a / 2, m, n - m + 1)
    hi = 1.0 if m == n else beta.ppf(1 - a / 2, m + 1, n - m)
    return lo, hi


def estimate(df, old, eye, unsure_match=False, B=4000, seed=0):
    """Per query: found = real matches kept, missed = real matches rejected, both stratified estimates (stratum size x
    labeled match rate in its uniform sample; a fully labeled stratum is exact).
    Intervals (95%): (1) stratified bootstrap: resample within each sampled stratum. Zero-width when a stratum has 0
    matches in its sample, i.e. it treats "0 of 80" as "exactly 0", so it is optimistic for the big rest stratum.
    (2) Bayesian: each sampled stratum's rate ~ Beta(m + 1/2, n - m + 1/2) (Jeffreys prior), so "0 of 80" still allows
    misses. (3) Conservative: every stratum at its Clopper-Pearson bound, Bonferroni over the query's strata.
    Pooled: (a) over all real matches of the 24 searches (dominated by the dog library), (b) mean of the 6 queries."""
    rng = np.random.default_rng(seed)
    ok = {"match", "unsure"} if unsure_match else {"match"}
    rows, bf, bm, jf, jm, rb_all, rj_all = [], np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B), [], []
    for q in QUERIES:
        d = df[df["query"] == q]
        r = dict(query=q)
        kd = d[d.stratum == "kept"]
        has_old = kd.item_id.map(lambda i: (q, i) in old)
        r["kept"] = len(kd)
        r["kept_old_labels"] = int(has_old.sum())
        old_right = sum(old[(q, i)] in ok for i in kd.item_id[has_old])
        r["kept_old_right"] = old_right
        # rejected photos that already had an eye label (earlier audits): exact, my label first; round 2 drew only
        # from the rest of each stratum, so the sampled part of a stratum is "stratum minus old-labeled" (uniform there)
        rej = d[d.stratum != "kept"]
        rold = rej.item_id.map(lambda i: (q, i) in old)
        old_miss = sum((eye.get((q, i)) or old[(q, i)]) in ok for i in rej.item_id[rold])
        r["rej_old_labels"], r["rej_old_match"] = int(rold.sum()), int(old_miss)
        rr = rej[~rold]
        parts = {"kept_rest": (kd[~has_old], True), "A": (rr[rr.stratum == "A"], False),
                 "B": (rr[rr.stratum == "B"], False), "C": (rr[rr.stratum == "C"], False)}
        f_hat, m_hat, f_lo, m_hi = float(old_right), float(old_miss), float(old_right), float(old_miss)
        f_b, m_b = np.full(B, float(old_right)), np.full(B, float(old_miss))
        f_j, m_j = np.full(B, float(old_right)), np.full(B, float(old_miss))
        k = sum(1 for p, _ in parts.values() if len(p))
        for s, (p, is_kept) in parts.items():
            L = [eye.get((q, i)) for i in p.item_id]
            L = [v for v in L if v is not None]
            N, n = len(p), len(L)
            x = np.array([v in ok for v in L], float)
            m = int(x.sum())
            r[f"{s}_N"], r[f"{s}_n"], r[f"{s}_match"] = N, n, m
            r[f"{s}_unsure"] = sum(v == "unsure" for v in L)
            if N == 0:
                continue
            if n == 0:
                raise SystemExit(f"{q} {s}: no labels")
            est = N * m / n
            if n == N:                                   # census: exact
                boot = jef = np.full(B, float(m)); lo = hi = m / n
            else:
                boot = N * x[rng.integers(0, n, size=(B, n))].mean(1)
                jef = N * rng.beta(m + 0.5, n - m + 0.5, size=B)
                lo, hi = _cp(m, n, 0.05 / k)
            if is_kept:
                f_hat += est; f_b += boot; f_j += jef; f_lo += N * lo
            else:
                m_hat += est; m_b += boot; m_j += jef; m_hi += N * hi
        r["found"] = round(f_hat, 1); r["missed"] = round(m_hat, 1)
        r["recall"] = f_hat / (f_hat + m_hat)
        rb, rj = f_b / (f_b + m_b), f_j / (f_j + m_j)
        r["boot_lo"], r["boot_hi"] = np.percentile(rb, [2.5, 97.5])
        r["bayes_lo"], r["bayes_hi"] = np.percentile(rj, [2.5, 97.5])
        r["cons_lo"] = f_lo / (f_lo + m_hi)
        bf += f_b; bm += m_b; jf += f_j; jm += m_j; rb_all.append(rb); rj_all.append(rj)
        rows.append(r)
    R = pd.DataFrame(rows)
    F, M = R.found.sum(), R.missed.sum()
    mb, mj = np.mean(rb_all, 0), np.mean(rj_all, 0)
    pooled = dict(all_found=F, all_missed=M, all_recall=F / (F + M),
                  all_boot=tuple(np.percentile(bf / (bf + bm), [2.5, 97.5])),
                  all_bayes=tuple(np.percentile(jf / (jf + jm), [2.5, 97.5])),
                  mean6_recall=R.recall.mean(), mean6_boot=tuple(np.percentile(mb, [2.5, 97.5])),
                  mean6_bayes=tuple(np.percentile(mj, [2.5, 97.5])))
    return R, pooled


def old_labels():
    """Earlier eye labels (eval_question_variants.labels) in this file's vocabulary."""
    sys.path.insert(0, "eval")
    from eval_question_variants import labels
    return {k: {"right": "match", "wrong": "no_match", "unsure": "unsure"}[v] for k, v in labels().items()}


def score_set(df, old, eye, keep, unsure_match=False, B=4000, seed=0):
    """Precision and recall of ANY returned set on the RESULTS 34 truth (Apple Photos, another judge, fast mode...).
    keep: {query: set of item_ids returned}. Same strata, labels and estimator as estimate(): a photo with an earlier
    eye label counts exactly (kept-stratum photos: the earlier label; rejected: my blind label first); every other
    photo stands in its stratum (kept_rest / A / B / C of the RESULTS 34 judge) for stratum size / labeled count
    photos (Horvitz-Thompson), for BOTH "real" and "real AND returned". recall = est. real returned / est. real.
    On the RESULTS 34 judge's own kept set this is exactly estimate() (score_apple_photos.py selftest).
    Intervals (95%): bootstrap within each stratum (resamples the labeled photos; joint for numerator and
    denominator); Bayesian: per stratum a Dirichlet(counts + 1/2) over {real&returned, real&not, not real&returned,
    not real&not}, cells that cannot occur (no photo of the stratum on that side of the returned set) left out, so
    it is estimate()'s Jeffreys Beta when a stratum is all-returned or none-returned."""
    rng = np.random.default_rng(seed)
    ok = {"match", "unsure"} if unsure_match else {"match"}
    rows = []
    for q in QUERIES:
        d = df[df["query"] == q]
        S = set(map(str, keep.get(q, ())))
        ids = d.item_id.to_numpy()
        ins = np.isin(ids, list(S))
        kept = (d.stratum == "kept").to_numpy()
        has_old = np.array([(q, i) in old for i in ids])
        lab_any = [eye.get((q, i)) or old.get((q, i)) for i in ids]
        r = dict(query=q, returned=len(S), returned_in_library=int(ins.sum()),
                 returned_not_in_library=len(S - set(ids)))
        Lr = [v for v, s in zip(lab_any, ins) if s and v is not None]
        r["returned_labelled"] = len(Lr)
        r["returned_labelled_match"] = sum(v == "match" for v in Lr)
        r["returned_labelled_unsure"] = sum(v == "unsure" for v in Lr)
        # exact part
        ex_lab = [old[(q, i)] if k else (eye.get((q, i)) or old[(q, i)]) for i, k in zip(ids[has_old], kept[has_old])]
        ex_real = np.array([v in ok for v in ex_lab], bool)
        t_hat = float(ex_real.sum()); f_hat = float((ex_real & ins[has_old]).sum())
        t_b, f_b = np.full(B, t_hat), np.full(B, f_hat)
        t_j, f_j = np.full(B, t_hat), np.full(B, f_hat)
        groups = {"kept_rest": kept & ~has_old}
        for s in ("A", "B", "C"):
            groups[s] = (d.stratum == s).to_numpy() & ~has_old
        for s, g in groups.items():
            N = int(g.sum())
            if N == 0:
                continue
            gi = np.where(g)[0]
            L = [(eye.get((q, ids[j])), ins[j]) for j in gi]
            L = [(v in ok, b) for v, b in L if v is not None]
            n = len(L)
            r[f"{s}_N"], r[f"{s}_returned"], r[f"{s}_n"] = N, int(ins[gi].sum()), n
            if n == 0:
                raise SystemExit(f"{q} {s}: no labels")
            y = np.array([a for a, _ in L], float); yr = np.array([a and b for a, b in L], float)
            t_hat += N * y.mean(); f_hat += N * yr.mean()
            if n == N:
                t_b += y.sum(); f_b += yr.sum(); t_j += y.sum(); f_j += yr.sum()
                continue
            ix = rng.integers(0, n, size=(B, n))
            t_b += N * y[ix].mean(1); f_b += N * yr[ix].mean(1)
            sides = [bool(ins[gi].any()), bool((~ins[gi]).any())]
            cells = [(rl, sd) for rl in (True, False) for sd, present in zip((True, False), sides) if present]
            cnt = np.array([sum(1 for a, b in L if a == rl and b == sd) for rl, sd in cells], float)
            th = rng.dirichlet(cnt + 0.5, size=B)
            t_j += N * th[:, [k for k, (rl, _) in enumerate(cells) if rl]].sum(1)
            f_j += N * th[:, [k for k, (rl, sd) in enumerate(cells) if rl and sd]].sum(1) if any(rl and sd for rl, sd in cells) else 0
        r["real_est"] = round(t_hat, 1); r["real_returned_est"] = round(f_hat, 1)
        r["recall"] = f_hat / t_hat if t_hat else float("nan")
        r["recall_boot_lo"], r["recall_boot_hi"] = np.percentile(f_b / t_b, [2.5, 97.5])
        r["recall_bayes_lo"], r["recall_bayes_hi"] = np.percentile(f_j / t_j, [2.5, 97.5])
        nr = max(r["returned_in_library"], 1)
        r["precision_est"] = f_hat / nr
        r["precision_boot_lo"], r["precision_boot_hi"] = np.percentile(f_b / nr, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


def report():
    sys.path.insert(0, "eval")
    from eval_question_variants import labels
    df = load_scores()
    old = {k: {"right": "match", "wrong": "no_match", "unsure": "unsure"}[v] for k, v in labels().items()}
    eye = _labels_eye()
    pd.set_option("display.width", 250)
    for um in (False, True):
        R, P = estimate(df, old, eye, unsure_match=um)
        print(f"\n== unsure counted as {'MATCH' if um else 'no match'}")
        print(R.round(3).to_string())
        print("POOLED", {k: (round(float(v), 3) if np.isscalar(v) else tuple(round(float(x), 3) for x in v)) for k, v in P.items()})
        R.to_csv(OUT / f"recall{'_unsure_match' if um else ''}.csv", index=False)
    # pooled conservative bound: Bonferroni over every sampled stratum of every query
    k = 0; f_lo = m_hi = 0.0
    parts = []
    for q in QUERIES:
        d = df[df["query"] == q]; kd = d[d.stratum == "kept"]
        has_old = kd.item_id.map(lambda i: (q, i) in old)
        f_lo += sum(old[(q, i)] == "match" for i in kd.item_id[has_old])
        rej = d[d.stratum != "kept"]; rold = rej.item_id.map(lambda i: (q, i) in old); rr = rej[~rold]
        m_hi += sum((eye.get((q, i)) or old[(q, i)]) == "match" for i in rej.item_id[rold])
        for s, p, is_kept in [("kept_rest", kd[~has_old], True), ("A", rr[rr.stratum == "A"], False),
                              ("B", rr[rr.stratum == "B"], False), ("C", rr[rr.stratum == "C"], False)]:
            L = [eye.get((q, i)) for i in p.item_id]; L = [v for v in L if v is not None]
            if len(p):
                parts.append((len(p), len(L), sum(v == "match" for v in L), is_kept))
    for N, n, m, is_kept in parts:
        lo, hi = _cp(m, n, 0.05 / len(parts))
        if is_kept:
            f_lo += N * lo
        else:
            m_hi += N * hi
    # the bulk strata C pooled over queries (one CP bound on their combined rate, same N-weighting since n_C/N_C is ~equal)
    print(f"POOLED conservative (Bonferroni over {len(parts)} strata): recall >= {f_lo / (f_lo + m_hi):.3f}")


if __name__ == "__main__":
    {"gen": gen, "sample": sample, "sample2": sample2, "report": report}[sys.argv[1]]()
