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


def gen():
    os.environ["FP_VLM_MODEL"] = MODEL
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
    J = VLLMJudge(model=MODEL, gpu_mem=0.8)
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
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(OUT / "scores" / f"part{k}.parquet")


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


def _labels_eye():
    """eval/recall_audit/labels.txt: '<sheet> <number> <match|no_match|unsure> <note>', written blind."""
    key = {(r["sheet"], r["number"]): r for r in json.load(open(OUT / "key.json"))}
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
    """Per query: found = real matches kept, missed = real matches rejected, both as stratified estimates (stratum size x
    labeled match rate in its uniform sample). 95% interval: stratified bootstrap (resample within each sampled stratum).
    Conservative bound: every sampled stratum at its Clopper-Pearson bound with Bonferroni over the strata of the query
    (missed at the upper bound, found at the lower)."""
    rng = np.random.default_rng(seed)
    ok = {"match", "unsure"} if unsure_match else {"match"}
    rows, bf, bm = [], np.zeros(B), np.zeros(B)
    for q in QUERIES:
        d = df[df["query"] == q]
        r = dict(query=q)
        kd = d[d.stratum == "kept"]
        has_old = kd.item_id.map(lambda i: (q, i) in old)
        r["kept"] = len(kd)
        r["kept_old_labels"] = int(has_old.sum())
        old_right = sum(old[(q, i)] in ok for i in kd.item_id[has_old])
        parts = {"kept_rest": (kd[~has_old], True), "A": (d[d.stratum == "A"], False), "B": (d[d.stratum == "B"], False),
                 "C": (d[d.stratum == "C"], False)}
        f_hat, m_hat, f_b, m_b, f_lo, m_hi = float(old_right), 0.0, np.full(B, float(old_right)), np.zeros(B), float(old_right), 0.0
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
            boot = N * x[rng.integers(0, n, size=(B, n))].mean(1)
            lo, hi = _cp(m, n, 0.05 / k)
            if is_kept:
                f_hat += est; f_b += boot; f_lo += N * lo
            else:
                m_hat += est; m_b += boot; m_hi += N * hi
        r["found"] = round(f_hat, 1); r["missed"] = round(m_hat, 1)
        r["recall"] = f_hat / (f_hat + m_hat)
        rb = f_b / (f_b + m_b)
        r["boot_lo"], r["boot_hi"] = np.percentile(rb, [2.5, 97.5])
        r["cons_lo"] = f_lo / (f_lo + m_hi)
        bf += f_b; bm += m_b
        rows.append(r)
    R = pd.DataFrame(rows)
    pb = bf / (bf + bm)
    pooled = dict(found=R.found.sum(), missed=R.missed.sum(), recall=R.found.sum() / (R.found.sum() + R.missed.sum()),
                  boot_lo=np.percentile(pb, 2.5), boot_hi=np.percentile(pb, 97.5))
    return R, pooled


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
        print("POOLED", {k: round(float(v), 3) for k, v in P.items()})
        R.to_csv(OUT / f"recall{'_unsure_match' if um else ''}.csv", index=False)
    # pooled conservative bound: Bonferroni over every sampled stratum of every query
    from eval_recall import QUERIES as _Q  # noqa: F401
    k = 0; f_lo = m_hi = 0.0
    parts = []
    for q in QUERIES:
        d = df[df["query"] == q]; kd = d[d.stratum == "kept"]
        has_old = kd.item_id.map(lambda i: (q, i) in old)
        f_lo += sum(old[(q, i)] == "match" for i in kd.item_id[has_old])
        for s, p, is_kept in [("kept_rest", kd[~has_old], True), ("A", d[d.stratum == "A"], False),
                              ("B", d[d.stratum == "B"], False), ("C", d[d.stratum == "C"], False)]:
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
    {"gen": gen, "sample": sample, "report": report}[sys.argv[1]]()
