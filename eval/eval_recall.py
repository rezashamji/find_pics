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


# ---- rejected-side strata (per query, pooled over the 4 libraries) ----
# A: judge "near miss"   0.05 <= P < 0.7          (all of them if few; else a random draw)
# B: high fast score     P < 0.05 and look rank in the top 5% of this library for this query
# C: the rest            P < 0.05, look rank below the top 5%      (uniform random draw)
N_DRAW = {"A": 40, "B": 30, "C": 40}


def strata(df):
    df = df.copy()
    df["look_pct"] = df.groupby(["user", "query"]).look.rank(pct=True, ascending=False)
    df["stratum"] = np.where(df.p >= ACCEPT, "kept",
                             np.where(df.p >= 0.05, "A", np.where(df.look_pct <= 0.05, "B", "C")))
    return df


def load_scores():
    return strata(pd.concat([pd.read_parquet(f) for f in sorted((OUT / "scores").glob("part*.parquet"))],
                            ignore_index=True))


def tile(path, num, font, T=512):
    from PIL import Image, ImageDraw
    from findpics.media import load_image
    im = load_image(path, max_side=896)          # the judge's own input size (vlm._data_url max 896); DISBench ~500 px
    canvas = Image.new("RGB", (max(T, im.width), max(T, im.height)), (40, 40, 40))
    canvas.paste(im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 60, 46], fill=(255, 255, 0)); d.text((8, 2), str(num), fill=(0, 0, 0), font=font)
    return canvas


def sheets(items, prefix, font):
    """2x2 sheets, every tile at native pixels (no upscaling, no thumbnailing below the judge's input)."""
    from PIL import Image
    key = []
    for s in range(0, len(items), 4):
        tiles = [tile(r.path, j + 1, font) for j, r in enumerate(items[s:s + 4])]
        W = max(t.width for t in tiles); H = max(t.height for t in tiles)
        sheet = Image.new("RGB", (2 * W, 2 * H), (0, 0, 0))
        for j, t in enumerate(tiles):
            sheet.paste(t, ((j % 2) * W, (j // 2) * H))
        name = f"{prefix}_{s // 4 + 1:02d}.jpg"
        sheet.save(OUT / "sheets" / name, quality=92)
        key += [dict(sheet=name, number=j + 1, item_id=r.item_id, user=r.user, query=r.query, stratum=r.stratum)
                for j, r in enumerate(items[s:s + 4])]
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
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    key, counts = [], []
    for q in QUERIES:
        d = df[df["query"] == q]
        picks = []
        # kept side: every kept photo without an existing eye label (so precision is not borrowed from other libraries)
        kept = d[(d.stratum == "kept") & ~d.item_id.map(lambda i: (q, i) in lab)]
        picks.append(kept)
        for s, n in N_DRAW.items():
            pool = d[d.stratum == s]
            picks.append(pool.iloc[rng.choice(len(pool), size=min(n, len(pool)), replace=False)] if len(pool) else pool)
        counts.append(dict(query=q, **{f"N_{s}": int((d.stratum == s).sum()) for s in ["kept", "A", "B", "C"]},
                           **{f"n_{s}": len(p) for s, p in zip(["kept_unlabeled", "A", "B", "C"], picks)}))
        allp = pd.concat(picks, ignore_index=True)
        allp = allp.iloc[rng.permutation(len(allp))]        # strata mixed on the sheets: the labeler cannot tell
        key += sheets(list(allp.itertuples()), q.replace(" ", "_"), font)
    json.dump(key, open(OUT / "key.json", "w"), indent=1)   # holds the stratum: do NOT read before labeling
    pd.DataFrame(counts).to_csv(OUT / "sample_counts.csv", index=False)
    print(pd.DataFrame(counts).to_string())


def _labels_eye():
    """eval/recall_audit/labels.txt: '<sheet> <number> <match|no_match|unsure> <note>' written blind."""
    key = {(r["sheet"], r["number"]): r for r in json.load(open(OUT / "key.json"))}
    out = {}
    for line in open(OUT / "labels.txt"):
        if not line.strip() or line.startswith("#"):
            continue
        sh, n, v = line.split()[:3]
        r = key[(sh, int(n))]
        out[(r["query"], r["item_id"])] = v
    return out


def report():
    sys.path.insert(0, "eval")
    from eval_question_variants import labels
    df = load_scores()
    old = {k: {"right": "match", "wrong": "no_match", "unsure": "unsure"}[v] for k, v in labels().items()}
    eye = _labels_eye()
    rng = np.random.default_rng(0)
    B = 4000
    res, boot_found, boot_miss = [], np.zeros(B), np.zeros(B)
    for unsure_as in ["no_match", "match"]:
        pass
    lines = []
    for q in QUERIES:
        d = df[df["query"] == q]
        r = dict(query=q, library=int(d.drop_duplicates("user").library.sum()))
        # kept side: every kept photo has a label (existing eye label first, else this audit's)
        kd = d[d.stratum == "kept"]
        kl = [old.get((q, i)) or eye.get((q, i)) for i in kd.item_id]
        r["kept"] = len(kd); r["kept_labeled"] = sum(v is not None for v in kl)
        r["kept_right"] = sum(v == "match" for v in kl); r["kept_unsure"] = sum(v == "unsure" for v in kl)
        r["from_old_labels"] = sum((q, i) in old for i in kd.item_id)
        found_b = np.full(B, float(r["kept_right"]))
        miss_b = np.zeros(B); miss_hat = 0.0; miss_hi_cp = 0.0
        for s in ["A", "B", "C"]:
            ds = d[d.stratum == s]
            L = [eye.get((q, i)) for i in ds.item_id]
            Ls = [v for v in L if v is not None]
            N, n, m = len(ds), len(Ls), sum(v == "match" for v in Ls)
            r[f"{s}_N"], r[f"{s}_n"], r[f"{s}_miss"], r[f"{s}_unsure"] = N, n, m, sum(v == "unsure" for v in Ls)
            if n:
                miss_hat += N * m / n
                x = np.array([v == "match" for v in Ls], float)
                miss_b += N * x[rng.integers(0, n, size=(B, n))].mean(1)
        r["miss_hat"] = round(miss_hat, 1)
        r["recall"] = r["kept_right"] / (r["kept_right"] + miss_hat) if r["kept_right"] + miss_hat else float("nan")
        rb = found_b / np.maximum(found_b + miss_b, 1e-9)
        r["lo"], r["hi"] = np.percentile(rb, [2.5, 97.5])
        boot_found += found_b; boot_miss += miss_b
        res.append(r)
    R = pd.DataFrame(res)
    F, M = R.kept_right.sum(), R.miss_hat.sum()
    pooled = boot_found / (boot_found + boot_miss)
    print(R.to_string())
    print(f"POOLED: found {F}, est. missed {M:.1f}, recall {F / (F + M):.3f} "
          f"[{np.percentile(pooled, 2.5):.3f}, {np.percentile(pooled, 97.5):.3f}]")
    R.to_csv(OUT / "recall.csv", index=False)


if __name__ == "__main__":
    {"gen": gen, "sample": sample, "report": report}[sys.argv[1]]()
