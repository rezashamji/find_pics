"""Tile vectors for the phone's fast ranking (RESULTS 42; PLAN step 3 "per-crop vectors only where the cheap stage
loses photos"). Question: RESULTS 41's fast mode still misses small background objects because the one whole-photo
PE-Core-B-16 vector ranks them too deep (~65 of car's 410 judge-yes past rank 2,000; 7 of 48 bicycles past 1,500).
Do extra vectors per photo from TILES pull them into the judged head, and what does that cost on the phone?

Set: the RESULTS 34/40/41 set (4 DISBench libraries as ONE 7,886-photo library, 6 searches), judge = the real phone
weights' stored P(yes) on every photo (eval/recall_audit/scores_q3vl4b_real; no new judge calls), cut = judgeCutoff,
truth = RESULTS 34 eye labels (eval_recall.score_set).

Phone simulation of each photo (docs/PHONE_PARITY.md): PhotoKit's local rendition = long side 480 (Lanczos) + JPEG
round trip q80; every crop is cut from that 480 px image, then the phone's preprocessing (open_clip's PE-Core-B-16
transform = Pillow bilinear squash to 224, no centre crop) and fp16 image tower (ImageTextEncoder on GPU = fp16).
15 vectors per photo: 0 = whole, 1-4 = 2x2 grid, 5-13 = 3x3 grid, 14 = centre half (the middle 50% x 50%, i.e. a 2x
zoom; it is also "the centre crop" of the 2x2 + centre tiling). Text = the plan's look ("a dog", ...), as
eval/apple_photos/looks_b16.parquet (which embedded the ORIGINAL 500 px file, no JPEG round trip).

  embed    (GPU job via scripts/race_sbatch.sh) -> eval/tiles_phone/vectors.npy [N,15,D] fp16, ids.txt, texts.npz
  analyze  (CPU) -> eval/tiles_phone/report.txt + ranks.tsv + replay.tsv: per tiling, judge-yes counts in the top
           600 / 1,500 / 2,000, ranks of the yes photos, and the ADOPTED fast-mode rule (RESULTS 41: window 100,
           cap 2,000, auto round 2 at >= 3 random hits) replayed over the new ranking (20 seeds): judge calls,
           found / judge-yes, eye-label recall and precision.
  embed16  (GPU job) validation on the 16 unseen libraries of RESULTS 27/41 (ev16; yes = Qwen3-VL-4B 16-bit on every
           in-scope photo, eval/stop_rule/ev16_ranked.parquet): whole + 2x2 vectors (same phone simulation) of the
           30,273 photos + the plans' look/avoid texts -> eval/tiles_phone/ev16_vectors.npy, ev16_texts.npz
  val16    (CPU) adopted rule replayed on ev16 for whole vs 2x2 vs gated 2x2: found / judge-yes, judge calls
  sheets   (CPU) -> eval/tiles_phone/sheets/: judge-yes cars / bicycles the chosen tiling's fast mode finds and the
           single vector's does not (native pixels; view before any claim)
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/tiles_phone")
NV = 15
SL = dict(whole=[0], g2=list(range(1, 5)), g3=list(range(5, 14)), center=[14])
GATE_TEXTS = ["a busy street scene with many people and cars", "a wide outdoor view of a town",
              "a close-up photo of one object", "a portrait of a person"]


def rendition(im, side=480, q=80):
    from PIL import Image
    im = im.convert("RGB").copy(); im.thumbnail((side, side), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def crops(im):
    w, h = im.size; out = [im]
    for n in (2, 3):
        for i in range(n):
            for j in range(n):
                out.append(im.crop((round(j * w / n), round(i * h / n), round((j + 1) * w / n), round((i + 1) * h / n))))
    out.append(im.crop((round(w / 4), round(h / 4), round(3 * w / 4), round(3 * h / 4))))
    return out


def manifest():
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores").glob("part*.parquet"))])
    return sc.drop_duplicates("item_id")[["item_id", "path"]].reset_index(drop=True)   # same order as looks()


def embed():
    from PIL import Image
    from findpics.models import ImageTextEncoder
    from eval_recall import QUERIES, plans
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    print("device", enc.device, enc.dtype, flush=True)
    P = plans()
    looks = {q: P[q].albums[0].looks for q in QUERIES}
    np.savez(OUT / "texts.npz", **{f"q{i}": enc.texts(looks[q]) for i, q in enumerate(QUERIES)},
             gate=enc.texts(GATE_TEXTS))
    json.dump(dict(queries=QUERIES, looks=looks, gate=GATE_TEXTS), open(OUT / "texts.json", "w"), indent=1)
    V = None
    sizes = []
    for b in range(0, len(man), 32):
        ims = [rendition(Image.open(f)) for f in man.path[b:b + 32]]
        sizes += [im.size for im in ims]
        v = enc.images([c for im in ims for c in crops(im)])
        if V is None:
            V = np.zeros((len(man), NV, v.shape[1]), np.float16)
        V[b:b + len(ims)] = v.reshape(len(ims), NV, -1)
        if b % 1024 == 0:
            print(b, flush=True)
    np.save(OUT / "vectors.npy", V)
    (OUT / "ids.txt").write_text("\n".join(man.item_id.astype(str)) + "\n")
    s = np.array(sizes)
    print("EMBED_DONE", V.shape, "rendition long side min/median/max", s.max(1).min(), np.median(s.max(1)), s.max(1).max())


# ------------------------------------------------------------------------------------------------------------ analysis
def load():
    V = np.load(OUT / "vectors.npy").astype(np.float32)
    ids = (OUT / "ids.txt").read_text().split()
    T = np.load(OUT / "texts.npz")
    return V, np.array(ids), T


def tilings(S):
    """S [N,15] look scores of one query -> candidate rankings (higher = judged earlier)."""
    w = S[:, 0]
    g2, g3, c = S[:, SL["g2"]].max(1), S[:, SL["g3"]].max(1), S[:, 14]
    R = {"whole (phone sim)": w,
         "2x2": np.maximum(w, g2),
         "3x3": np.maximum(w, g3),
         "2x2+centre": np.maximum(w, np.maximum(g2, c)),
         "centre 2x zoom": np.maximum(w, c),
         "2x2+3x3+centre": np.maximum.reduce([w, g2, g3, c])}
    for name, t in (("2x2", g2), ("2x2+centre", np.maximum(g2, c)), ("3x3", g3)):     # weighted max: tiles discounted
        for d in (0.01, 0.02, 0.03):
            R[f"{name} -{d:.2f}"] = np.maximum(w, t - d)
        for a in (0.5,):
            R[f"{name} mix{a}"] = a * w + (1 - a) * np.maximum(w, t)
    return R


ADOPTED = "window=100 head_max=2000 more=hits>=3,max2"


def analyze():
    from eval_recall import QUERIES, _labels_eye, load_scores, old_labels, score_set
    from score_apple_photos import D, judge_cutoff
    from tune_stop_rule import parse_rule, simulate
    V, ids, T = load()
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores_q3vl4b_real").glob("part*.parquet"))])
    lk = pd.read_parquet(D / "looks_b16.parquet").set_index("item_id")
    df = load_scores(); old = old_labels(); eye_l = _labels_eye()
    rule = parse_rule(ADOPTED)
    seeds = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--seeds=")), 20))
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    gates = gate_masks(V)
    rank_rows, rep_rows, found_sets = [], [], {}
    cases = {}
    for qi, q in enumerate(QUERIES):
        d = sc[sc["query"] == q].set_index("item_id").loc[ids]
        cut = judge_cutoff(d.question.iloc[0])
        y = (d.p >= cut).to_numpy()
        S = (V @ T[f"q{qi}"].T).mean(2)                       # [N,15]: mean over the plan's looks (one look here)
        R = {"B-16 as RESULTS 40/41 (500 px original)": lk.loc[ids, q].to_numpy()}
        R.update(tilings(S))
        for gname, gm in gates.items():                      # tiles only on a subset of photos (storage/compute gate)
            for base in ("2x2", "2x2+centre"):
                R[f"{base} | gate {gname}"] = np.where(gm, R[base], S[:, 0])
        if only:
            R = {k: v for k, v in R.items() if k in only}
        cases[q] = (y, R)
        for name, s in R.items():
            order = np.argsort(-s, kind="stable")
            pos = np.empty(len(s), int); pos[order] = np.arange(len(s))
            r = np.sort(pos[y]) + 1
            rank_rows.append(dict(query=q, tiling=name, yes=int(y.sum()), top150=int((r <= 150).sum()),
                                  top600=int((r <= 600).sum()), top1500=int((r <= 1500).sum()),
                                  top2000=int((r <= 2000).sum()), past2000=int((r > 2000).sum()),
                                  median_rank=float(np.median(r)), p90_rank=float(np.percentile(r, 90))))
    RK = pd.DataFrame(rank_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    RK.to_csv(OUT / "ranks.tsv", sep="\t", index=False)
    names = list(cases[QUERIES[0]][1])
    for name in names:
        per = []
        for sd in range(seeds):
            keep, calls, rounds = {}, {}, {}
            for q in QUERIES:
                y, R = cases[q]
                order = np.argsort(-R[name], kind="stable")
                m, c, _, k, _ = simulate(y[order], rule, sd)
                keep[q] = set(ids[order][m]); calls[q] = c; rounds[q] = k
                found_sets[(name, q, sd)] = keep[q]
            E = score_set(df, old, eye_l, keep, B=10).set_index("query")
            for q in QUERIES:
                per.append(dict(tiling=name, seed=sd, query=q, found=len(keep[q]), calls=calls[q], rounds=rounds[q],
                                recall=E.loc[q, "recall"], precision=E.loc[q, "precision_est"],
                                real_found=E.loc[q, "real_returned_est"], real=E.loc[q, "real_est"]))
        rep_rows += per
        P = pd.DataFrame(per)
        pooled = P.groupby("seed").apply(lambda x: x.real_found.sum() / x.real.sum(), include_groups=False).mean()
        print(f"{name:45s} pooled eye recall {pooled:.3f} calls {P.calls.sum() / seeds:6.0f}", flush=True)
    RP = pd.DataFrame(rep_rows)
    RP.to_csv(OUT / "replay.tsv", sep="\t", index=False, float_format="%.4g")
    # exhaustive reference
    EX = score_set(df, old, eye_l, {q: set(ids[cases[q][0]]) for q in QUERIES}, B=10).set_index("query")
    import pickle
    pickle.dump({k: v for k, v in found_sets.items() if k[2] < 3}, open(OUT / "found_sets.pkl", "wb"))
    report(RK, RP, EX, QUERIES, seeds, gates)


def gate_masks(V):
    """Query-independent gates decided at index time. 'novel': tiles whose 2x2 vectors differ most from the whole
    photo (computing them is still needed; saves storage only). 'busy': the whole-photo vector alone (cheap: decides
    BEFORE any tile is computed) - busy/wide scene prompts minus close-up/portrait prompts."""
    T = np.load(OUT / "texts.npz")["gate"]
    w = V[:, 0]
    nov = 1 - (V[:, 1:5] * w[:, None]).sum(2).mean(1)
    g = w @ T.T
    busy = g[:, :2].max(1) - g[:, 2:].max(1)
    out = {}
    for frac in (0.25, 0.5):
        out[f"novel top{int(frac * 100)}%"] = nov >= np.quantile(nov, 1 - frac)
        out[f"busy top{int(frac * 100)}%"] = busy >= np.quantile(busy, 1 - frac)
    return out


def report(RK, RP, EX, QUERIES, seeds, gates):
    short = {q: q.replace("photos with a ", "").replace(" photos", "") for q in QUERIES}
    lines = [f"seeds {seeds}; rule {ADOPTED}; eye recall = RESULTS 34 truth (score_set, unsure = no match)",
             "exhaustive: " + " | ".join(f"{short[q]} {EX.loc[q, 'recall']:.3f}" for q in QUERIES) +
             f" | pooled {EX.real_returned_est.sum() / EX.real_est.sum():.3f}", ""]
    lines.append("== judge-yes photos by rank cutoff (top600 / top1500 / top2000 of yes; median rank)")
    for name, g in RK.groupby("tiling", sort=False):
        lines.append(f"{name:45s} " + " | ".join(
            f"{short[r.query]} {r.top600}/{r.top1500}/{r.top2000} of {r.yes} (med {r.median_rank:.0f})" for r in g.itertuples()))
    lines.append("\n== adopted fast rule replayed (mean of seeds): eye recall / judge calls / found; pooled")
    for name, P in RP.groupby("tiling", sort=False):
        pooled = P.groupby("seed").apply(lambda x: x.real_found.sum() / x.real.sum(), include_groups=False)
        g = P.groupby("query", sort=False).agg(recall=("recall", "mean"), calls=("calls", "mean"), found=("found", "mean"),
                                               prec=("precision", "mean"), r2=("rounds", lambda r: (r > 1).sum()))
        lines.append(f"{name:45s} pooled {pooled.mean():.3f} [{pooled.min():.3f},{pooled.max():.3f}] calls "
                     f"{P.calls.sum() / seeds:6.0f} || " + " | ".join(
                         f"{short[q]} {r.recall:.3f}/{r.calls:.0f}/{r.found:.0f} p{r.prec:.2f} r2:{r.r2}" for q, r in g.iterrows()))
    lines.append("\ngate sizes: " + ", ".join(f"{k} {int(v.sum())}" for k, v in gates.items()))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "report.txt").write_text(txt)


def sheets(name="2x2", base="whole (phone sim)"):
    """Judge-yes cars/bicycles the tiling's fast mode finds and the single vector's does not (seed 0), 12 random each,
    plus judge-yes cars the tiling moves from past 2,000 into the top 2,000. Native pixels -> eval/tiles_phone/sheets."""
    import pickle
    import eval_recall as ER
    from PIL import ImageFont
    F = pickle.load(open(OUT / "found_sets.pkl", "rb"))
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores_q3vl4b_real").glob("part*.parquet"))])
    ER.OUT = OUT; (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0); keys = []
    for q, sh in (("photos with a car", "car"), ("photos with a bicycle", "bicycle")):
        gained = sorted(F[(name, q, 0)] - F[(base, q, 0)])
        lost = sorted(F[(base, q, 0)] - F[(name, q, 0)])
        print(q, "gained", len(gained), "lost", len(lost))
        for tag, lst in (("gain", gained), ("lost", lost)):
            pick = list(rng.choice(lst, size=min(12, len(lst)), replace=False)) if lst else []
            d = sc[(sc["query"] == q) & sc.item_id.isin(pick)].drop_duplicates("item_id").assign(stratum=tag)
            keys += ER.sheets(list(d.itertuples()), f"{tag}_{sh}", ImageFont.load_default())
    json.dump(keys, open(OUT / "sheets" / "key.json", "w"), indent=1, default=str)


def _ev16_plans():
    from findpics.converse import Plan
    recs = [r for f in sorted(Path("eval/everyday16_q3vl").glob("part*.json")) for r in json.load(open(f))]
    out = {}
    for r in recs:
        if r["error"]:
            continue
        a = Plan.model_validate(r["plan"]).albums
        if len(a) == 1:
            out[f"{r['query']}|{r['user']}"] = (a[0].looks, a[0].avoid)
    return out


def embed16():
    from PIL import Image
    from findpics import store
    from findpics.models import ImageTextEncoder
    d = pd.read_parquet("eval/stop_rule/ev16_ranked.parquet")
    ids = sorted(d.item_id.astype(str).unique())
    idx = store.load("data/public/index_disbench")
    path = dict(zip(idx.items.item_id.astype(str), idx.items.path))
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    print("device", enc.device, enc.dtype, len(ids), flush=True)
    P = _ev16_plans()
    T = {}
    for k, (lk, av) in P.items():
        T[k + "|looks"] = enc.texts(lk) if lk else np.zeros((0, 1024), np.float32)
        T[k + "|avoid"] = enc.texts(av) if av else np.zeros((0, 1024), np.float32)
    T["gate"] = enc.texts(GATE_TEXTS)
    np.savez(OUT / "ev16_texts.npz", **{k.replace("/", "_"): v for k, v in T.items()})
    json.dump(sorted(T), open(OUT / "ev16_text_keys.json", "w"))
    V = np.zeros((len(ids), 5, 1024), np.float16)
    for b in range(0, len(ids), 32):
        ims = [rendition(Image.open(path[i])) for i in ids[b:b + 32]]
        V[b:b + len(ims)] = enc.images([c for im in ims for c in crops(im)[:5]]).reshape(len(ims), 5, -1)
        if b % 4096 == 0:
            print(b, flush=True)
    np.save(OUT / "ev16_vectors.npy", V)
    (OUT / "ev16_ids.txt").write_text("\n".join(ids) + "\n")
    print("EMBED16_DONE", V.shape)


def val16():
    from tune_stop_rule import parse_rule, run
    d = pd.read_parquet("eval/stop_rule/ev16_ranked.parquet")
    d["item_id"] = d.item_id.astype(str)
    V = np.load(OUT / "ev16_vectors.npy").astype(np.float32)
    ids = (OUT / "ev16_ids.txt").read_text().split()
    row = {i: k for k, i in enumerate(ids)}
    T = np.load(OUT / "ev16_texts.npz")
    # gate per library (decided at index time on the person's own library): busy = whole-vector prompt score
    G = V[:, 0] @ T["gate"].T
    busy = G[:, :2].max(1) - G[:, 2:].max(1)
    gate = {}
    for u, g in d.drop_duplicates("item_id").groupby("user"):
        r = np.array([row[i] for i in g.item_id])
        for frac in (0.25, 0.5):
            thr = np.quantile(busy[r], 1 - frac)
            for i, b in zip(g.item_id, busy[r]):
                gate.setdefault(frac, {})[i] = b >= thr
    cases = {n: [] for n in ("whole (phone sim)", "2x2", "2x2 | gate busy top25%", "2x2 | gate busy top50%", "PE-Core-L (RESULTS 41)")}
    for (u, q), g in d.groupby(["user", "query"], sort=True):
        k = f"{q}|{u}".replace("/", "_")
        if k + "|looks" not in T.files:
            continue
        r = np.array([row[i] for i in g.item_id])
        L, A = T[k + "|looks"], T[k + "|avoid"]
        w = (V[r, 0] @ L.T).mean(1) if len(L) else np.zeros(len(r))
        t = (V[r, 1:5] @ L.T).mean(2).max(1) if len(L) else np.zeros(len(r))
        av = (V[r, 0] @ A.T).mean(1) if len(A) else 0
        y = g.yes.to_numpy()
        S = {"whole (phone sim)": w - av, "2x2": np.maximum(w, t) - av, "PE-Core-L (RESULTS 41)": g.score.to_numpy()}
        for frac in (0.25, 0.5):
            m = np.array([gate[frac][i] for i in g.item_id])
            S[f"2x2 | gate busy top{int(frac * 100)}%"] = np.where(m, np.maximum(w, t), w) - av
        for n, s in S.items():
            o = np.argsort(-s, kind="stable")
            cases[n].append(dict(key=f"{q}|{u}", y=y[o]))
    rule = parse_rule(ADOPTED)
    lines = [f"ev16 validation (16 unseen DISBench libraries, judge-yes = Qwen3-VL-4B 16-bit exhaustive), rule {ADOPTED}, 20 seeds"]
    rows = []
    for n, cs in cases.items():
        R = run(cs, rule); R["query"] = R.key.str.split("|").str[0]; R.insert(0, "tiling", n); rows.append(R)
        lines.append(f"-- {n}: found {R.found.sum():.1f} / {R.kept.sum()} ({R.found.sum() / R.kept.sum():.3f}), calls {R.calls.sum():.0f}, {len(R)} searches")
        for q, g in R.groupby("query", sort=True):
            lines.append(f"   {q:22s} {g.found.sum():7.1f} / {g.kept.sum():5d} ({g.found.sum() / max(g.kept.sum(), 1):.3f})  calls {g.calls.sum():6.0f}  round2 {(g.rounds - 1).mean():.0%}")
    pd.concat(rows).to_csv(OUT / "val16.tsv", sep="\t", index=False, float_format="%.4g")
    txt = "\n".join(lines); print(txt); (OUT / "val16.txt").write_text(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "embed":
        embed()
    elif cmd == "analyze":
        analyze()
    elif cmd == "embed16":
        embed16()
    elif cmd == "val16":
        val16()
    elif cmd == "sheets":
        sheets(*[a for a in sys.argv[2:] if not a.startswith("--")])
