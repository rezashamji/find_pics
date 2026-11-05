"""M82 cluster anchor, last step: facts P (towervec head) vs judge P on the same photos; the runFactsCompare fields + AUC
with a photo-bootstrap 95% CI (2,000 draws, numpy default_rng(0), percentile). GPU job in the main env (towervec encoder
and head in fp32: autocast off, TF32 off), or CPU (slow).
  python eval/device_anchor82/score.py --manifest=dev5k[,devext_N] --tag=dev5k [--sources=a,b] [--pairs=a:b,...]
      [--ablate=<source>]   (+ abl_const / abl_shuffle_<source> / abl_zero_v_<source>, see main())
  -> eval/device_anchor82/results_<tag>.json + results_<tag>.txt (small, public-data numbers)
     data/public/device_anchor82/per_photo_<tag>.parquet (key, source, line, stopped, tokens, facts P, judge P)

Facts P exactly as runFactsCompare: head.fold(q: towervec(questionInput(scorerQuestion(q))), t: PE text(scorerQuestion(q)))
.p(f: towervec(factsInput(factsPrefix(place, taken) + FactsLine.applyRepeatStop(raw).line)), v: PE image vector).
judge yes = judge P >= 0.7 (SearchEngine.accept); facts yes = facts P >= 0.7. --pairs: paired AUC differences
(same bootstrap draws) between two sources on the photos both have.
"""
import contextlib
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/facts_scorer")
sys.path.insert(0, "eval/device_anchor82")
sys.path.insert(0, "src")
from findpics.facts_line import apply_repeat_stop, repeat_scan  # noqa: E402
from judge import QUESTIONS  # noqa: E402

D = Path("data/public/device_anchor82")
OUTD = Path("eval/device_anchor82")
TOWERVEC = "eval/facts_scorer/models/tvw_t2b_final_rs2_xfit_full"
JUDGES = {"j4": "phone judge, cluster form: models/q3vl4b_real = mlx-community/Qwen3-VL-4B-Instruct-4bit weights (4-bit "
                "language model, bf16 vision tower), vLLM, findpics.vlm.VLLMJudge prob mode",
          "j9": "Qwen/Qwen3.5-9B bf16, vLLM, same pipeline (the judge of the cluster cascade runs, RESULTS 59d-70)"}
SOURCES = {"base_vllm": "BASE 2B (mlx-community/Qwen3-VL-2B-Instruct-4bit weights, de-quantized phone form "
                        "models/qwen3vl_2b_mlx4sim) in vLLM",
           "trained_vllm61d": "TRAINED writer p2b_all, its RESULTS 61d DEV lines (vLLM phone form p2b_all_final_q4, same "
                              "rendition bytes, write_facts.py; text whitespace-collapsed by short_format.clean)",
           "base_mlx": "BASE 2B, real MLX (mlx[cuda12] 0.32.3 + mlx-vlm 0.7.4), local mlx-community snapshot",
           "trained_mlx": "TRAINED writer, real MLX, eval/facts_writer/models/p2b_all_final_mlx"}
SHORT = {QUESTIONS[0]: "dog", QUESTIONS[1]: "food", QUESTIONS[2]: "beach"}


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def read_jsonl(f):
    out = []
    for line in open(f):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:      # a torn last line from a preempted job
            pass
    return out


def sources():
    S = {}
    for name in ("trained", "base"):
        for f in sorted(glob.glob(str(D / "out" / "writer_*" / f"{name}_*.jsonl"))):
            for r in read_jsonl(f):
                S.setdefault(f"{name}_mlx", {})[r["key"]] = r
    for f in sorted(glob.glob(str(D / "out" / "vllm_*" / "*_vllm.jsonl"))):
        for r in read_jsonl(f):
            S.setdefault(f"{r['writer']}_vllm", {})[r["key"]] = r
    for f in sorted(glob.glob("eval/facts_writer/facts/p2b_all_dev/*.jsonl")):
        for r in read_jsonl(f):
            S.setdefault("trained_vllm61d", {})[str(r["item_id"])] = dict(text=r["facts"]["caption"], tokens=[0] * int(r["n_out"]))
    return S


def tokens_to_stop(r, stopped):
    """Tokens the phone generates: all of them, or up to the token that completed the repeated item (it stops there)."""
    n = len(r.get("tokens") or [])
    if not stopped or not r.get("segs"):
        return n
    end = repeat_scan(r["text"])[1]
    acc = 0
    for i, s in enumerate(r["segs"]):
        acc += len(s)
        if acc >= end:
            return i + 1
    return n


def auc_w(y, s, w):
    """Weighted Mann-Whitney AUC (ties count 1/2) for each row of photo weights w [b, n]."""
    o = np.argsort(s, kind="mergesort"); y, s, w = y[o], s[o], w[:, o]
    _, start, cnt = np.unique(s, return_index=True, return_counts=True)
    grp = np.repeat(np.arange(len(start)), cnt)
    gp = np.zeros((w.shape[0], len(start))); gn = np.zeros_like(gp)
    np.add.at(gp.T, grp, (w * y).T); np.add.at(gn.T, grp, (w * ~y).T)
    below = np.cumsum(gn, axis=1) - gn
    with np.errstate(invalid="ignore", divide="ignore"):
        return (gp * (below + 0.5 * gn)).sum(1) / (gp.sum(1) * gn.sum(1))


def boot_weights(n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    for c in range(0, B, 100):
        idx = rng.integers(0, n, size=(min(100, B - c), n))
        w = np.zeros((idx.shape[0], n))
        np.add.at(w, (np.arange(idx.shape[0])[:, None], idx), 1.0)
        yield w


def auc_ci(y, scores):
    """point AUC per score vector + percentile CI; with 2 vectors also the paired difference (same draws)."""
    y = np.asarray(y, bool)
    pts = [float(auc_w(y, s, np.ones((1, len(y))))[0]) for s in scores]
    bs = [[] for _ in scores]
    for w in boot_weights(len(y)):
        for i, s in enumerate(scores):
            bs[i].append(auc_w(y, s, w))
    bs = [np.concatenate(b) for b in bs]
    ok = np.all([np.isfinite(b) for b in bs], axis=0)
    out = [(p, float(np.percentile(b[ok], 2.5)), float(np.percentile(b[ok], 97.5))) for p, b in zip(pts, bs)]
    diff = None
    if len(scores) == 2:
        d = bs[0][ok] - bs[1][ok]
        diff = (pts[0] - pts[1], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out, diff, int(ok.sum())


@contextlib.contextmanager
def fp32():
    import torch
    old = torch.autocast
    torch.autocast = lambda *a, **k: contextlib.nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.autocast = old


def main():
    tag = arg("tag")
    M = pd.concat([pd.read_parquet(D / f"manifest_{m}.parquet") for m in arg("manifest").split(",")], ignore_index=True)
    M = M.drop_duplicates("key").reset_index(drop=True)
    img, txt = {}, None
    for f in glob.glob(str(D / "out" / "pe_*.npz")):
        z = np.load(f)
        img.update(zip(z["keys"].tolist(), z["img"].astype(np.float32)))
        assert list(z["qs"]) == QUESTIONS
        txt = {q: z["txt"][i] for i, q in enumerate(QUESTIONS)}
    J = {}
    for t in JUDGES:
        fs = [f for f in glob.glob(str(D / "out" / "judge_*" / f"{t}_*.parquet")) if re.search(rf"/{t}_\d+\.parquet$", f)]
        if fs:
            x = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates(["key", "q"])
            J[t] = {q: dict(zip(g.key, g.p)) for q, g in x.groupby("q")}
    S = sources()
    if arg("sources"):
        S = {k: v for k, v in S.items() if k in arg("sources").split(",")}
    print(tag, len(M), "photos | PE", len(img), "| judges", {t: len(v[QUESTIONS[0]]) for t, v in J.items()},
          "| sources", {k: len(v) for k, v in S.items()}, flush=True)
    from scorers import Scorer
    P, meta = {}, {}
    # --ablate=<source>[,<source>]: what the AUC is made of (does a wiring error really land near .5?)
    #   abl_const            = facts text = the product prefix + an EMPTY line (f carries place/month only)
    #   abl_shuffle_<s>      = <s>'s facts texts (prefix + line) permuted across photos, numpy default_rng(0): every photo
    #                          gets another photo's f (a chain that is wired wrong but still produces sentences)
    #   abl_zero_v_<s>       = <s>'s own facts, image vector set to 0: the ranking then depends on f only (t is constant
    #                          within a question) = a FACTS-ONLY score
    #   abl_zero_v_shuffle_<s> = both: the facts-only score of a broken chain
    runs = [(src, pv, lines, None) for src, lines in S.items() for pv in ("prefix", "prefix_nom")]
    abl = [a for a in (arg("ablate") or "").split(",") if a in S]
    if abl:
        runs.append(("abl_const", "prefix", S[abl[0]], "const"))
    for ab in abl:
        runs += [(f"abl_shuffle_{ab}", "prefix", S[ab], "shuffle"), (f"abl_zero_v_{ab}", "prefix", S[ab], "zero_v"),
                 (f"abl_zero_v_shuffle_{ab}", "prefix", S[ab], "zero_v_shuffle")]
    with fp32():
        for src, pv, lines, mode in runs:
            if True:
                m = M[M.key.isin(lines) & M.key.isin(img)]
                if not len(m):
                    continue
                stop = {k: apply_repeat_stop(lines[k]["text"], 2) for k in m.key}
                facts = {k: p + stop[k][0] for k, p in zip(m.key, m[pv])}
                iv = {k: img[k] for k in m.key}
                if mode == "const":
                    facts = {k: p for k, p in zip(m.key, m[pv])}
                if mode in ("shuffle", "zero_v_shuffle"):
                    ks = m.key.tolist(); perm = np.random.default_rng(0).permutation(len(ks))
                    facts = {k: facts[ks[j]] for k, j in zip(ks, perm)}
                if mode in ("zero_v", "zero_v_shuffle"):
                    iv = {k: np.zeros(1024, np.float32) for k in m.key}
                sc = Scorer(TOWERVEC, facts=facts, img=iv, qvec_fn=lambda qs: [txt[q] for q in qs])
                keys = m.key.tolist()
                P[(src, pv)] = {q: dict(zip(keys, sc.p(keys, [q] * len(keys)))) for q in QUESTIONS}
                if pv == "prefix":
                    meta[src] = {k: (stop[k][0], stop[k][1], len(lines[k].get("tokens") or []),
                                     tokens_to_stop(lines[k], stop[k][1])) for k in keys}
    results, pairs = [], []
    for (src, pv), PQ in P.items():
        keys = sorted(PQ[QUESTIONS[0]])
        for t in J:
            for q in QUESTIONS:
                kk = [k for k in keys if k in J[t][q]]
                pj = np.array([J[t][q][k] for k in kk]); pf = np.array([PQ[q][k] for k in kk], np.float64)
                j, f = pj >= 0.7, pf >= 0.7
                (a,), _, nb = auc_ci(j, [pf]) if 0 < j.sum() < len(j) else ([(np.nan,) * 3], None, 0)
                st = meta.get(src, {})
                r = dict(tag=tag, source=src, prefix=pv, judge=t, q=SHORT[q], n=len(kk), judgeYes=int(j.sum()),
                         factsYes=int(f.sum()), agree=int((f == j).sum()), missedByFacts=int((j & ~f).sum()),
                         extraFromFacts=int((f & ~j).sum()), auc=a[0], lo=a[1], hi=a[2], boot_ok=nb,
                         repeat_stop_fired=int(sum(st[k][1] for k in kk if k in st)),
                         tokens_mean=float(np.mean([st[k][3] for k in kk if k in st])) if st else None)
                results.append(r)
                print(f"{tag} {src:16s} {pv:10s} {t} {r['q']:5s} n={r['n']} judgeYes={r['judgeYes']} factsYes={r['factsYes']} "
                      f"missed={r['missedByFacts']} extra={r['extraFromFacts']} AUC {r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]",
                      flush=True)
    for pr in (arg("pairs") or "").split(","):
        if not pr:
            continue
        a_, b_ = pr.split(":")
        if (a_, "prefix") not in P or (b_, "prefix") not in P:
            continue
        for t in J:
            for q in QUESTIONS:
                kk = sorted(set(P[(a_, "prefix")][q]) & set(P[(b_, "prefix")][q]) & set(J[t][q]))
                y = np.array([J[t][q][k] for k in kk]) >= 0.7
                sa = np.array([P[(a_, "prefix")][q][k] for k in kk], np.float64)
                sb = np.array([P[(b_, "prefix")][q][k] for k in kk], np.float64)
                if not 0 < y.sum() < len(y):
                    continue
                (ra, rb), d, nb = auc_ci(y, [sa, sb])
                pairs.append(dict(tag=tag, a=a_, b=b_, judge=t, q=SHORT[q], n=len(kk), judgeYes=int(y.sum()), auc_a=ra[0],
                                  auc_b=rb[0], diff=d[0], lo=d[1], hi=d[2]))
                print(f"PAIR {a_} - {b_} {t} {SHORT[q]} n={len(kk)} yes={int(y.sum())} {ra[0]:.3f} vs {rb[0]:.3f}: "
                      f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]", flush=True)
    (OUTD / f"results_{tag}.json").write_text(json.dumps(dict(
        judges=JUDGES, sources=SOURCES, towervec=TOWERVEC, questions=QUESTIONS, manifest=arg("manifest"),
        bootstrap="photo resampling, 2,000 draws, numpy default_rng(0), percentile 95% CI", rows=results, pairs=pairs),
        indent=1))
    R = pd.DataFrame(results)
    with open(OUTD / f"results_{tag}.txt", "w") as fo:
        fo.write(R.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n")
        if pairs:
            fo.write("\n" + pd.DataFrame(pairs).to_string(index=False, float_format=lambda v: f"{v:+.3f}") + "\n")
    pp = []
    for src, mm in meta.items():
        for k, (line, stp, nt, nts) in mm.items():
            row = dict(key=k, source=src, line=line, stopped=stp, tokens=nt, tokens_to_stop=nts)
            for q in QUESTIONS:
                row[f"pf_{SHORT[q]}"] = float(P[(src, "prefix")][q][k])
                for t in J:
                    row[f"{t}_{SHORT[q]}"] = J[t][q].get(k, np.nan)
            pp.append(row)
    pd.DataFrame(pp).to_parquet(D / f"per_photo_{tag}.parquet")
    print("SCORE_DONE", tag, len(results), "rows", len(pairs), "pairs", flush=True)


if __name__ == "__main__":
    main()
