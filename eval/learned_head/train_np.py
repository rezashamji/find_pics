"""M48: can a small learned head over the STORED PE-Core-B-16 vectors replace the photo judge? NumPy only (CPU).

(Written in NumPy because on 10-12 the lab file system served memory-mapped shared libraries at ~22 KB/s: `import
torch` hung for > 45 min on login and compute nodes alike, while plain reads were fast. The heads are small enough that
CPU training takes minutes.)

Input per (photo, query): the photo's stored image vectors (whole photo + 2x2 tiles, phone simulation; Open Images:
whole, videos: frames) and the PE-Core-B-16 text vector of the plan's judge question. Output: P(teacher says yes).
Models
  cos_q   cosine(whole vector, question text)                          (no training)
  look    the phone's fast score exactly (mean look - mean avoid; FindPicsCore lookScores; eval sets only)
  lr      logistic regression on [cos, whole * text]                    (1,026 params)
  mlp     MLP over [img, txt, img*txt, |img-txt|] (standardised) -> 1024 -> 256 -> 1, ReLU   (~4.5M params)
  att     same MLP over the query-attended vector: pooled = sum_j softmax_j(50 * cos(txt, v_j)) v_j over the 5 stored
          vectors (whole + 2x2 tiles; fixed temperature, no learned attention), features [pooled, txt, pooled*txt,
          |pooled-txt|]                                                (~4.5M params; needs 5 vectors per photo)
Experiments (held-out = the query never appears in training)
  E1   train JD + OI (15 non-everyday concepts) -> test EV16 (12 searches x 16 libraries), A4 (6 x 4), OI everyday-5
  E1b  as E1, minus every JD photo from the 20 test libraries (held-out query AND library)
  E2   JD 5-fold, folds = clusters of near-duplicate questions (text cosine >= 0.93 linked) -> per-kind, held-out questions
  E3   OI leave-concepts-out (5 folds x 4 concepts; JD questions naming the concept removed) -> full 19,218-item library
  E4   teacher-error trap, SEEN queries / new libraries: train JD + OI15 + EV16 with (a) Qwen3-VL-4B labels raw,
       (b) only photos where Qwen3-VL-4B and Qwen3.5-9B agree, (c) Qwen3.5-9B raw -> test A4 (teacher = phone real
       weights; eye labels as TEST only)
  python eval/learned_head/train_np.py [E1 E1b E2 E3 E4]  -> eval/learned_head/scores_*.parquet, weights_*.npz, train.log
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("eval/learned_head")
EPOCHS, BS, LR = 6, 1024, 1e-3
SEEDS = (0, 1, 2)
TEMP = 50.0
OI_EVERYDAY = {"oi:dog", "oi:cat", "oi:bicycle", "oi:dog_on_beach", "oi:bike_street"}
OI_WORDS = {"oi:bread": r"bread|loaf|baguette|toast", "oi:baked_goods": r"baked|pastr|bread|cake|cookie|muffin|croissant|bakery",
            "oi:pizza": r"pizza", "oi:cake": r"cake", "oi:sandwich": r"sandwich|burger", "oi:dog": r"dog|pupp",
            "oi:bicycle": r"bicycle|bike|cycl", "oi:guitar": r"guitar", "oi:christmas_tree": r"christmas|xmas",
            "oi:swimming_pool": r"pool|swim", "oi:cat": r"\bcats?\b|kitten", "oi:horse": r"horse|pony|riding",
            "oi:sunglasses": r"sunglass|glasses|shades", "oi:wine_glass": r"wine|glass",
            "oi:coffee_cup": r"coffee|cup|mug|latte|espresso", "oi:dog_on_beach": r"dog|beach",
            "oi:person_sunglasses": r"sunglass|glasses|shades", "oi:birthday_candles": r"candle|birthday|cake",
            "oi:eating_pizza": r"pizza", "oi:bike_street": r"bicycle|bike|cycl"}
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n"); LOGF.flush()


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(s):
        s.A = pd.read_parquet(OUT / "pairs.parquet")
        V = np.load(OUT / "vecs.npy").astype(np.float32)
        V /= np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-6)
        s.V = V.astype(np.float16)
        s.n = np.load(OUT / "nvec.npy").astype(np.int64)
        T = np.load(OUT / "texts.npy").astype(np.float32)
        s.T = T / np.linalg.norm(T, axis=-1, keepdims=True)
        s.A["cos_q"] = s.cos(s.A.vrow.to_numpy(), s.A.tid.to_numpy())

    def cos(s, vrow, tid, B=200000):
        return np.concatenate([np.einsum("nd,nd->n", s.V[vrow[b:b + B], 0].astype(np.float32), s.T[tid[b:b + B]])
                               for b in range(0, len(vrow), B)])

    def img(s, vrow, tid, kind):
        """The image vector the model sees: whole photo, or (att) the query-attended mix of the stored vectors."""
        if kind != "att":
            return s.V[vrow, 0].astype(np.float32)
        Vv = s.V[vrow].astype(np.float32); t = s.T[tid]
        a = np.einsum("nvd,nd->nv", Vv, t) * TEMP
        a = np.where(np.arange(5)[None] < s.n[vrow][:, None], a, -1e4)
        a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        p = np.einsum("nv,nvd->nd", a, Vv)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def feats(s, vrow, tid, kind):
        i = s.img(vrow, tid, kind); t = s.T[tid]
        if kind == "lr":
            return np.concatenate([(i * t).sum(1, keepdims=True) * 10, i * t * 256], 1)
        return np.concatenate([i, t, i * t, np.abs(i - t)], 1)


# ------------------------------------------------------------------------------------------------------------ models
class Adam:
    def __init__(s, params, lr, wd):
        s.p, s.lr, s.wd = params, lr, wd
        s.m = [np.zeros_like(p) for p in params]; s.v = [np.zeros_like(p) for p in params]; s.t = 0

    def step(s, grads, lr):
        s.t += 1
        for p, g, m, v in zip(s.p, grads, s.m, s.v):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            mh = m / (1 - 0.9 ** s.t); vh = v / (1 - 0.999 ** s.t)
            if p.ndim > 1:
                p -= lr * s.wd * p
            p -= lr * mh / (np.sqrt(vh) + 1e-8)


class Head:
    """kind 'lr': z = w.x + b. 'mlp' / 'att': x standardised -> W1 (1024) ReLU -> W2 (256) ReLU -> w3."""
    def __init__(s, kind, din, rng, H=1024, H2=256):
        s.kind = kind
        if kind == "lr":
            s.P = [np.zeros((din, 1), np.float32), np.zeros(1, np.float32)]
        else:
            s.P = [(rng.standard_normal((din, H)) * np.sqrt(2 / din)).astype(np.float32), np.zeros(H, np.float32),
                   (rng.standard_normal((H, H2)) * np.sqrt(2 / H)).astype(np.float32), np.zeros(H2, np.float32),
                   (rng.standard_normal((H2, 1)) * np.sqrt(1 / H2)).astype(np.float32), np.zeros(1, np.float32)]
        s.mu = np.zeros(din, np.float32); s.sd = np.ones(din, np.float32)

    def nparams(s):
        return sum(p.size for p in s.P)

    def forward(s, x, keep=False):
        x = (x - s.mu) / s.sd
        if s.kind == "lr":
            z = (x @ s.P[0])[:, 0] + s.P[1][0]
            return (z, x) if keep else z
        W1, b1, W2, b2, w3, b3 = s.P
        h1 = np.maximum(x @ W1 + b1, 0); h2 = np.maximum(h1 @ W2 + b2, 0)
        z = (h2 @ w3)[:, 0] + b3[0]
        return (z, (x, h1, h2)) if keep else z

    def backward(s, cache, dz):
        if s.kind == "lr":
            x = cache
            return [x.T @ dz[:, None], np.array([dz.sum()], np.float32)]
        x, h1, h2 = cache
        W1, b1, W2, b2, w3, b3 = s.P
        g3 = h2.T @ dz[:, None]; gb3 = np.array([dz.sum()], np.float32)
        d2 = (dz[:, None] @ w3.T) * (h2 > 0)
        g2 = h1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ W2.T) * (h1 > 0)
        g1 = x.T @ d1; gb1 = d1.sum(0)
        return [g1, gb1, g2, gb2, g3, gb3]


def target(d, label="y"):
    if label == "y":
        p = d.p.to_numpy(dtype=np.float64)
        return np.where(np.isnan(p), d.y.to_numpy().astype(float), p).astype(np.float32)
    return d[label].to_numpy().astype(np.float32)


def fit(D, tr, kind, seed, label="y", verbose=False):
    rng = np.random.default_rng(seed)
    vr, ti = tr.vrow.to_numpy(), tr.tid.to_numpy()
    y = target(tr, label)
    src = tr.src.to_numpy(); w = np.ones(len(tr), np.float32)
    for s_ in np.unique(src):                    # each source carries total weight ~ sqrt(its rows)
        k = src == s_; w[k] = np.sqrt(k.sum()) / k.sum()
    w = w / w.mean()
    samp = rng.choice(len(tr), size=min(20000, len(tr)), replace=False)
    X0 = D.feats(vr[samp], ti[samp], kind)
    m = Head(kind, X0.shape[1], rng)
    m.mu = X0.mean(0); m.sd = X0.std(0) + 1e-4
    opt = Adam(m.P, LR, 0.01 if kind != "lr" else 1e-4)
    nb = (len(tr) + BS - 1) // BS; steps = EPOCHS * nb; st = 0
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr)); tot = 0.0
        for b in range(nb):
            j = perm[b * BS:(b + 1) * BS]
            x = D.feats(vr[j], ti[j], kind)
            z, cache = m.forward(x, keep=True)
            p = sig(z)
            tot += float((w[j] * -(y[j] * np.log(p + 1e-7) + (1 - y[j]) * np.log(1 - p + 1e-7))).sum())
            dz = (w[j] * (p - y[j]) / len(j)).astype(np.float32)
            lr = LR * (min(1.0, (st + 1) / (0.1 * steps)) * 0.5 * (1 + np.cos(np.pi * min(1.0, st / steps))))
            opt.step(m.backward(cache, dz), lr); st += 1
        if verbose:
            log(f"    {kind} seed {seed} epoch {ep} loss {tot / len(tr):.4f}")
    return m


def predict(D, m, d, B=16384):
    vr, ti = d.vrow.to_numpy(), d.tid.to_numpy()
    return np.concatenate([sig(m.forward(D.feats(vr[b:b + B], ti[b:b + B], m.kind))) for b in range(0, len(d), B)])


def subsample(d, rng, neg=3000, hard=500, by="query", score="cos_q"):
    """Full-library sets are ~99% teacher-no: keep every teacher-yes, the `hard` highest-cosine teacher-no and `neg`
    random teacher-no per group."""
    keep = []
    for _, g in d.groupby(by):
        y = g.y.to_numpy().astype(bool)
        no = g[~y]
        h = no.sort_values(score, ascending=False).index[:hard]
        rest = no.index.difference(h)
        r = rng.choice(rest, size=min(neg, len(rest)), replace=False) if len(rest) else np.array([], int)
        keep.append(np.concatenate([g.index[y].to_numpy(), h.to_numpy(), np.asarray(r)]))
    return d.loc[np.concatenate(keep)]


def clusters(D, qs, thr=0.93):
    """Union-find over questions whose PE-Core text vectors have cosine >= thr (paraphrases share a fold). 0.8 chained
    2,339 of 2,412 questions into one cluster (10-12 run); at 0.93 the largest has 249 ("Is there pizza in this photo?" /
    "Is there a pizza ..." 0.999; "desert dunes" / "open desert" 0.90 stay apart)."""
    tid = D.A.drop_duplicates("query").set_index("query").tid
    X = D.T[tid.loc[qs].to_numpy()]
    S = X @ X.T
    par = list(range(len(qs)))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in zip(*np.where(np.triu(S >= thr, 1))):
        par[f(a)] = f(b)
    return np.array([f(a) for a in range(len(qs))])


# ------------------------------------------------------------------------------------------------------------ main
def main():
    global LOGF
    want = set(sys.argv[1:]) or {"E1", "E1b", "E2", "E3", "E4"}
    t0 = time.time()
    LOGF = open(OUT / f"train_{'_'.join(sorted(want))}.log", "w")
    D = Data(); A = D.A
    log("pairs", len(A), "vectors", D.V.shape, f"threads {os.environ.get('OMP_NUM_THREADS')}", f"{time.time() - t0:.0f}s")
    rng = np.random.default_rng(0)
    for k in ("lr", "mlp", "att"):
        log("params", k, Head(k, 1025 if k == "lr" else 4096, rng).nparams())
    JD = A[A.src == "JD"]; OI = A[A.src == "OI"]; EV = A[A.src == "EV16"]; A4 = A[A.src == "A4"]
    OI15 = OI[~OI["query"].isin(OI_EVERYDAY)]
    OI15s = subsample(OI15, rng)
    test_libs = set(EV.lib) | set(A4.lib)
    keep_cols = ["src", "lib", "query", "item", "kind", "y", "p", "cos_q", "look"]

    def run(name, tr, te, kinds=("lr", "mlp", "att"), seeds=SEEDS, label="y", save=None):
        log(f"[{name}] train rows {len(tr)} (queries {tr['query'].nunique()}, teacher-yes {int(target(tr, label).round().sum())})"
            f", test rows {len(te)} (queries {te['query'].nunique()})")
        te = te.copy()
        for k in kinds:
            ps = []
            for sd in seeds:
                m = fit(D, tr, k, sd, label, verbose=sd == seeds[0])
                ps.append(predict(D, m, te))
                if save and sd == seeds[0]:
                    np.savez(OUT / f"weights_{save}_{k}.npz", mu=m.mu.astype(np.float16), sd=m.sd.astype(np.float16),
                             **{f"p{i}": p.astype(np.float16) for i, p in enumerate(m.P)})
                log(f"    {k} seed {sd} trained+scored {time.time() - t0:.0f}s")
            for sd, p in zip(seeds, ps):
                te[f"{k}_s{sd}"] = p
            te[k] = np.mean(ps, 0)
        extra = [c for c in te.columns if c not in keep_cols and (c in ("y9", "y4", "eye", "p_sim", "cut", "jd_part")
                                                                  or re.match(r"(lr|mlp|att)(_s\d)?$", c))]
        te[keep_cols + extra].to_parquet(OUT / f"scores_{name}.parquet")
        log(f"[{name}] done {time.time() - t0:.0f}s")

    oi5 = OI[OI["query"].isin(OI_EVERYDAY)]
    test1 = pd.concat([EV, A4, oi5])
    if "E1" in want:
        run("E1", pd.concat([JD, OI15s]), test1, save="E1")
    if "E1b" in want:
        run("E1b", pd.concat([JD[~JD.lib.isin(test_libs)], OI15s]), test1, kinds=("lr", "mlp"), seeds=(0,))
    if "E2" in want:
        qs = np.array(sorted(JD["query"].unique()))
        cl = clusters(D, qs)
        ucl, size = np.unique(cl, return_counts=True)            # largest-first greedy: 5 folds of ~equal size
        fold_of_cl, load = {}, np.zeros(5)
        for c in ucl[np.lexsort((np.random.default_rng(1).random(len(ucl)), -size))]:
            f = int(np.argmin(load)); fold_of_cl[c] = f; load[f] += size[ucl == c][0]
        fold = pd.Series([fold_of_cl[c] for c in cl], index=qs)
        log(f"E2 question clusters: {len(ucl)} for {len(qs)} questions (largest {np.bincount(np.unique(cl, return_inverse=True)[1]).max()})")
        parts = []
        for f in range(5):
            te_q = set(fold.index[fold == f])
            run(f"E2_f{f}", pd.concat([JD[~JD["query"].isin(te_q)], OI15s]), JD[JD["query"].isin(te_q)], seeds=(0,))
            parts.append(pd.read_parquet(OUT / f"scores_E2_f{f}.parquet").assign(fold=f))
        pd.concat(parts).to_parquet(OUT / "scores_E2.parquet")
        for f in range(5):
            (OUT / f"scores_E2_f{f}.parquet").unlink()
    if "E3" in want:
        oq = sorted(OI["query"].unique()); perm = np.random.default_rng(2).permutation(len(oq))
        parts = []
        for f in range(5):
            te_q = {oq[i] for i in perm[f::5]}
            pat = "(?i)" + "|".join(OI_WORDS[q] for q in te_q)
            jd_ok = ~JD["query"].str.contains(pat, regex=True)
            log(f"E3 fold {f}: held out {sorted(te_q)}; JD questions dropped for naming them: {JD[~jd_ok]['query'].nunique()}")
            run(f"E3_f{f}", pd.concat([JD[jd_ok], subsample(OI[~OI["query"].isin(te_q)], rng)]),
                OI[OI["query"].isin(te_q)], seeds=(0,))
            parts.append(pd.read_parquet(OUT / f"scores_E3_f{f}.parquet"))
        pd.concat(parts).to_parquet(OUT / "scores_E3.parquet")
        for f in range(5):
            (OUT / f"scores_E3_f{f}.parquet").unlink()
    if "E4" in want:
        EVs = subsample(EV, rng, neg=300, hard=200, by=["query", "lib"])
        EVs9 = subsample(EV.assign(y=EV.y9.astype(bool)), rng, neg=300, hard=200, by=["query", "lib"])
        agree = EV[EV.y == EV.y9.astype(bool)]
        EVsa = subsample(agree, rng, neg=300, hard=200, by=["query", "lib"])
        log(f"E4: EV16 Qwen3-VL-4B yes {int(EV.y.sum())}, 9B yes {int(EV.y9.sum())}, both {int((EV.y & EV.y9.astype(bool)).sum())}, "
            f"disagree {int((EV.y != EV.y9.astype(bool)).sum())} of {len(EV)}")
        for q, g in EV.groupby("query"):
            log(f"    {q:24s} 4B-yes {int(g.y.sum()):5d}  9B-yes {int(g.y9.sum()):5d}  both {int((g.y & g.y9.astype(bool)).sum()):5d}")
        run("E4a_raw4b", pd.concat([JD, OI15s, EVs]), A4, kinds=("mlp",), save="E4a")
        run("E4b_agree", pd.concat([JD, OI15s, EVsa]), A4, kinds=("mlp",), save="E4b")
        run("E4c_raw9b", pd.concat([JD, OI15s, EVs9]), A4, kinds=("mlp",))
    log("TRAIN_DONE", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
