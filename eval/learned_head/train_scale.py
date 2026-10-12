"""M48-scale step 3: does the learned head generalise to UNSEEN queries once training covers ~10x more distinct
questions? Learning curve over the number of generated questions, every point tested on queries AND libraries it never
saw. Same heads, features and optimiser as train_np.py (lr: [cos, img*txt]; mlp: [img, txt, img*txt, |img-txt|]
standardised -> 1024 -> 256 -> 1; Adam 1e-3, 10% warm-up + cosine, decoupled weight decay 0.01 on matrices, batch 1024,
6 epochs; each source weighted ~sqrt(its rows)), ported to torch so millions of pairs train in minutes on one GPU
(the E1 port check below reproduces train_np's E1 numbers).

Training data
  NEW    eval/learned_head/scale: generated questions (questions.json) x photos of the 37 TRAINING DISBench libraries
         (250 most similar by cosine + 150 random), teacher = the phone judge (Qwen3-VL-4B real 4-bit weights, P(yes)).
         Target = sigmoid(logit(P) - logit(cut)), cut = the phone's judge_cutoff(question) (0.999 for "Is this a photo
         of X?", 0.7 otherwise): a soft label that is 0.5 exactly at the phone's own accept line.
  JD     data/public/judge_distill (2,412 real planner questions x 48 photos, Qwen3.5-9B soft P), photos of the 20 test
         libraries removed (as E1b).  OI15  Open Images, 15 non-everyday concepts (as E1).
Runs
  E1port   JD (all) + OI15        = report E1 (port check, numbers must match train_np's)
  JDonly   JD (no test libs) + OI15 = report E1b (the 2.4k-question reference point)
  N{n}     NEW only, the first n questions of a fixed random order (nested), n in CURVE
  ALL      NEW (all) + JD (no test libs) + OI15
  <run>-na same data as <run>, mlp WITHOUT the |img - txt| block (inputs [img, txt, img*txt], 3072 -> 1024 -> 256 -> 1):
           the first layer then folds per search into ONE 1024 x 1024 matrix per photo (W_img + W_prod diag(txt)) plus
           a per-search constant (RESULTS 48, the phone-cost variant)
  <run>-r  a re-run of <run> (same data, features, seeds), so every seed's weights exist (weights_<run>-r_mlp_s{k}.pt)
Every run saves each mlp seed's weights (weights_{run}_mlp_s{k}.pt; seed 0 also as weights_{run}_mlp.pt).
Held out from every NEW run: HOLD_N new questions chosen as whole clusters (text cosine >= 0.93 linked) -> 'HN' test.
Tests (scored for every run): EV16 (12 searches x 16 libraries), A4 (6 x 4, eye labels), OI everyday-5, JD (all 2,412
questions; unseen for N runs), HN.

  python eval/learned_head/train_scale.py [RUN ...]   -> eval/learned_head/scale/scores_{run}.parquet, weights_{run}_mlp.npz
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "eval"); sys.path.insert(0, "eval/learned_head")
OUT = Path("eval/learned_head"); S = OUT / "scale"
EPOCHS, BS, LR, WD = 6, 1024, 1e-3, 0.01
SEEDS = (0, 1, 2)
CURVE = (2400, 5000, 10000)
HOLD_N = 500
OI_EVERYDAY = {"oi:dog", "oi:cat", "oi:bicycle", "oi:dog_on_beach", "oi:bike_street"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n"); LOGF.flush()


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))


def vec_keys():
    """Row keys of vecs.npy, exactly build.disbench_vectors() + oi_vectors() order (first occurrence wins)."""
    keys = Path("eval/tiles_phone/ids.txt").read_text().split()
    keys += Path("eval/tiles_phone/ev16_ids.txt").read_text().split()
    for f in sorted(OUT.glob("img_part*_ids.txt")):
        keys += f.read_text().split()
    first = {}
    for k, key in enumerate(keys):
        first.setdefault(key, k)
    return first, len(keys)


def clusters(T, thr=0.93):
    S_ = T @ T.T
    par = list(range(len(T)))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in zip(*np.where(np.triu(S_ >= thr, 1))):
        par[f(a)] = f(b)
    return np.array([f(a) for a in range(len(T))])


class Data:
    def __init__(s):
        from score_apple_photos import judge_cutoff
        t0 = time.time()
        A = pd.read_parquet(OUT / "pairs.parquet")
        V = np.load(OUT / "vecs.npy").astype(np.float32)
        V /= np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-6)
        s.n = np.load(OUT / "nvec.npy").astype(np.int64)
        T0 = np.load(OUT / "texts.npy").astype(np.float32); T0 /= np.linalg.norm(T0, axis=-1, keepdims=True)
        first, nd = vec_keys()
        assert nd + int((A.src == "OI").any()) * 0 <= len(V)
        s.Q = json.load(open(S / "questions.json"))
        QT = np.load(S / "qtext.npy").astype(np.float32); QT /= np.linalg.norm(QT, axis=-1, keepdims=True)
        s.T = np.concatenate([T0, QT])
        parts = sorted((S / "labels").glob("part*.parquet"))
        L = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else \
            pd.DataFrame(dict(qid=pd.Series([], dtype=int), item_id=pd.Series([], dtype=str), how=pd.Series([], dtype=str), p=pd.Series([], dtype=float)))
        L["vrow"] = L.item_id.map(lambda u: first.get(u, -1))
        miss = int((L.vrow < 0).sum())
        L = L[L.vrow >= 0]
        cut = np.array([judge_cutoff(q["q"]) for q in s.Q])
        c = cut[L.qid.to_numpy()]
        p = L.p.to_numpy(np.float64)
        N = pd.DataFrame(dict(src="NEW", lib="pool", query=L.qid.map(lambda i: f"n{i:05d}"), item=L.item_id,
                              kind=L.qid.map(lambda i: s.Q[i]["jkind"]), y=p >= c, p=1 / (1 + np.exp(-(logit(p) - logit(c)))),
                              vrow=L.vrow.to_numpy(), tid=len(T0) + L.qid.to_numpy(), qid=L.qid.to_numpy(), how=L.how,
                              p_raw=p, cut=c))
        log(f"NEW pairs {len(N)} (dropped {miss} without a vector), questions {N.qid.nunique()}, photos {N.item.nunique()}, "
            f"teacher-yes at the phone cut {int(N.y.sum())}; {time.time() - t0:.0f}s")
        A["qid"] = -1
        s.A = pd.concat([A, N], ignore_index=True)
        s.Vg = torch.tensor(V, dtype=torch.float16, device=DEV)
        s.Tg = torch.tensor(s.T, dtype=torch.float32, device=DEV)
        s.A["cos_q"] = s.cos(s.A.vrow.to_numpy(), s.A.tid.to_numpy())

    def cos(s, vr, ti, B=1 << 18):
        out = []
        for b in range(0, len(vr), B):
            i = s.Vg[torch.as_tensor(vr[b:b + B], device=DEV), 0].float()
            out.append((i * s.Tg[torch.as_tensor(ti[b:b + B], device=DEV)]).sum(1).cpu().numpy())
        return np.concatenate(out)

    def feats(s, vr, ti, kind):
        i = s.Vg[vr, 0].float(); t = s.Tg[ti]
        if kind == "lr":
            return torch.cat([(i * t).sum(1, keepdim=True) * 10, i * t * 256], 1)
        if kind == "mlp_na":                       # no |img - txt|: foldable per search (RESULTS 48)
            return torch.cat([i, t, i * t], 1)
        return torch.cat([i, t, i * t, (i - t).abs()], 1)


class Head(torch.nn.Module):
    def __init__(s, kind, din, H=1024, H2=256):
        super().__init__()
        s.kind = kind
        s.register_buffer("mu", torch.zeros(din)); s.register_buffer("sd", torch.ones(din))
        if kind == "lr":
            s.net = torch.nn.Linear(din, 1)
            torch.nn.init.zeros_(s.net.weight); torch.nn.init.zeros_(s.net.bias)
        else:
            s.net = torch.nn.Sequential(torch.nn.Linear(din, H), torch.nn.ReLU(), torch.nn.Linear(H, H2), torch.nn.ReLU(),
                                        torch.nn.Linear(H2, 1))
            for m in s.net:
                if isinstance(m, torch.nn.Linear):          # train_np init: N(0, 2/fan_in) (last layer 1/fan_in), zero bias
                    torch.nn.init.normal_(m.weight, 0, math.sqrt((1 if m.out_features == 1 else 2) / m.in_features))
                    torch.nn.init.zeros_(m.bias)

    def forward(s, x):
        return s.net((x - s.mu) / s.sd)[:, 0]


def fit(D, tr, kind, seed):
    g = torch.Generator(device="cpu").manual_seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vr = torch.as_tensor(tr.vrow.to_numpy(), device=DEV); ti = torch.as_tensor(tr.tid.to_numpy(), device=DEV)
    p = tr.p.to_numpy(dtype=np.float64)
    y = torch.as_tensor(np.where(np.isnan(p), tr.y.to_numpy().astype(float), p), dtype=torch.float32, device=DEV)
    src = tr.src.to_numpy(); w = np.ones(len(tr), np.float32)
    for s_ in np.unique(src):
        k = src == s_; w[k] = np.sqrt(k.sum()) / k.sum()
    w = torch.as_tensor(w / w.mean(), device=DEV)
    samp = torch.as_tensor(rng.choice(len(tr), size=min(20000, len(tr)), replace=False), device=DEV)
    X0 = D.feats(vr[samp], ti[samp], kind)
    m = Head(kind, X0.shape[1]).to(DEV)
    m.mu.copy_(X0.mean(0)); m.sd.copy_(X0.std(0) + 1e-4)
    decay = [q for n, q in m.named_parameters() if q.ndim > 1]; nodecay = [q for n, q in m.named_parameters() if q.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WD if kind != "lr" else 1e-4),
                             dict(params=nodecay, weight_decay=0.0)], lr=LR, betas=(0.9, 0.999), eps=1e-8)
    nb = (len(tr) + BS - 1) // BS; steps = EPOCHS * nb
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / (0.1 * steps)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, st / steps))))
    for ep in range(EPOCHS):
        perm = torch.randperm(len(tr), generator=g).to(DEV); tot = 0.0
        for b in range(nb):
            j = perm[b * BS:(b + 1) * BS]
            z = m(D.feats(vr[j], ti[j], kind))
            l = (w[j] * torch.nn.functional.binary_cross_entropy_with_logits(z, y[j], reduction="none")).sum() / len(j)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sched.step()
            tot += float(l) * len(j)
        log(f"    {kind} seed {seed} epoch {ep} loss {tot / len(tr):.4f}")
    return m


@torch.no_grad()
def predict(D, m, d, B=1 << 16):
    vr, ti = d.vrow.to_numpy(), d.tid.to_numpy()
    out = []
    for b in range(0, len(d), B):
        x = D.feats(torch.as_tensor(vr[b:b + B], device=DEV), torch.as_tensor(ti[b:b + B], device=DEV), m.kind)
        out.append(torch.sigmoid(m(x)).cpu().numpy())
    return np.concatenate(out)


def subsample(d, rng, neg=3000, hard=500, by="query", score="cos_q"):
    keep = []
    for _, g in d.groupby(by):
        y = g.y.to_numpy().astype(bool)
        no = g[~y]
        h = no.sort_values(score, ascending=False).index[:hard]
        rest = no.index.difference(h)
        r = rng.choice(rest, size=min(neg, len(rest)), replace=False) if len(rest) else np.array([], int)
        keep.append(np.concatenate([g.index[y].to_numpy(), h.to_numpy(), np.asarray(r)]))
    return d.loc[np.concatenate(keep)]


def main():
    global LOGF
    want = sys.argv[1:] or ["E1port", "JDonly"] + [f"N{n}" for n in CURVE] + ["ALL"]
    t0 = time.time()
    LOGF = open(S / f"train_{'_'.join(want)}.log", "w")
    D = Data(); A = D.A
    log("pairs", len(A), "device", DEV, f"{time.time() - t0:.0f}s")
    JD = A[A.src == "JD"]; OI = A[A.src == "OI"]; EV = A[A.src == "EV16"]; A4 = A[A.src == "A4"]; NEW = A[A.src == "NEW"]
    test_libs = set(EV.lib) | set(A4.lib)
    rng = np.random.default_rng(0)
    OI15s = subsample(OI[~OI["query"].isin(OI_EVERYDAY)], rng)
    JDb = JD[~JD.lib.isin(test_libs)]
    # held-out new questions: whole paraphrase clusters, ~HOLD_N questions
    nq = len(D.Q)
    QT = D.T[-nq:]
    cl = clusters(QT)
    order = np.random.default_rng(7).permutation(nq)
    hold, seen_cl = set(), set()
    for i in order:
        if len(hold) >= HOLD_N:
            break
        if cl[i] in seen_cl:
            continue
        mem = np.where(cl == cl[i])[0]
        if len(mem) > 5:
            continue
        seen_cl.add(cl[i]); hold |= set(mem.tolist())
    train_order = [int(i) for i in np.random.default_rng(8).permutation(nq) if i not in hold]
    json.dump(dict(hold=sorted(hold), train_order=train_order), open(S / "split.json", "w"))
    HN = NEW[NEW.qid.isin(hold)]
    log(f"questions {nq}: held out {len(hold)} ({HN.qid.nunique()} with labels), trainable {len(train_order)}")
    tests = pd.concat([EV, A4, OI[OI["query"].isin(OI_EVERYDAY)], JD, HN])
    keep_cols = ["src", "lib", "query", "item", "kind", "y", "p", "cos_q", "look", "qid", "how", "p_raw", "cut",
                 "eye", "y9", "y4", "p_sim", "jd_part"]
    for run in want:
        full_run = run
        run, _, opt = run.partition("-")
        assert opt in ("", "na", "r"), f"unknown run option {opt}"
        if run == "E1port":
            tr = pd.concat([JD, OI15s]); te = tests[tests.src != "JD"]
        elif run == "JDonly":
            tr = pd.concat([JDb, OI15s]); te = tests[tests.src != "JD"]
        elif run.startswith("N"):
            n = int(run[1:]); qs = set(train_order[:n])
            tr = NEW[NEW.qid.isin(qs)]; te = tests
        elif run == "ALL":
            tr = pd.concat([NEW[NEW.qid.isin(set(train_order))], JDb, OI15s]); te = tests[tests.src != "JD"]
        else:
            raise SystemExit(f"unknown run {run}")
        run = full_run
        log(f"[{run}] train rows {len(tr)} (questions {tr['query'].nunique()}, sources {tr.src.value_counts().to_dict()}), "
            f"test rows {len(te)}")
        te = te[[c for c in keep_cols if c in te.columns] + ["vrow", "tid"]].copy()
        for kind, seeds in (("lr", (0,)), ("mlp", SEEDS)):
            fk = "mlp_na" if (kind == "mlp" and opt == "na") else kind
            ps = []
            for sd in seeds:
                m = fit(D, tr, fk, sd)
                ps.append(predict(D, m, te))
                if kind == "mlp":
                    torch.save(m.state_dict(), S / f"weights_{run}_mlp_s{sd}.pt")
                    if sd == seeds[0]:
                        torch.save(m.state_dict(), S / f"weights_{run}_mlp.pt")
                log(f"    {fk} seed {sd} done {time.time() - t0:.0f}s")
            for sd, p in zip(seeds, ps):
                te[f"{kind}_s{sd}"] = p.astype(np.float32)
            te[kind] = np.mean(ps, 0).astype(np.float32)
        if run.partition("-")[0].startswith("N") or run.partition("-")[0] == "ALL":       # nearness of each test question to this run's training questions
            tq = np.array(sorted(set(tr.tid[tr.src == "NEW"])))
            Tt = D.T[tq]
            ut = te.tid.unique()
            near = {t: float((Tt @ D.T[t]).max()) for t in ut}
            te["near_train"] = te.tid.map(near).astype(np.float32)
        te.drop(columns=["vrow"]).to_parquet(S / f"scores_{run}.parquet")
        log(f"[{run}] done {time.time() - t0:.0f}s")
    log("TRAIN_DONE", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
