"""RESULTS 59: train a query-time scorer on TRAIN libraries' 27B oracle answers (soft target = oracle P), predict
held-out TRAIN libraries (selection) and DEV pairs (evaluation). Test split never read.

  python eval/facts_scorer/train.py --kind=ce|tower|vec|towervec --base=<hf id> --out=eval/facts_scorer/models/<name>
      [--hold=K,...]   train folds left out (fold k = sorted TRAIN users [k::4]); predictions written for them
      [--facts=eval/facts_scorer/data/facts_27b.parquet] [--epochs=1] [--steps=6000] [--lr=3e-5] [--bs=256]
      [--max_len=256] [--predict=hold,dev]
  -> <out>/config.json, weights.pt, train.log, pred_hold.parquet / pred_dev.parquet (user, item_id, q, p, pred)
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).parent))
from scorers import CE, DEV, Tok, Tower  # noqa: E402

D = Path("eval/facts_scorer/data")


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def train_folds():
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
    tr = sorted(sp["train"])
    return {u: k for k in range(4) for u in tr[k::4]}


def b16_image_vectors(need: set):
    sys.path.insert(0, "eval/learned_head")
    from build import disbench_vectors
    keys, arrs = disbench_vectors()
    out, off = {}, 0
    for a in arrs:
        n = len(a)
        for i, k in enumerate(keys[off:off + n]):
            if k in need:
                out[k] = np.asarray(a[i, 0], np.float32)
        off += n
    return out


_ENC = None


def b16_text(texts):
    global _ENC
    from findpics.models import ImageTextEncoder
    if _ENC is None:
        _ENC = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    return np.concatenate([_ENC.texts(texts[s:s + 256]) for s in range(0, len(texts), 256)]).astype(np.float32)


def main():
    kind = arg("kind", "ce")
    base = arg("base", "cross-encoder/ms-marco-MiniLM-L12-v2")
    out = Path(arg("out")); out.mkdir(parents=True, exist_ok=True)
    hold = [int(x) for x in arg("hold", "").split(",") if x != ""]
    facts_f = arg("facts", str(D / "facts_27b.parquet"))
    lr, bs, max_len = float(arg("lr", "3e-5")), int(arg("bs", "256")), int(arg("max_len", "256"))
    epochs, steps = float(arg("epochs", "1")), int(arg("steps", "6000"))
    seed = int(arg("seed", "0"))
    predict = arg("predict", "hold,dev").split(",")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    log = open(out / "train.log", "a")

    def say(*a):
        s = " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()

    cfg = dict(kind=kind, base=base, hold=hold, facts=facts_f, lr=lr, bs=bs, max_len=max_len, epochs=epochs,
               steps=steps, seed=seed, extra=arg("extra"))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    say("config", cfg)
    FT = pd.read_parquet(facts_f)
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    fold = train_folds()
    P = pd.read_parquet(D / "pairs_train.parquet")
    if arg("extra"):     # fix B: oracle answers on the shipped planner's own TRAIN questions (same columns)
        E = pd.read_parquet(arg("extra"))
        P = pd.concat([P, E[P.columns.intersection(E.columns)]], ignore_index=True)
        say(f"extra pairs {len(E)} from {arg('extra')}")
    P["fold"] = P.user.map(fold)
    TR = P[~P.fold.isin(hold)].reset_index(drop=True)
    HO = P[P.fold.isin(hold)].reset_index(drop=True)
    say(f"train pairs {len(TR)} ({TR.user.nunique()} libraries, {TR.q.nunique()} q); held-out {len(HO)} ({HO.user.nunique()})")
    tok = None
    if kind != "vec":
        from transformers import AutoTokenizer
        tok = Tok(AutoTokenizer.from_pretrained(base))
    if kind == "ce":
        model = CE(base).to(DEV)
    else:
        model = Tower(base, use_txt=kind in ("tower", "towervec"), use_img=kind in ("vec", "towervec")).to(DEV)
    use_img = kind in ("vec", "towervec")
    t0 = time.time()
    ftok = {}
    if tok is not None:
        ids = list(facts)
        for s in range(0, len(ids), 4096):
            for i, t in zip(ids[s:s + 4096], tok.ids([facts[i] for i in ids[s:s + 4096]], max_len)):
                ftok[i] = t
        qs_all = sorted(set(P.qs) | set(pd.read_parquet(D / "pairs_dev.parquet", columns=["qs"]).qs))
        qtok = dict(zip(qs_all, tok.ids(qs_all, 64)))
        say(f"tokenised {len(ftok)} facts, {len(qtok)} questions in {time.time() - t0:.0f}s; facts tokens median "
            f"{np.median([len(x) for x in ftok.values()]):.0f}, p95 {np.percentile([len(x) for x in ftok.values()], 95):.0f}")
    IV = TV = None
    if use_img:
        IV = b16_image_vectors(set(facts))
        qs_all = sorted(set(P.qs) | set(pd.read_parquet(D / "pairs_dev.parquet", columns=["qs"]).qs))
        TV = dict(zip(qs_all, b16_text(qs_all)))
        say(f"B-16 vectors: {len(IV)} photos, {len(TV)} questions")
    def img_feats(ids, qs):
        z = np.zeros(1024, np.float32)
        v = torch.tensor(np.stack([IV.get(i, z) for i in ids])).to(DEV)
        t = torch.tensor(np.stack([TV[q] for q in qs])).to(DEV)
        return v, t

    enc_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    groups = ([{"params": enc_params, "lr": lr}] if enc_params else []) + \
             ([{"params": head_params, "lr": float(arg("head_lr", "1e-3"))}] if head_params else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    if kind == "ce":
        n_steps = int(math.ceil(len(TR) * epochs / bs))
    else:
        n_steps = steps
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / n_steps))
    model.train()
    say(f"steps {n_steps}")
    if kind == "ce":
        order = np.concatenate([np.random.permutation(len(TR)) for _ in range(int(math.ceil(epochs)))])
        qs_, ids_, ps_ = TR.qs.to_numpy(), TR.item_id.astype(str).to_numpy(), TR.p.to_numpy(np.float32)
        for st in range(n_steps):
            b = order[st * bs:(st + 1) * bs]
            if len(b) == 0:
                break
            x = tok.pair([qtok[q] for q in qs_[b]], [ftok.get(i, []) for i in ids_[b]], max_len)
            y = torch.tensor(ps_[b]).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(*x)
            loss = Fn.binary_cross_entropy_with_logits(lg.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if st % 200 == 0:
                say(f"step {st}/{n_steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    else:
        # grid batches inside one library: nq questions x npho photos, every cell labelled by the oracle (person
        # questions only where asked: masked)
        libs = {}
        for u, g in TR.groupby("user"):
            qs = sorted(g.qs.unique()); ids = sorted(g.item_id.astype(str).unique())
            qi = {q: k for k, q in enumerate(qs)}; ii = {i: k for k, i in enumerate(ids)}
            M = np.full((len(qs), len(ids)), np.nan, np.float32)
            M[g.qs.map(qi).to_numpy(), g.item_id.astype(str).map(ii).to_numpy()] = g.p.to_numpy(np.float32)
            libs[u] = (qs, ids, M)
        users = list(libs); wts = np.array([np.isfinite(libs[u][2]).sum() for u in users], float); wts /= wts.sum()
        nq, npho = int(arg("nq", "32")), int(arg("npho", "128"))
        for st in range(n_steps):
            u = users[np.random.choice(len(users), p=wts)]
            qs, ids, M = libs[u]
            a = np.random.choice(len(qs), min(nq, len(qs)), replace=False)
            # half the photos uniformly, half from rows where the sampled questions have a yes (positives are ~7%)
            pos = np.where((np.nan_to_num(M[a]) >= 0.5).any(0))[0]
            k1 = npho // 2 if len(pos) else 0
            b = np.unique(np.r_[np.random.choice(pos, min(k1, len(pos)), replace=False) if k1 else np.zeros(0, int),
                                np.random.choice(len(ids), npho - min(k1, len(pos)), replace=False)]).astype(int)
            Y = torch.tensor(M[np.ix_(a, b)]).to(DEV)
            mask = torch.isfinite(Y)
            bq = [qs[x] for x in a]; bi = [ids[x] for x in b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                kw = {}
                if model.use_txt:
                    fv = model.embed(*tok.single([ftok.get(i, []) for i in bi], max_len))
                    qv = model.embed(*tok.single([qtok[q] for q in bq], 66))
                    kw.update(f=fv.unsqueeze(0).expand(len(bq), -1, -1).reshape(-1, fv.shape[-1]),
                              q=qv.unsqueeze(1).expand(-1, len(bi), -1).reshape(-1, qv.shape[-1]))
                if use_img:
                    v, t = img_feats(bi, bq)
                    kw.update(v=v.unsqueeze(0).expand(len(bq), -1, -1).reshape(-1, 1024),
                              t=t.unsqueeze(1).expand(-1, len(bi), -1).reshape(-1, 1024))
                lg = model.head(**kw).view(len(bq), len(bi))
            loss = Fn.binary_cross_entropy_with_logits(lg.float()[mask], Y[mask])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if st % 200 == 0:
                say(f"step {st}/{n_steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    torch.save(model.state_dict(), out / "weights.pt")
    say(f"trained in {time.time() - t0:.0f}s")
    # predictions
    from scorers import Scorer
    qfn = b16_text if use_img else None
    S = Scorer(out, facts=facts, img=IV, qvec_fn=qfn, max_len=max_len)
    for which in predict:
        if which == "eye":     # eye-labelled everyday searches (eye_sets.py); oracle p absent
            from data import scorer_question
            X = pd.read_parquet(D / "eye_sets.parquet")
            X = X.assign(q=X.question, qs=X.question.map(scorer_question), p=np.nan)
        else:
            X = HO if which == "hold" else pd.read_parquet(D / "pairs_dev.parquet") if which == "dev" else None
        if X is None or not len(X):
            continue
        t1 = time.time()
        X = X[["user", "item_id", "q", "qs", "p"] + (["set", "eye", "weight", "query"] if which == "eye" else [])].copy()
        X["item_id"] = X.item_id.astype(str)
        X = X.sort_values(["user", "qs", "item_id"]).reset_index(drop=True)
        if arg("limit"):
            X = X.sample(int(arg("limit")), random_state=0).reset_index(drop=True)
        X["pred"] = S.p(X.item_id.tolist(), X.qs.tolist(), bs=2048)
        X.to_parquet(out / f"pred_{which}.parquet")
        say(f"pred_{which}: {len(X)} pairs in {time.time() - t1:.0f}s")
    say("TRAIN_DONE")


if __name__ == "__main__":
    main()
