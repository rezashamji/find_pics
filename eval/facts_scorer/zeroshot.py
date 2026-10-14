"""RESULTS 59, zero-training baselines on the same pairs as the trained scorers (TRAIN for a 2-parameter Platt fit
only, DEV for evaluation):
  zs_bm25    BM25 of the question's content words against the photo's facts text (IDF within the library)
  zs_emb     cosine(all-MiniLM-L6-v2(question), all-MiniLM-L6-v2(facts text))
  zs_embmax  max cosine of the question to any single fact phrase (each list item / caption / person description)
  zs_pecore  cosine of the phone's PE-Core-B-16 image vector to the question's B-16 text vector (today's fast stage)
  -> eval/facts_scorer/models/<name>/pred_{train,dev}.parquet (user, item_id, q, qs, p, raw, pred = Platt(raw))
Usage: python eval/facts_scorer/zeroshot.py [--facts=eval/facts_scorer/data/facts_27b.parquet] [--tag=]
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "eval/complex_bench")
sys.path.insert(0, str(Path(__file__).parent))
from common import norm  # noqa: E402

D = Path("eval/facts_scorer/data")
M = Path("eval/facts_scorer/models")
STOP = set("""a an the is are was were be been being there this that these those it its of in on at to for with
from by and or not no any some anywhere visible photo photos picture image shot shots scene someone something one
do does did can could clearly real actually shown show showing taken being has have having as into onto over under
which what who whom whose where when how answer yes word you your we our they their he she his her them him me my
i am""".split())


def toks(s: str):
    return [norm(w) for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 1]


def phrases(text: str):
    out = []
    for part in text.split(". "):
        body = part.split(": ", 1)[-1]
        out += [x.strip() for x in re.split(r"[;,]", body) if x.strip()]
    return out or [text]


def platt(raw_tr, y_tr, raw):
    """2-parameter logistic fit (Newton), raw score -> P(oracle yes); numpy only (the vLLM env has no sklearn)."""
    x = raw_tr.astype(np.float64); y = y_tr.astype(np.float64)
    mu, sd = x.mean(), x.std() + 1e-9
    z = (x - mu) / sd
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(w[0] * z + w[1])))
        g = np.array([((p - y) * z).sum(), (p - y).sum()])
        h = p * (1 - p)
        H = np.array([[(h * z * z).sum(), (h * z).sum()], [(h * z).sum(), h.sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    a, b = w[0] / sd, w[1] - w[0] * mu / sd
    return (1 / (1 + np.exp(-(a * raw.astype(np.float64) + b)))).astype(np.float32), (float(a), float(b))


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    facts_f = arg("facts", str(D / "facts_27b.parquet"))
    tag = arg("tag", "")
    only = arg("only", "bm25,emb,embmax,pecore").split(",")
    FT = pd.read_parquet(facts_f)
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    PR = {s: pd.read_parquet(D / f"pairs_{s}.parquet", columns=["user", "item_id", "q", "qs", "p"]) for s in ("train", "dev")}
    for s in PR:
        PR[s]["item_id"] = PR[s].item_id.astype(str)
    raw = {name: {} for name in only}
    # BM25 per library
    if "bm25" in only:
        k1, b = 1.2, 0.75
        for s, X in PR.items():
            r = np.zeros(len(X), np.float32)
            for u, g in X.groupby("user"):
                ids = g.item_id.unique()
                tf = {i: Counter(toks(facts.get(i, ""))) for i in ids}
                dl = {i: sum(c.values()) for i, c in tf.items()}
                avg = np.mean(list(dl.values())) or 1.0
                df = Counter(w for c in tf.values() for w in c)
                N = len(ids)
                for q, gq in g.groupby("qs"):
                    qt = toks(q)
                    sc = []
                    for i in gq.item_id:
                        c = tf[i]; v = 0.0
                        for w in qt:
                            if c.get(w):
                                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                                v += idf * c[w] * (k1 + 1) / (c[w] + k1 * (1 - b + b * dl[i] / avg))
                        sc.append(v)
                    r[gq.index.to_numpy()] = sc
            raw["bm25"][s] = r
            print("bm25", s, "done", flush=True)
    if "emb" in only or "embmax" in only:
        from transformers import AutoModel, AutoTokenizer
        name = "sentence-transformers/all-MiniLM-L6-v2"
        tk = AutoTokenizer.from_pretrained(name); md = AutoModel.from_pretrained(name).cuda().eval().half()

        @torch.no_grad()
        def emb(texts, bs=512, L=256):
            out = []
            for s in range(0, len(texts), bs):
                x = tk(texts[s:s + bs], padding=True, truncation=True, max_length=L, return_tensors="pt").to("cuda")
                h = md(**x).last_hidden_state.float(); m = x["attention_mask"].unsqueeze(-1).float()
                out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu())
            return torch.cat(out)
        ids_all = sorted(set(PR["train"].item_id) | set(PR["dev"].item_id))
        qs_all = sorted(set(PR["train"].qs) | set(PR["dev"].qs))
        QE = dict(zip(qs_all, emb(qs_all, L=64)))
        if "emb" in only:
            FE = dict(zip(ids_all, emb([facts.get(i, "") for i in ids_all])))
            for s, X in PR.items():
                r = np.zeros(len(X), np.float32)
                for q, g in X.groupby("qs"):
                    F_ = torch.stack([FE[i] for i in g.item_id]); r[g.index.to_numpy()] = (F_ @ QE[q]).numpy()
                raw["emb"][s] = r
            print("emb done", flush=True)
        if "embmax" in only:
            ph = {i: phrases(facts.get(i, "")) for i in ids_all}
            uniq = sorted({p for v in ph.values() for p in v})
            PE = emb(uniq, bs=2048, L=48); pidx = {p: k for k, p in enumerate(uniq)}
            rows = {i: torch.tensor([pidx[p] for p in v]) for i, v in ph.items()}
            print("phrases", len(uniq), flush=True)
            PEc = PE.cuda()
            for s, X in PR.items():
                r = np.zeros(len(X), np.float32)
                for q, g in X.groupby("qs"):
                    simq = (PEc @ QE[q].cuda()).cpu()
                    r[g.index.to_numpy()] = [float(simq[rows[i]].max()) for i in g.item_id]
                raw["embmax"][s] = r
            print("embmax done", flush=True)
    if "pecore" in only:
        from train import b16_image_vectors, b16_text
        ids_all = sorted(set(PR["train"].item_id) | set(PR["dev"].item_id))
        IV = b16_image_vectors(set(ids_all))
        qs_all = sorted(set(PR["train"].qs) | set(PR["dev"].qs))
        TV = dict(zip(qs_all, b16_text(qs_all)))
        z = np.zeros(1024, np.float32)
        for s, X in PR.items():
            r = np.zeros(len(X), np.float32)
            for q, g in X.groupby("qs"):
                r[g.index.to_numpy()] = np.stack([IV.get(i, z) for i in g.item_id]) @ TV[q]
            raw["pecore"][s] = r
        print("pecore done", flush=True)
    for name, R in raw.items():
        d = M / f"zs_{name}{tag}"; d.mkdir(parents=True, exist_ok=True)
        ytr = (PR["train"].p.to_numpy() >= 0.5).astype(int)
        sub = np.random.RandomState(0).choice(len(ytr), min(400000, len(ytr)), replace=False)
        info = {}
        for s, X in PR.items():
            pred, coef = platt(R["train"][sub], ytr[sub], R[s])
            info["platt"] = coef
            Y = X.copy(); Y["raw"] = R[s]; Y["pred"] = pred
            Y.to_parquet(d / f"pred_{s}.parquet")
        (d / "config.json").write_text(json.dumps(dict(kind=f"zeroshot_{name}", facts=facts_f, **info)))
        print(name, "written", info, flush=True)
    print("ZS_DONE", flush=True)


if __name__ == "__main__":
    main()
