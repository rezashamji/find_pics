"""RESULTS 59: query-time scorers that answer a yes/no photo question from STORED per-photo data (no pixels).

Kinds (all map (question text, photo) -> P(yes)):
  ce        cross-encoder: one small text transformer reads "[CLS] question [SEP] facts text [SEP]" per photo (upper bound
            of what a small reader gets out of the facts; per-photo transformer pass at query time = NOT milliseconds)
  tower     two towers, shared small text encoder: facts text -> vector ONCE at index time, question -> vector once per
            search; a tiny MLP over [f, q, f*q] per photo (one folded matrix-vector product per photo: milliseconds)
  vec       control, no facts: the same MLP over the phone's PE-Core-B-16 image vector and the question's B-16 text
            vector [v, t, v*t] (RESULTS 48's input, retrained on the oracle answers)
  towervec  tower + vec features
Facts text and question rewriting: eval/facts_scorer/data.py (facts_text, scorer_question).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


class Tok:
    """Pre-tokenised pieces -> padded BERT inputs ([CLS] a [SEP] (b [SEP]))."""

    def __init__(self, tokenizer):
        self.t = tokenizer
        self.cls, self.sep, self.pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    def ids(self, texts, max_len=256):
        return [x[:max_len] for x in self.t(list(texts), add_special_tokens=False)["input_ids"]]

    def pair(self, A, B, max_len):
        seqs, tts = [], []
        for a, b in zip(A, B):
            a = a[:64]
            b = b[:max(0, max_len - 3 - len(a))]
            seqs.append([self.cls] + a + [self.sep] + b + [self.sep])
            tts.append([0] * (len(a) + 2) + [1] * (len(b) + 1))
        return self._pad(seqs, tts)

    def single(self, A, max_len):
        seqs = [[self.cls] + a[:max_len - 2] + [self.sep] for a in A]
        return self._pad(seqs, [[0] * len(s) for s in seqs])

    def _pad(self, seqs, tts):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        tt = torch.zeros((len(seqs), L), dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, (s, t) in enumerate(zip(seqs, tts)):
            ids[i, :len(s)] = torch.tensor(s); tt[i, :len(t)] = torch.tensor(t); am[i, :len(s)] = 1
        return ids.to(DEV), tt.to(DEV), am.to(DEV)


class CE(nn.Module):
    def __init__(self, base):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1)

    def forward(self, ids, tt, am):
        return self.m(input_ids=ids, token_type_ids=tt, attention_mask=am).logits[:, 0].float()


class Head(nn.Module):
    """MLP over [f, q, f*q] (+ [v, t, v*t]); every input block is linear in the photo side once the question is
    fixed, so the first layer folds into ONE matrix per search (as RESULTS 48's head)."""

    def __init__(self, d_txt, d_img, hid=512, use_txt=True, use_img=False):
        super().__init__()
        self.use_txt, self.use_img = use_txt, use_img
        d = (3 * d_txt if use_txt else 0) + (3 * d_img if use_img else 0)
        self.net = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 128), nn.GELU(), nn.Linear(128, 1))
        self.ln_f = nn.LayerNorm(d_txt) if use_txt else None
        self.ln_q = nn.LayerNorm(d_txt) if use_txt else None

    def forward(self, f=None, q=None, v=None, t=None):
        """f: photo facts vectors [N, d]; q: question vectors [N, d] (row-aligned), v/t likewise for images."""
        x = []
        if self.use_txt:
            f, q = self.ln_f(f), self.ln_q(q)
            x += [f, q, f * q]
        if self.use_img:
            v, t = v * 32.0, t * 32.0          # unit B-16 vectors -> ~unit-variance components
            x += [v, t, v * t]
        return self.net(torch.cat(x, -1))[:, 0].float()


class Tower(nn.Module):
    def __init__(self, base, d_img=1024, use_txt=True, use_img=False):
        super().__init__()
        from transformers import AutoModel
        self.use_txt, self.use_img = use_txt, use_img
        self.enc = AutoModel.from_pretrained(base) if use_txt else None
        d_txt = self.enc.config.hidden_size if use_txt else 0
        self.head = Head(d_txt, d_img, use_txt=use_txt, use_img=use_img)

    def embed(self, ids, tt, am):
        return mean_pool(self.enc(input_ids=ids, token_type_ids=tt, attention_mask=am).last_hidden_state, am).float()


class Scorer:
    """Load a trained model dir (config.json + weights.pt) and answer P(yes) for (photo ids, question texts).
    facts: {item_id: text}; img: {item_id: B-16 vector} (vec kinds); qvec_fn: texts -> B-16 text vectors."""

    def __init__(self, mdir, facts=None, img=None, qvec_fn=None, max_len=256):
        from transformers import AutoTokenizer
        mdir = Path(mdir)
        self.cfg = json.loads((mdir / "config.json").read_text())
        k = self.cfg["kind"]
        self.kind = k
        if k == "ens":     # equal-weight logit average of member scorers (config: members, weights)
            self.members = [Scorer(m, facts=facts, img=img, qvec_fn=qvec_fn, max_len=max_len) for m in self.cfg["members"]]
            self.w = self.cfg.get("weights") or [1 / len(self.members)] * len(self.members)
            return
        self.tok = Tok(AutoTokenizer.from_pretrained(self.cfg["base"])) if k != "vec" else None
        if k == "ce":
            self.model = CE(self.cfg["base"])
        else:
            self.model = Tower(self.cfg["base"], use_txt=k in ("tower", "towervec"), use_img=k in ("vec", "towervec"))
        self.model.load_state_dict(torch.load(mdir / "weights.pt", map_location="cpu"))
        self.model.to(DEV).eval()
        self.facts, self.img, self.qvec_fn = facts or {}, img or {}, qvec_fn
        self.max_len = self.cfg.get("max_len", max_len)
        self._ftok, self._fvec, self._qvec, self._tvec = {}, {}, {}, {}

    def _facts_tok(self, ids):
        miss = [i for i in dict.fromkeys(ids) if i not in self._ftok]
        if miss:
            for i, t in zip(miss, self.tok.ids([self.facts.get(i, "no description") for i in miss], self.max_len)):
                self._ftok[i] = t
        return [self._ftok[i] for i in ids]

    @torch.no_grad()
    def _fvecs(self, ids):
        miss = [i for i in dict.fromkeys(ids) if i not in self._fvec]
        for s in range(0, len(miss), 512):
            b = miss[s:s + 512]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                e = self.model.embed(*self.tok.single(self._facts_tok(b), self.max_len))
            for i, x in zip(b, e.float().cpu()):
                self._fvec[i] = x
        return torch.stack([self._fvec[i] for i in ids]).to(DEV)

    @torch.no_grad()
    def _qvecs(self, qs):
        miss = [q for q in dict.fromkeys(qs) if q not in self._qvec]
        if miss:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                e = self.model.embed(*self.tok.single(self.tok.ids(miss, 64), 66))
            for q, x in zip(miss, e.float().cpu()):
                self._qvec[q] = x
        return torch.stack([self._qvec[q] for q in qs]).to(DEV)

    def _tvecs(self, qs):
        miss = [q for q in dict.fromkeys(qs) if q not in self._tvec]
        if miss:
            for q, x in zip(miss, self.qvec_fn(miss)):
                self._tvec[q] = torch.tensor(np.asarray(x, np.float32))
        return torch.stack([self._tvec[q] for q in qs]).to(DEV)

    def _ivecs(self, ids):
        z = np.zeros(1024, np.float32)
        return torch.tensor(np.stack([np.asarray(self.img.get(i, z), np.float32) for i in ids])).to(DEV)

    @torch.no_grad()
    def p(self, ids, qs, bs=1024):
        """P(yes) for row-aligned photo ids and scorer-question texts."""
        ids, qs = list(map(str, ids)), list(qs)
        if self.kind == "ens":
            lg = 0.0
            for w, m in zip(self.w, self.members):
                pm = np.clip(m.p(ids, qs, bs), 1e-6, 1 - 1e-6)
                lg = lg + w * np.log(pm / (1 - pm))
            return (1 / (1 + np.exp(-lg))).astype(np.float32)
        out = np.zeros(len(ids), np.float32)
        for s in range(0, len(ids), bs):
            bi, bq = ids[s:s + bs], qs[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                if self.kind == "ce":
                    lg = self.model(*self.tok.pair(self.tok.ids(bq, 64), self._facts_tok(bi), self.max_len))
                else:
                    kw = {}
                    if self.model.use_txt:
                        kw.update(f=self._fvecs(bi), q=self._qvecs(bq))
                    if self.model.use_img:
                        kw.update(v=self._ivecs(bi), t=self._tvecs(bq))
                    lg = self.model.head(**kw)
            out[s:s + bs] = torch.sigmoid(lg.float()).cpu().numpy()
        return out
