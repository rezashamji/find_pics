"""RESULTS 68 (4): the exported towervec encoder in REAL MLX (Python, CPU backend), built the way mlx-swift-lm's
MLXEmbedders BertModel builds it (Libraries/MLXEmbedders/Models/Bert.swift, read 10-16): HF keys renamed by its
sanitizeBertWeights, word + position (+ token-type) embeddings -> LayerNorm, 12 post-LN blocks (MultiHeadAttention with
bias, ln1, linear1 -> gelu (erf) -> linear2, ln2), mean pool over all tokens. Loads every weight STRICTLY (all names must
match the module = the layout mlx-swift expects) and compares with the original PyTorch fp32 vectors
(data/public/cascade_port/encoder/lines.json from encoder.py export).
Run (CPU, MLX 0.32.3 in the Debian container):
  singularity exec --cleanenv -B $FP_ROOT:$FP_ROOT tools/mlxpy/py312.sif tools/mlxpy/venv/bin/python \
      eval/cascade_port/encoder_mlx.py <dir> <float32|float16> <facts|question> <zeros|none> <n>
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "eval/cascade_port/models"
L = json.loads((ROOT / "data/public/cascade_port/encoder/lines.json").read_text())


def sanitize(w):     # = sanitizeBertWeights (Bert.swift)
    out = {}
    for k, v in w.items():
        k2 = (k.replace(".layer.", ".layers.").replace(".self.key.", ".key_proj.").replace(".self.query.", ".query_proj.")
              .replace(".self.value.", ".value_proj.").replace(".attention.output.dense.", ".attention.out_proj.")
              .replace(".attention.output.LayerNorm.", ".ln1.").replace(".output.LayerNorm.", ".ln2.")
              .replace(".intermediate.dense.", ".linear1.").replace(".output.dense.", ".linear2.")
              .replace(".LayerNorm.", ".norm.").replace("pooler.dense.", "pooler.").replace("bert.", ""))
        if k2 != "embeddings.position_ids":
            out[k2] = v
    return out


class Emb(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.word_embeddings = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.position_embeddings = nn.Embedding(c["max_position_embeddings"], c["hidden_size"])
        self.token_type_embeddings = nn.Embedding(c["type_vocab_size"], c["hidden_size"])
        self.norm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])

    def __call__(self, ids, types):
        x = self.word_embeddings(ids) + self.position_embeddings(mx.arange(ids.shape[1])[None])
        if types is not None:
            x = x + self.token_type_embeddings(types)
        return self.norm(x)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.attention = nn.MultiHeadAttention(d, c["num_attention_heads"], bias=True)
        self.ln1 = nn.LayerNorm(d, eps=c["layer_norm_eps"]); self.ln2 = nn.LayerNorm(d, eps=c["layer_norm_eps"])
        self.linear1 = nn.Linear(d, c["intermediate_size"]); self.linear2 = nn.Linear(c["intermediate_size"], d)

    def __call__(self, x):
        h = self.ln1(x + self.attention(x, x, x))
        return self.ln2(self.linear2(nn.gelu(self.linear1(h))) + h)


class Enc(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = [Block(c) for _ in range(c["num_hidden_layers"])]


class Bert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = Emb(c); self.encoder = Enc(c); self.pooler = nn.Linear(c["hidden_size"], c["hidden_size"])

    def __call__(self, ids, types):
        x = self.embeddings(ids, types)
        for b in self.encoder.layers:
            x = b(x)
        return x


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def main(name, dt, kind, types, limit):
    """one check: exported dir, compute dtype, facts|question, token_type_ids zeros|none, first `limit` lines"""
    mx.set_default_device(mx.cpu)
    d = MODELS / name
    c = json.loads((d / "config.json").read_text())
    w = sanitize(mx.load(str(d / "model.safetensors")))
    m = Bert(c)
    m.load_weights(list(w.items()), strict=True)     # every name and shape must match
    m.set_dtype(getattr(mx, dt))
    out = []
    seqs = L[f"{kind}_ids"][:limit]
    for s in seqs:
        ids = mx.array([s])
        h = m(ids, mx.zeros_like(ids) if types == "zeros" else None)
        out.append(np.array(h.astype(mx.float32).mean(axis=1)[0]))
    ref = np.array(L[f"{kind}_vec"][:limit])
    cc = cos(out, ref)
    print(f"MLX {name} compute {dt} {kind} token_type_ids={types}: n {len(cc)} cosine min {cc.min():.6f} "
          f"mean {cc.mean():.6f} max |diff| {np.abs(np.array(out) - ref).max():.2e}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
