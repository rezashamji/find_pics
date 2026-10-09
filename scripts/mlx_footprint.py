"""Phone weight footprint of an mlx-community checkpoint, by component, and what each weight-only compression saves.
Reads only the safetensors HEADERS (dtype + shape per tensor) over HTTP; no weights are downloaded.
MLX affine storage per quantized matrix [out, in] at b bits, group g: packed uint32 = out*in*b/8 bytes, plus one bf16
scale and one bf16 bias per group = out*in*4/g bytes. So 4-bit g64 = 0.5625 B/weight, 8-bit g64 = 1.0625, 3-bit g32 = 0.5.
Usage: python scripts/mlx_footprint.py mlx-community/Qwen3-VL-4B-Instruct-4bit
"""
import json
import math
import sys

from huggingface_hub import get_safetensors_metadata

B = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1}
VISION_LINEAR = (".attn.qkv.weight", ".attn.proj.weight", ".mlp.linear_fc1.weight", ".mlp.linear_fc2.weight",
                 ".linear_fc1.weight", ".linear_fc2.weight")


def qbytes(n, bits, group):
    return n * bits / 8 + n * 4 / group


def main(repo):
    meta = get_safetensors_metadata(repo)
    t = {k: (v.dtype, v.shape) for fm in meta.files_metadata.values() for k, v in fm.tensors.items()}
    part = {}
    for k, (d, s) in t.items():
        g = ("vision linears (bf16)" if k.startswith("vision_tower") and k.endswith(VISION_LINEAR) else
             "vision other (conv patch embed, pos embed, norms, biases)" if k.startswith("vision_tower") else
             "embed_tokens = lm_head (tied), 4-bit" if "embed_tokens" in k or "lm_head" in k else
             "language layers, 4-bit (+ norms)")
        part[g] = part.get(g, 0) + math.prod(s) * B[d]
    tot = sum(part.values())
    for g, v in sorted(part.items()):
        print(f"{g:58s} {v / 1e9:6.3f} GB  {v / 2**30:6.3f} GiB")
    print(f"{'TOTAL weights':58s} {tot / 1e9:6.3f} GB  {tot / 2**30:6.3f} GiB")
    nv = sum(math.prod(s) for k, (d, s) in t.items() if k.startswith("vision_tower") and k.endswith(VISION_LINEAR))
    ne = math.prod(t["language_model.model.embed_tokens.scales"][1]) * 64
    nl = sum(math.prod(t[k][1]) * 64 for k in t if k.endswith(".scales") and "embed_tokens" not in k)
    print(f"\nweights: vision linears {nv / 1e6:.1f} M, embeddings {ne / 1e6:.1f} M, language linears {nl / 1e6:.1f} M")
    cands = {"(a) vision 8-bit g64": nv * 2 - qbytes(nv, 8, 64), "vision 6-bit g64": nv * 2 - qbytes(nv, 6, 64),
             "vision 5-bit g64": nv * 2 - qbytes(nv, 5, 64), "(b) vision 4-bit g64": nv * 2 - qbytes(nv, 4, 64),
             "(c) embeddings 3-bit g64": qbytes(ne, 4, 64) - qbytes(ne, 3, 64),
             "(d) language + embeddings 3-bit g32": qbytes(nl + ne, 4, 64) - qbytes(nl + ne, 3, 32)}
    print("\nsaved by each candidate:")
    for c, v in cands.items():
        print(f"{c:40s} {v / 1e9:6.3f} GB  {v / 2**30:6.3f} GiB  -> weights {(tot - v) / 1e9:.3f} GB")
    tc = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    if tc:   # optional: the text config, for the KV cache per token
        kv = tc["num_hidden_layers"] * 2 * tc["num_key_value_heads"] * tc["head_dim"] * 2
        print(f"\nKV cache (bf16): {kv} B per token")


if __name__ == "__main__":
    main(sys.argv[1])
