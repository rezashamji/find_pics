"""RESULTS 61d: compare two MLX checkpoints tensor by tensor (codes byte-equal; bf16 tensors equal) - validates
to_mlx.py's quantization + packing + key / layout mapping against the real mlx-community file when both start from the
same bf16 weights.  python eval/facts_writer/mlx_cmp.py <ours dir> <ref dir>"""
import sys
import torch
from safetensors import safe_open

a, b = sys.argv[1], sys.argv[2]
fa, fb = safe_open(a + "/model.safetensors", "pt"), safe_open(b + "/model.safetensors", "pt")
ka, kb = set(fa.keys()), set(fb.keys())
print("keys only ours", len(ka - kb), "only ref", len(kb - ka))
tot = eq = 0; worst = []
for k in sorted(ka & kb):
    x, y = fa.get_tensor(k), fb.get_tensor(k)
    if x.shape != y.shape:
        worst.append((k, "shape", tuple(x.shape), tuple(y.shape))); continue
    if x.dtype == torch.uint32:
        x, y = x.view(torch.int32), y.view(torch.int32)
        codes = lambda t: torch.stack([(t >> (4 * i)) & 15 for i in range(8)], -1)
        cx, cy = codes(x), codes(y); n = cx.numel(); e = int((cx == cy).sum())
        if e < n: worst.append((k, "codes equal", e / n, "max code diff", int((cx - cy).abs().max())))
    else:
        n = x.numel(); e = int((x == y).sum())
        if e < n: worst.append((k, "equal", e / n, "max abs", float((x.float() - y.float()).abs().max())))
    tot += n; eq += e
print(f"MLX_CMP {eq:,}/{tot:,} elements identical ({eq / tot:.6f}); tensors not identical: {len(worst)}")
for w in worst[:15]:
    print(" ", w)
