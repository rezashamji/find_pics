"""RESULTS 61d (2) parity, step 3 (main env): MLX outputs vs the vLLM phone-form lines on the same photos.
Per photo: exact line match; tokens (2B tokenizer on the cleaned lines): common prefix / longer length, first divergence;
facts F1 MLX vs vLLM per field (quality.py's parsing), and each vs the 27B's line (the training target).
  python eval/facts_writer/r61d_parity_cmp.py <photos json> <mlx jsonl> ... [--out=file]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "eval/complex_bench")
from common import norm  # noqa: E402
from quality import items, load, prf  # noqa: E402
from short_format import parse  # noqa: E402


def f1s(a, b):
    A, B = parse(a), parse(b)
    out = {}
    for f in ("objects", "colours"):
        x = prf([norm(t) if f == "objects" else t.lower() for t in items(A.get(f))],
                [norm(t) if f == "objects" else t.lower() for t in items(B.get(f))])
        if x is not None:
            out[f] = x[2]
    for f in ("scene", "activity", "people"):
        x = prf(re.findall(r"\w+", A.get(f, "").lower()), re.findall(r"\w+", B.get(f, "").lower()))
        if x is not None:
            out[f] = x[2]
    return out


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/qwen3vl_2b_mlx4sim")
    P = {p["item_id"]: p for p in json.loads(Path(pos[1]).read_text())}
    ref27 = load("short27b")
    L = []
    for mf in pos[2:]:
        M = [json.loads(x) for x in open(mf)]
        ex, pre, div, posagree, pm = [], [], [], [], []
        fv = {}; fm27 = {}; fv27 = {}
        for m in M:
            p = P[m["item_id"]]
            a, b = m["mlx"], p["vllm"]
            ta, tb = tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
            k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            ex.append(a == b); pre.append(k / max(len(ta), len(tb), 1))
            div.append(k if a != b else None)
            posagree.append(np.mean([x == y for x, y in zip(ta, tb)]) if min(len(ta), len(tb)) else float(a == b))
            pm.append(m.get("n_prompt") == p["vllm_n_prompt"])
            for d, (x, y) in ((fv, (a, b)), (fm27, (a, ref27[m["item_id"]][0])), (fv27, (b, ref27[m["item_id"]][0]))):
                for f, v in f1s(x, y).items():
                    d.setdefault(f, []).append(v)
        n = len(M)
        dv = [d for d in div if d is not None]
        L.append(f"=== {mf} (ctx={M[0].get('ctx') if M else '-'}): {n} photos")
        L.append(f"exact line match {sum(ex)}/{n}; token common prefix / longer length mean {np.mean(pre):.3f}; "
                 f"position-wise token agreement mean {np.mean(posagree):.3f}; first divergence token (when not exact) "
                 f"median {np.median(dv) if dv else '-'}, min {min(dv) if dv else '-'}; prompt token count equal {sum(pm)}/{n}")
        L.append("facts F1 per field (n photos where defined): " + "; ".join(
            f"{f}: MLX vs vLLM {np.mean(fv[f]):.2f} (n={len(fv[f])}), vs 27B MLX {np.mean(fm27.get(f, [np.nan])):.2f} / "
            f"vLLM {np.mean(fv27.get(f, [np.nan])):.2f}" for f in ("objects", "colours", "scene", "activity", "people") if f in fv))
        mt = [len(m["tokens"]) for m in M]; vt = [P[m["item_id"]]["vllm_n_out"] for m in M]
        L.append(f"output tokens mean MLX {np.mean(mt):.1f} / vLLM {np.mean(vt):.1f}; MLX at 120 cap {sum(t >= 120 for t in mt)}/{n}, "
                 f"vLLM {sum(t >= 120 for t in vt)}/{n}; MLX CPU s/photo mean {np.mean([m['s'] for m in M]):.1f}")
    txt = "\n".join(L); print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
