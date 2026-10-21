"""RESULTS 70: cap the existing writer lines at K output tokens (the writer's own tokenizer, Qwen3-VL-2B) = an
APPROXIMATION of decoding with max_new_tokens=K: greedy decoding is prefix-deterministic, so a capped run emits the first K
tokens of the uncapped run; we only have the decoded TEXT, so the first K tokens are re-tokenized from it (round trip
checked in fields.txt), and a cap does not let the writer plan a shorter line (it just stops mid-line).
Rule: tokenize the caption (add_special_tokens=False); if it has > K tokens, caption = decode(first K); tokens decoded per
photo = min(n_out, K) (n_out = tokens the writer decoded, incl. EOS / the repeat-stop overshoot). Lines <= K unchanged.
ORDER=R (optional): fields are reordered to the pre-registered order (fields.py / lift.py) before the cap; tokens decoded =
min(len(tokenize(reordered)) + 1, K).
  python eval/trunc70/truncate.py   -> data/public/trunc70/facts/<src>_<tag>/*.jsonl (+ .tokens.json per dir)"""
import json, re, sys
from pathlib import Path
from transformers import AutoTokenizer

SRC = ["t2b_final_rs2_xfit", "p2b_devmix_rs2"]
KS = [16, 24, 32, 48, 64]
OUT = Path("data/public/trunc70/facts")
ORDER = [x for x in (next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--order=")), "")).split(",") if x]
RK = [int(x) for x in (next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rks=")), "")).split(",") if x]
tok = AutoTokenizer.from_pretrained("models/qwen3vl_2b_mlx4sim")


def reorder(cap):
    """'name: ...' segments (with any unnamed '; ' continuation) moved to ORDER; fields not in ORDER keep their relative
    order after the listed ones."""
    segs, cur = [], None
    for sg in cap.split("; "):
        m = re.match(r"([a-z]+):", sg)
        if m or cur is None:
            cur = [m.group(1) if m else "(unnamed)", sg]; segs.append(cur)
        else:
            cur[1] += "; " + sg
    rank = {n: i for i, n in enumerate(ORDER)}
    segs = sorted(segs, key=lambda s: rank.get(s[0], len(ORDER)))     # stable
    return "; ".join(s[1] for s in segs)


for src in SRC:
    files = sorted(Path("eval/facts_writer/facts", src).glob("*.jsonl"))
    variants = [(f"k{K}", K, False) for K in KS] + ([(f"r{K}", K, True) for K in RK + [10 ** 6]] if ORDER else [])
    for tag, K, ro in variants:
        if ro and K == 10 ** 6:
            tag = "rfull"
        d = OUT / f"{src}_{tag}"; d.mkdir(parents=True, exist_ok=True)
        toks = {}
        for f in files:
            out = []
            for line in open(f):
                r = json.loads(line); cap = (r.get("facts") or {}).get("caption")
                if cap:
                    c2 = reorder(cap) if ro else cap
                    ids = tok(c2, add_special_tokens=False).input_ids
                    n_dec = min(len(ids) + 1, K) if ro else min(r["n_out"], K)
                    if len(ids) > K:
                        c2 = tok.decode(ids[:K])
                    r = dict(r); r["facts"] = {"caption": c2}; r["n_out_r70"] = n_dec; r["capped"] = len(ids) > K
                    toks[str(r["item_id"])] = n_dec
                out.append(json.dumps(r))
            (d / f.name).write_text("\n".join(out) + "\n")
        (d / ".tokens.json").write_text(json.dumps(toks))
        print(src, tag, len(toks), "lines", flush=True)
