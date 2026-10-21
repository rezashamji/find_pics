"""RESULTS 70 pre-scoring look at the writer lines (no cascade numbers): writer-tokenizer round trip vs recorded n_out,
per-field token cost and survival at each cap K, per split (TRAIN = t2b_final_rs2_xfit lines of TRAIN libraries, dev =
p2b_devmix_rs2 lines of dev libraries = the lines eval/scope69/job.sh scores).
  python eval/trunc70/fields.py > eval/trunc70/fields.txt"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path("eval/facts_writer/facts")
SRC = {"train": "t2b_final_rs2_xfit", "dev": "p2b_devmix_rs2"}
KS = [16, 24, 32, 48, 64]
S = json.load(open("eval/complex_bench/splits.json"))
tok = AutoTokenizer.from_pretrained("models/qwen3vl_2b_mlx4sim")


def lines(split):
    for u in S[split]:
        for l in open(ROOT / SRC[split] / f"{u}.jsonl"):
            r = json.loads(l)
            yield u, r


for split in ("train", "dev"):
    n = 0; d_nout = []; rt_bad = 0; nofacts = 0
    ftok = defaultdict(list); fend = defaultdict(list); nl = 0
    for u, r in lines(split):
        cap = (r.get("facts") or {}).get("caption")
        if not cap:
            nofacts += 1; continue
        ids = tok(cap, add_special_tokens=False).input_ids
        n += 1; d_nout.append(r["n_out"] - len(ids))
        if tok.decode(ids) != cap:
            rt_bad += 1
        # field boundaries in tokens: cumulative token count of the prefix ending each field
        segs = cap.split("; ")
        pos = 0; prev = 0
        for i, sg in enumerate(segs):
            m = re.match(r"([a-z]+):", sg)
            name = m.group(1) if m else "(unnamed)"
            pos += len(sg) + (2 if i < len(segs) - 1 else 0)
            t = len(tok(cap[:pos], add_special_tokens=False).input_ids)
            ftok[name].append(t - prev); fend[name].append(t); prev = t
    d = np.array(d_nout)
    print(f"== {split} ({SRC[split]}): {n} lines with a caption, {nofacts} without; n_out - len(tokenize(caption)): "
          f"mean {d.mean():.2f}, ==1 {np.mean(d == 1):.3f}, ==0 {np.mean(d == 0):.3f}, >1 {np.mean(d > 1):.3f}, <0 {np.mean(d < 0):.3f}; "
          f"decode(tokenize(caption)) != caption: {rt_bad}")
    print("field | lines with it | tokens mean | ends at token mean (p50) | survives whole at K=" + "/".join(map(str, KS)))
    for name in sorted(ftok, key=lambda x: np.mean(fend[x])):
        e = np.array(fend[name])
        print(f"{name} | {len(e)} ({len(e)/n:.2f}) | {np.mean(ftok[name]):.1f} | {e.mean():.1f} ({np.median(e):.0f}) | "
              + " / ".join(f"{np.mean(e <= K) * len(e) / n:.2f}" for K in KS))
