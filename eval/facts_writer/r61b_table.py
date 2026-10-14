"""RESULTS 61b: learning curve + ship table. For each writer (RESULTS 61 students at ~23k photos seen, RESULTS 61b students
at their checkpoints): output tokens, repetition, R59c cascade ALL difference vs the all-judge stack on TRAIN (643) and dev
(351) with EXACT bootstrap bounds (2,000 resamples, seed 0 = compare.py), the unchanged ship rule (lower >= -.03 on both),
and est. phone index hours for 187,084 photos at 10 / 20 / 30 tok/s (1.7 s + tokens / r per photo).
  python eval/facts_writer/r61b_table.py [--out=file]
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

N = 187084
REF = {"train": "eval/facts_scorer/replay/train_alljudge", "dev": "eval/complex_bench/runs/cmp/r60v2_base_search"}


def scored(d):
    f = Path(d) / "scored.json"
    return {r["qid"]: r for r in json.loads(f.read_text())} if f.exists() else None


def diff(split, d):
    R, S = scored(REF[split]), scored(d)
    if S is None:
        return None
    qs = sorted(set(R) & set(S))
    x = np.array([S[q]["final"]["f1"] - R[q]["final"]["f1"] for q in qs])
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi, len(qs)


def toks(tag):
    n, rep = [], 0
    for f in glob.glob(f"eval/facts_writer/facts/{tag}/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            n.append(r["n_out"])
            w = re.findall(r"\w+", r["facts"]["caption"].lower())
            g = Counter(zip(w, w[1:], w[2:]))
            rep += r["n_out"] >= 120 or (max(g.values()) >= 3 if g else False)
    return (np.array(n), rep / max(1, len(n))) if n else (None, None)


def seen_of(tag):
    m = re.search(r"seen_(\d+)", tag)
    if m:
        return int(m.group(1))
    if tag.startswith("s"):          # RESULTS 61: one epoch over its cross-fit TRAIN half
        return 22600
    log = Path(f"eval/facts_writer/models/{tag.split('_')[0]}_A/train.log")
    if log.exists():
        m = re.findall(r"TRAIN_DONE steps \d+ seen (\d+)", log.read_text())
        if m:
            return int(m[-1])
    return None


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    tags = [("4B", "s4b_xfit"), ("4B", "t4b_seen_100000_xfit"), ("4B", "t4b_seen_200000_xfit"), ("4B", "t4b_seen_400000_xfit"),
            ("4B", "t4b_final_xfit"), ("2B", "s2b_xfit"), ("2B", "t2b_seen_100000_xfit"), ("2B", "t2b_seen_400000_xfit"),
            ("2B", "t2b_final_xfit"), ("0.8B", "s08_xfit"), ("0.8B", "t08_final_xfit")]
    lines = ["size | writer (cross-fit facts) | photos seen per student | out tokens mean (p90) | repetition | TRAIN R59c diff "
             "[95% CI] | dev R59c diff [95% CI] | acceptable | est. index hours at 10 / 20 / 30 tok/s"]
    for size, tag in tags:
        n, rep = toks(tag)
        t = diff("train", f"eval/facts_writer/replay/train_{tag}/r59c")
        d = diff("dev", f"eval/facts_writer/replay/dev_{tag}/r59c")
        if n is None and t is None:
            continue
        f = lambda x: "-" if x is None else f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"
        ok = "-" if (t is None or d is None) else ("YES" if t[1] >= -0.03 and d[1] >= -0.03 else "no")
        m = float(n.mean()) if n is not None else float("nan")
        hrs = " / ".join(f"{N * (1.7 + m / r) / 3600:,.0f}" for r in (10, 20, 30))
        sn = seen_of(tag)
        sn = f"{sn:,}" if sn else "-"
        lines.append(f"{size} | {tag} | {sn} | {m:.0f} ({np.percentile(n, 90):.0f}) | {rep:.3f} | {f(t)} | {f(d)} | {ok} | {hrs}"
                     if n is not None else f"{size} | {tag} | - | - | - | {f(t)} | {f(d)} | {ok} | -")
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
