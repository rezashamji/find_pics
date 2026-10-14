"""RESULTS 61 summary table: writer x size x output tokens x repetition x R59c cascade (TRAIN 643 / dev 351, vs the
all-judge stack) x est. phone index hours for 187,084 photos at 10 / 20 / 30 decode tok/s.
Index hours (ESTIMATE): per photo 1.7 s (the phone's measured judge call: vision tower + a short prompt, RESULTS 44,
4B, cool) + output tokens / r. Reads <replay dir>/compare.txt written by cascade_report.py.
  python eval/facts_writer/r61_table.py [--out=file]
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

N = 187084
ROWS = [  # (label, params, facts dir or None, TRAIN replay dir, dev replay dir)
    ("27B JSON facts (server reference, RESULTS 59c)", "27B", None, "eval/facts_scorer/replay/c59c_train_27b", "eval/facts_scorer/replay/c59c_dev_27b"),
    ("27B short line (format check)", "27B", "eval/facts_writer/facts/short27b", "eval/facts_writer/replay/train_short27b", "eval/facts_writer/replay/dev_short27b"),
    ("Qwen3-VL-4B zero-shot JSON, 4-bit (RESULTS 59c)", "4B", "eval/facts_scorer/phone_facts/q3vl4bq4r", "eval/facts_scorer/replay/c59c_train_4b", "eval/facts_scorer/replay/c59c_dev_4b"),
    ("Qwen3-VL-4B zero-shot keyword line, 4-bit (dev only)", "4B", "eval/facts_scorer/phone_facts/q3vl4bq4s", None, "eval/facts_scorer/replay/c59c_dev_4bshort"),
    ("STUDENT Qwen3-VL-4B, 4-bit phone form", "4B", "eval/facts_writer/facts/s4b_xfit", "eval/facts_writer/replay/train_s4b_xfit", "eval/facts_writer/replay/dev_s4b_xfit"),
    ("STUDENT Qwen3-VL-2B, 4-bit phone form", "2B", "eval/facts_writer/facts/s2b_xfit", "eval/facts_writer/replay/train_s2b_xfit", "eval/facts_writer/replay/dev_s2b_xfit"),
    ("STUDENT Qwen3.5-0.8B, 4-bit phone form", "0.8B", "eval/facts_writer/facts/s08_xfit", "eval/facts_writer/replay/train_s08_xfit", "eval/facts_writer/replay/dev_s08_xfit"),
]


def diff(d, var="r59c"):
    f = Path(d or "", "compare.txt")
    if not d or not f.exists():
        return None
    txt = f.read_text()
    m = re.search(r"^ALL  \(\d+\).*?\n((?:\s+\S+ - judge9b:.*\n)+)", txt, re.M)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if line.strip().startswith(f"{var} - judge9b"):
            x = re.search(r"([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", line)
            return float(x.group(1)), float(x.group(2)), float(x.group(3))
    return None


def tokens(d):
    if not d:
        return None
    n, rep = [], 0
    for f in glob.glob(f"{d}/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            n.append(r.get("n_out", 0))
    return np.array(n) if n else None


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    lines = ["writer | size | out tokens / photo mean (p90) | TRAIN R59c ALL diff [95% CI] | dev R59c ALL diff [95% CI] | "
             "acceptable (both lower >= -.03) | est. index hours, 187,084 photos, at 10 / 20 / 30 tok/s"]
    for label, size, fd, tr, dv in ROWS:
        n = tokens(fd)
        m = 243.0 if n is None else float(n.mean())
        p90 = "" if n is None else f" ({np.percentile(n, 90):.0f})"
        t, d = diff(tr), diff(dv)
        fmt = lambda x: "-" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
        ok = "-" if (t is None or d is None) else ("YES" if t[1] >= -0.03 and d[1] >= -0.03 else "no")
        hrs = " / ".join(f"{N * (1.7 + m / r) / 3600:,.0f}" for r in (10, 20, 30))
        lines.append(f"{label} | {size} | {m:.0f}{p90} | {fmt(t)} | {fmt(d)} | {ok} | {hrs}")
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
