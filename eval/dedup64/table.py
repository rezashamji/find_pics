"""RESULTS 64 table: shared-facts cascade (R59d) vs the per-photo facts baseline (paired) and vs the all-judge stack;
exact bootstrap as r61b_table.diff (2,000 resamples over requests, seed 0); judge calls per search; writer calls saved.
Every run dir (and a COPY of each per-photo baseline's part files, so all are scored against the same truth file at the
same moment) is re-scored with eval/complex_bench/score.py first.
  python eval/dedup64/table.py <split> <s tag> [<s tag> ...] [--out=file]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = {"train": "eval/facts_scorer/replay/train_alljudge", "dev": "eval/complex_bench/runs/cmp/r60v2_base_search"}
BASE = {"train": "eval/facts_writer/replay/c61d_train_t2b_final_rs2_xfit/r59d",
        "dev": "eval/facts_writer/replay/c61d_dev_p2b_devmix_rs2/r59d"}


def rescore(split, d):
    subprocess.run([sys.executable, "eval/complex_bench/score.py", split, str(d)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return {r["qid"]: r for r in json.loads((ROOT / d / "scored.json").read_text())}


def paired(A, B):
    qs = sorted(set(A) & set(B))
    x = np.array([A[q]["final"]["f1"] - B[q]["final"]["f1"] for q in qs])
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi, len(qs), int((x > .05).sum()), int((x < -.05).sum())


def calls(d):
    v = []
    for f in sorted((ROOT / d).glob("stats*.json")):
        v += [r["judge"] for r in json.loads(f.read_text()).values()]
    return np.mean(v), np.median(v)


def errors(S):
    return sum(1 for r in S.values() if r.get("error"))


def main():
    split = sys.argv[1]
    tags = [a for a in sys.argv[2:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    bcopy = ROOT / f"eval/dedup64/replay/{split}_perphoto/r59d"
    bcopy.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / BASE[split]).glob("*.json"):
        if f.name.startswith(("part", "stats")):
            shutil.copy(f, bcopy / f.name)
    rcopy = ROOT / f"eval/dedup64/replay/{split}_alljudge"
    rcopy.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / REF[split]).glob("part*.json"):
        shutil.copy(f, rcopy / f.name)
    R = rescore(split, rcopy.relative_to(ROOT))
    B = rescore(split, bcopy.relative_to(ROOT))
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}] (n={x[3]}; {x[4]} better / {x[5]} worse by > .05)"
    L = [f"{split}: per-photo baseline {BASE[split]} ALL {np.mean([r['final']['f1'] for r in B.values()]):.4f}; "
         f"vs all-judge {f(paired(B, R))}; judge calls / search mean, median {calls(bcopy.relative_to(ROOT))}; errors {errors(B)}",
         "s | writer calls saved | ALL F1 | shared - per-photo [95%] | shared - all-judge [95%] | judge calls / search mean, median | errors | bar (A lower >= -.01, B lower >= -.03)"]
    for t in tags:
        d = Path(f"eval/dedup64/replay/{split}_s{t}/r59d")
        if not (ROOT / d / "part0.json").exists():
            L.append(f"{t} | (not run)"); continue
        S = rescore(split, d)
        a, b = paired(S, B), paired(S, R)
        sv = (ROOT / f"eval/dedup64/data/saved_s{t}.txt").read_text().splitlines()
        sv = next((x.split(": ", 1)[1].split(" saved")[0] for x in sv if x.strip().startswith(split + ":")), "?")
        ok = "PASS" if a[1] >= -.01 and b[1] >= -.03 else "FAIL"
        cm = calls(d)
        L.append(f"{t} | {sv} | {np.mean([r['final']['f1'] for r in S.values()]):.4f} | {f(a)} | {f(b)} | "
                 f"{cm[0]:.0f}, {cm[1]:.0f} | {errors(S)} | {ok}")
    txt = "\n".join(L); print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
