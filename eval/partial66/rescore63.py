"""RESULTS 66 step 1: re-score the 61d / 65 reference runs on the ADOPTED RESULTS-63 truth (queries/{dev,train}.json since
10-16; pre-63 files in queries_pre63/). No new runs: part files are copied to eval/partial66/base63/<split>/<name> (the
source dirs are not written to) and scored with eval/complex_bench/score.py against queries/<split>.json.
Overlap check vs eval/complex_bench/t63/rescore/new (same runs, same truth): all-judge and the TRAIN 61d run must be equal.
  python eval/partial66/rescore63.py [--out=eval/partial66/base63.txt]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "eval/partial66/base63"
SRC = {
    "train": {"alljudge": "eval/facts_scorer/replay/train_alljudge",
              "61d_t2b_xfit": "eval/facts_writer/replay/c61d_train_t2b_final_rs2_xfit/r59d",
              **{f"65_{c}": f"eval/partial65/replay/train/{c}" for c in
                 ("100_N_A", "0_N_A", "0_N_V", "10_N_V", "25_N_V", "50_N_V", "75_N_V", "10_N_A", "25_N_A", "50_N_A",
                  "75_N_A")}},
    "dev": {"alljudge": "eval/complex_bench/runs/cmp/r60v2_base_search",
            "61d_t2b_xfit": "eval/facts_writer/replay/c61d_dev_t2b_final_rs2_xfit/r59d",
            "61d_p2b_product": "eval/facts_writer/replay/c61d_dev_p2b_devmix_rs2/r59d",
            **{f"65_{c}": f"eval/partial65/replay/dev/{c}" for c in
               ("100_N_A", "0_N_A", "0_N_V", "10_N_V", "25_N_V", "50_N_V", "75_N_V", "10_N_A", "25_N_A", "50_N_A",
                "75_N_A")}},
}
T63 = {("train", "alljudge"): "alljudge_train", ("train", "61d_t2b_xfit"): "ship_train",
       ("dev", "alljudge"): "alljudge_dev", ("dev", "61d_t2b_xfit"): "ship_dev"}


def f1s(d):
    return {r["qid"]: r["final"]["f1"] for r in json.loads((d / "scored.json").read_text())}


def boot(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    L = ["RESULTS 66 step 1: reference runs re-scored on the RESULTS-63 truth (queries/{dev,train}.json); final-round ALL F1",
         "split | run | n | ALL F1 | diff vs all-judge [95%] | ALL verified (a,d,e,f,g,h) n, F1 | t63/rescore/new same run"]
    for s, runs in SRC.items():
        F = {}
        for n, src in runs.items():
            d = B / s / n
            d.mkdir(parents=True, exist_ok=True)
            for f in (ROOT / src).glob("*part*.json"):
                shutil.copy(f, d / f.name)
            subprocess.run([sys.executable, "eval/complex_bench/score.py", s, str(d.relative_to(ROOT))], cwd=ROOT,
                           check=True, stdout=subprocess.DEVNULL)
            F[n] = f1s(d)
        Q = {q["qid"]: q for q in json.loads((ROOT / f"eval/complex_bench/queries/{s}.json").read_text())["queries"]}
        ref = F["alljudge"]
        for n, v in F.items():
            qs = sorted(v)
            assert qs == sorted(ref), f"{s} {n}: {len(qs)} vs {len(ref)}"
            ver = [q for q in qs if Q[q].get("verified", True)]
            t = boot([v[q] - ref[q] for q in qs])
            chk = ""
            if (s, n) in T63:
                o = f1s(ROOT / "eval/complex_bench/t63/rescore/new" / T63[(s, n)])
                same = sorted(o) == qs and all(abs(o[q] - v[q]) < 1e-12 for q in qs)
                chk = f"{np.mean(list(o.values())):.4f} {'IDENTICAL' if same else 'DIFFERENT'}"
            L.append(f"{s} | {n} | {len(qs)} | {np.mean([v[q] for q in qs]):.4f} | {t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}] | "
                     f"{len(ver)}, {np.mean([v[q] for q in ver]):.4f} | {chk}")
    txt = "\n".join(L)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
