"""RESULTS 70 table + pre-registered decision (JOURNAL 10-21 "RESULTS 70 PRE-REGISTRATION").
  python eval/trunc70/table70.py [--out=eval/trunc70/table.txt]
Runs eval/trunc70/replay/<split>/DRY_fp32[_F<tag>] (phone config: DRY scope, fp32, policy V, 100% coverage; lines capped
at K writer tokens, re-encoded by the unchanged towervec). DRY_fp32 = the uncapped control in the same jobs (must match
RESULTS 69 DRY_fp32 returned sets on >= 98% of requests). vs all-judge (eval/scope69/base/<split>/alljudge) and vs the
control; ALL F1 final round, RESULTS-63 truth; paired bootstrap 2,000, seed 0; judge calls = pass A (asserted 0) + pass B.
Tokens = writer tokens decoded per photo with a line (min(n_out, K); truncate.py), over the split's libraries."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scope69"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scope69b"))
from table69 import ROOT, SPLITS, arg, boot, fmt, returned, stats  # noqa: E402
from table69b import f1s  # noqa: E402

SRC = {"train": "t2b_final_rs2_xfit", "dev": "p2b_devmix_rs2"}
S = json.loads((ROOT / "eval/complex_bench/splits.json").read_text())
TAGS = [x for x in arg("tags", "k16,k24,k32,k48,k64,full").split(",") if x]


def tokens(split, tag):
    import pandas as pd
    T = pd.read_parquet(ROOT / "eval/facts_scorer/data/facts_w_t2b_final_rs2_xfitm.parquet", columns=["user", "item_id"])
    ids = set(T.item_id.astype(str)[T.user.isin(S[split])])
    if tag == "full":
        v = []
        for u in S[split]:
            for line in open(ROOT / "eval/facts_writer/facts" / SRC[split] / f"{u}.jsonl"):
                r = json.loads(line)
                if (r.get("facts") or {}).get("caption"):
                    v.append(r["n_out"])
        return np.array(v, float)
    t = json.loads((ROOT / f"data/public/trunc70/facts/{SRC[split]}_{tag}/.tokens.json").read_text())
    return np.array([x for i, x in t.items() if i in ids], float)


def main():
    F, C, ERR, TOK = {}, {}, {}, {}
    for s in SPLITS:
        F[(s, "aj")] = f1s(f"eval/scope69/base/{s}/alljudge", s)
        F[(s, "r69")] = f1s(f"eval/scope69/replay/{s}/DRY_fp32", s)
        for tg in TAGS:
            d = f"eval/trunc70/replay/{s}/" + ("DRY_fp32" if tg == "full" else f"DRY_fp32_F{tg}")
            F[(s, tg)] = f1s(d, s)
            st = stats(d)
            assert all(v.get("judgeA", 0) == 0 for v in st.values()), f"{d}: DRY pass A judged"
            C[(s, tg)] = {q: v["judge"] + v.get("judgeA", 0) for q, v in st.items()}
            ERR[(s, tg)] = sum(1 for f in (ROOT / d).glob("part*.json") for r in json.loads(f.read_text()) if r.get("error"))
            TOK[(s, tg)] = tokens(s, tg)
    qs = {s: sorted(F[(s, "aj")]) for s in SPLITS}
    for k, v in F.items():
        assert sorted(v) == qs[k[0]], f"{k}: {len(v)} vs {len(qs[k[0]])} (incomplete run)"

    def diff(s, a, b):
        return boot([F[(s, a)][q] - F[(s, b)][q] for q in qs[s]])

    def calls(s, c):
        v = np.array([C[(s, c)][q] for q in qs[s]], float)
        return f"{v.mean():.0f} / {np.median(v):.0f} / {np.percentile(v, 90):.0f}"
    L = ["RESULTS 70: writer line capped at K tokens; phone config (DRY fp32, policy V, 100%); ALL F1 final round, RESULTS-63 truth",
         f"all-judge ALL F1: TRAIN {np.mean(list(F[('train', 'aj')].values())):.4f}, dev {np.mean(list(F[('dev', 'aj')].values())):.4f}"]
    for s in SPLITS:
        a, b = returned(f"eval/trunc70/replay/{s}/DRY_fp32"), returned(f"eval/scope69/replay/{s}/DRY_fp32")
        same = sum(a[q] == b[q] for q in a)
        L.append(f"control check {s}: uncapped DRY_fp32 returned sets = RESULTS 69 DRY_fp32 on {same}/{len(a)} requests "
                 f"({'OK' if same >= .98 * len(a) else 'FAIL'}); F1 diff {fmt(diff(s, 'full', 'r69'))}")
    L += ["", "K | split | tokens/photo mean (p90) | saved vs full | ALL F1 | vs all-judge [95%] | vs uncapped control [95%] | "
          "judge calls mean / median / p90 | errors"]
    passes = {}
    for tg in TAGS:
        lows = []
        for s in SPLITS:
            t = diff(s, tg, "aj"); lows.append(t[1]); tk = TOK[(s, tg)]
            L.append(f"{tg} | {s} | {tk.mean():.1f} ({np.percentile(tk, 90):.0f}) | {TOK[(s, 'full')].mean() - tk.mean():.1f} | "
                     f"{np.mean([F[(s, tg)][q] for q in qs[s]]):.4f} | {fmt(t)} | {fmt(diff(s, tg, 'full'))} | {calls(s, tg)} | {ERR[(s, tg)]}")
        passes[tg] = min(lows) >= -.03
        L.append(f"   {tg}: lowers TRAIN {lows[0]:+.4f}, dev {lows[1]:+.4f} -> {'PASS' if passes[tg] else 'no'}")
    ks = [tg for tg in TAGS if tg.startswith("k") and passes[tg]]
    sm = min(ks, key=lambda x: int(x[1:])) if ks else None
    L += ["", f"SMALLEST PASSING K (truncation): {sm}"]
    rs = [tg for tg in TAGS if tg.startswith("r") and tg != "rfull" and passes[tg]]
    if any(tg.startswith("r") for tg in TAGS):
        L.append(f"SMALLEST PASSING K (reordered): {min(rs, key=lambda x: int(x[1:])) if rs else None}")
    txt = "\n".join(L)
    print(txt)
    if arg("out"):
        Path(arg("out")).write_text(txt + "\n")


if __name__ == "__main__":
    main()
