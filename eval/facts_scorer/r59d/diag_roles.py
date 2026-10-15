"""RESULTS 59d design diagnostic, TRAIN ONLY (reads train_ms_pairs.parquet from diag_ms.py): agreement of the millisecond
scorer (4B facts, fold models) with the 9B on every pair the all-judge TRAIN stack asked, by the question's ROLE in the
stored plan (album question / filter / exclude / anchor / other) and by category; only questions R59c routes to the
SCORER (59b routing: not a pixel kind). Then the judge BUDGET on judge-routed questions: the 9B's yes rate among the top
max(50, 5%) of the scope by the scorer, and the share of the 9B's matches the budget can reach.
  python eval/facts_scorer/r59d/diag_roles.py [cats=h]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/facts_scorer")
from route import TRIP, _TAIL, budget, route  # noqa: E402

cats = sys.argv[1] if len(sys.argv) > 1 else "abcdefgh"
R = pd.read_parquet("eval/facts_scorer/r59d/train_ms_pairs.parquet")
R["cat"] = R.qid.str.split(":").str[1].str[0]
R = R[R.cat.isin(list(cats))]
plans = json.loads(Path("eval/facts_scorer/replay/train_plans/plans_r60.json").read_text())["plans"]
role = {}
for qid, sp in plans.items():
    for t in sp.get("turns") or []:
        for a in t["plan"]["albums"]:
            for f, k in (("judge_question", "album"), ("filter_question", "filter"), ("exclude_question", "exclude")):
                if a.get(f):
                    role.setdefault((qid, a[f]), k)
            if a.get("anchor") and a["anchor"].get("judge_question"):
                role.setdefault((qid, a["anchor"]["judge_question"]), "anchor")
            if a.get("until") and a["until"].get("judge_question"):
                role.setdefault((qid, a["until"]["judge_question"]), "anchor")
R["role"] = [role.get((q, s), "other") for q, s in zip(R.qid, R.question)]
R["route"] = [route(q, c) or ("trip" if TRIP.search(_TAIL.sub("", q)) else "") for q, c in zip(R.question, R.crop)]
R["y9"] = R.p >= .7
R["ys"] = R.p_ms >= .7
S = R[R.route == ""]
print(f"SCORER-routed pairs (cats {cats}): {len(S):,}")
print("role | pairs | questions | 9B yes | scorer yes | agree | scorer yes & 9B no | scorer no & 9B yes | F1 vs 9B")
for k, g in S.groupby("role"):
    tp, fp, fn = (g.ys & g.y9).sum(), (g.ys & ~g.y9).sum(), (~g.ys & g.y9).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"{k} | {len(g):,} | {g.groupby(['qid', 'question']).ngroups} | {g.y9.mean():.3f} | {g.ys.mean():.3f} | "
          f"{(g.ys == g.y9).mean():.3f} | {fp:,} | {fn:,} | {f1:.3f}")
print("\nper category x role, scorer-routed: F1 vs 9B (pairs)")
for c, g0 in S.groupby("cat"):
    print(c, " ".join(f"{k}: {2 * (g.ys & g.y9).sum() / max(2 * (g.ys & g.y9).sum() + (g.ys != g.y9).sum(), 1):.2f} "
                      f"({len(g)})" for k, g in g0.groupby("role")))
print("\nJUDGE-routed questions: is the budget max(50, 5% of scope) big enough? (scope = the pairs the all-judge stack "
      "asked for that request's question; order = scorer P)")
J = R[R.route != ""]
rows = []
for (qid, q, cr), g in J.groupby(["qid", "question", "crop"]):
    n = len(g); M = budget(n)
    g = g.sort_values("p_ms", ascending=False)
    top = g.iloc[:M]
    rows.append(dict(qid=qid, cat=qid.split(":")[1][0], role=role.get((qid, q), "other"), q=q, n=n, M=M,
                     yes_all=int(g.y9.sum()), yes_top=int(top.y9.sum()), rate_top=top.y9.mean(),
                     last50=top.y9.iloc[-min(50, M):].mean()))
B = pd.DataFrame(rows)
B["reach"] = B.yes_top / B.yes_all.clip(lower=1)
B["sat"] = B.M < B.n
for k, g in B[B.sat].groupby("role"):
    print(f"{k}: {len(g)} budget-limited questions; 9B matches {g.yes_all.sum():,}, inside budget {g.yes_top.sum():,} "
          f"({g.yes_top.sum() / max(g.yes_all.sum(), 1):.2f}); questions whose budget's last 50 are >= 50% yes: "
          f"{(g.last50 >= .5).sum()} (their matches {g[g.last50 >= .5].yes_all.sum():,}, inside {g[g.last50 >= .5].yes_top.sum():,})")
print("\nbudget-limited questions with the most 9B matches outside the budget:")
B["out"] = B.yes_all - B.yes_top
for _, r in B[B.sat].sort_values("out", ascending=False).head(25).iterrows():
    print(f"  {r.qid:22s} {r.role:7s} n {r.n:5d} M {r.M:3d} 9B yes {r.yes_all:4d} in budget {r.yes_top:3d} "
          f"last50 yes {r.last50:.2f}  {r.q[:110]}")
