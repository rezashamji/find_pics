"""Scene / "X photos" searches keep too many wrong photos (RESULTS 34: beach 208/420, sunset 117/254, food 108/205 real).
Candidates, scored on (1) the 4 recall libraries of RESULTS 34 (stratified estimator: precision AND recall, same blind
labels) and (2) every earlier eye label of OTHER libraries, for all six "X photos" queries (generalisation: a rule
that fixes food but breaks flowers is not a rule the app can apply generally):
  (a) a higher cut on the planner's question;  (b) a stricter per-kind wording (food/beach/sunset only);
  (c) a second, generic question as a veto: keep = P(planner q) >= 0.7 AND P(second) >= s.
Judge: Qwen3-VL-4B phone 4-bit, same as RESULTS 34.
  gen  (vLLM env, GPU): python eval/eval_scene_fix.py gen --shard=k/K   -> eval/scene_fix/part{k}.parquet
  report (CPU):         python eval/eval_scene_fix.py report
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/scene_fix")
MODEL = os.path.abspath("models/qwen3vl_4b_mlx4sim")
X = {"food photos": "food", "beach photos": "a beach", "sunset photos": "a sunset", "photos of flowers": "flowers",
     "photos of a cat": "a cat", "photos of a church": "a church"}
RECALL_Q = ["beach photos", "sunset photos", "food photos"]
STRICT = {"food photos": "Is this a photo of food (a dish, meal or snack), not drinks or a storefront?",
          "beach photos": "Is this a photo of a sandy beach by the sea?",
          "sunset photos": "Does this photo show the sun setting or just set (not daytime backlight)?"}
GENERIC = {"q0": "Is this a photo of {x}?",
           "main": "Is {x} what this photo is mainly about, not just something present in the scene?",
           "desc": "Would most people describe this as a photo of {x}?"}


def questions(q):
    out = {k: v.format(x=X[q]) for k, v in GENERIC.items()}
    if q in STRICT:
        out["strict"] = STRICT[q]
    return out


def jobs():
    """(query, item_id, qname): all photos of the 4 recall libraries for the 3 recall queries (q0 is already in
    eval/recall_audit/scores) + every eye-labeled photo of the six queries outside those libraries."""
    from eval_question_variants import labels
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores").glob("part*.parquet"))])
    rec = sc[sc["query"].isin(RECALL_Q)]
    in4 = set(sc.item_id)
    J = [(q, i, n) for q in RECALL_Q for i in rec[rec["query"] == q].item_id for n in questions(q) if n != "q0"]
    held = sorted({(q, i) for (q, i) in labels() if q in X and i not in in4})
    J += [(q, i, n) for q, i in held for n in questions(q)]
    return J


def gen():
    os.environ["FP_VLM_MODEL"] = MODEL
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    J = pd.DataFrame(jobs(), columns=["query", "item_id", "qname"])
    J = J[J.item_id.map(lambda s: int(s) % K == k if s.isdigit() else hash(s) % K == k)]
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    J = J[J.item_id.isin(drow)]
    jud = VLLMJudge(model=MODEL, gpu_mem=0.8)
    out = []
    for (q, n), g in J.groupby(["query", "qname"]):
        rows = np.array([drow[i] for i in g.item_id], int)
        p = _judge_rows(di, jud, rows, np.full(len(rows), -1), questions(q)[n])
        out.append(g.assign(p=p))
        print(q, n, len(g), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(OUT / f"part{k}.parquet")


def rules(q):
    """name -> function(wide dataframe with columns q0, strict, main, desc) -> keep mask."""
    R = {f"q0>={t}": (lambda d, t=t: d.q0 >= t) for t in (0.7, 0.9, 0.95, 0.99)}
    if q in STRICT:
        R |= {f"strict>={t}": (lambda d, t=t: d.strict >= t) for t in (0.5, 0.7, 0.9)}
        R |= {"q0>=0.7 & strict>=0.5": lambda d: (d.q0 >= 0.7) & (d.strict >= 0.5)}
    for n in ("main", "desc"):
        R |= {f"q0>=0.7 & {n}>={s}": (lambda d, n=n, s=s: (d.q0 >= 0.7) & (d[n] >= s)) for s in (0.1, 0.3, 0.5, 0.7)}
    # combinations (added after the first report: the veto alone breaks flowers/church, a higher cut alone is mild)
    R |= {"q0>=0.99 | (q0>=0.7 & desc>=0.5)": lambda d: (d.q0 >= 0.99) | ((d.q0 >= 0.7) & (d.desc >= 0.5)),
          "q0>=0.95 & desc>=0.05": lambda d: (d.q0 >= 0.95) & (d.desc >= 0.05),
          "q0>=0.99 & desc>=0.05": lambda d: (d.q0 >= 0.99) & (d.desc >= 0.05),
          "mean(q0,desc)>=0.6": lambda d: (d.q0 >= 0.7) & ((d.q0 + d.desc) / 2 >= 0.6),
          "mean(q0,desc)>=0.75": lambda d: (d.q0 >= 0.7) & ((d.q0 + d.desc) / 2 >= 0.75)}
    return R


def report():
    from eval_question_variants import labels
    from eval_recall import load_scores, _labels_eye
    lab = labels()
    new = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("part*.parquet"))], ignore_index=True)
    W = new.pivot_table(index=["query", "item_id"], columns="qname", values="p").reset_index()
    lines = []
    # (1) the 4 recall libraries: stratified estimate (same strata/samples as RESULTS 34)
    df = load_scores()
    eye = _labels_eye()
    old = {k: {"right": "match", "wrong": "no_match", "unsure": "unsure"}[v] for k, v in lab.items()}
    rng = np.random.default_rng(0)
    B = 4000
    lines.append("## (1) 4 recall libraries (RESULTS 34 labels; unsure = no match). real kept / kept, recall [95% boot]")
    for q in RECALL_Q:
        d = df[df["query"] == q].merge(W[W["query"] == q].drop(columns="query"), on="item_id", how="left")
        d["q0"] = d.p
        d["exact"] = d.item_id.map(lambda i: (q, i) in old)
        d["lab"] = [eye.get((q, i)) or old.get((q, i)) for i in d.item_id]
        groups = [("exact", d[d.exact])] + [(s, d[(~d.exact) & (d.stratum == s)]) for s in ("kept", "A", "B", "C")]
        def est(keep):
            tot_hat, f_hat, tot_b, f_b = 0.0, 0.0, np.zeros(B), np.zeros(B)
            for s, g in groups:
                if s == "exact":
                    real = (g.lab == "match").to_numpy()
                    tot_hat += real.sum(); f_hat += (real & keep[g.index]).sum()
                    tot_b += real.sum(); f_b += (real & keep[g.index]).sum()
                    continue
                L = g[g.lab.notna()]
                N, n = len(g), len(L)
                if N == 0:
                    continue
                real = (L.lab == "match").to_numpy().astype(float)
                rk = real * keep[L.index].to_numpy()
                tot_hat += N * real.mean(); f_hat += N * rk.mean()
                ix = rng.integers(0, n, size=(B, n))
                tot_b += N * real[ix].mean(1); f_b += N * rk[ix].mean(1)
            return f_hat, tot_hat, f_b / tot_b
        R = rules(q)
        lines.append(f"{q}: real photos in the 4 libraries (est.) = {est(d.q0 >= 0)[1]:.1f}")
        for name, fn in R.items():
            keep = fn(d).fillna(False)
            f, tot, rb = est(keep)
            lo, hi = np.percentile(rb, [2.5, 97.5])
            lines.append(f"  {name:24s} kept {int(keep.sum()):4d}  real kept {f:6.1f}  precision {f / max(keep.sum(), 1):.2f}"
                         f"  recall {f / tot:.3f} [{lo:.3f}, {hi:.3f}]")
    # (2) every earlier eye label of other libraries, six queries
    lines.append("\n## (2) earlier eye labels, OTHER libraries: right kept / right, wrong kept / wrong (unsure kept / unsure)")
    for q in X:
        d = W[W["query"] == q].copy()
        d["lab"] = [lab.get((q, i)) for i in d.item_id]
        d = d[d.lab.notna()]
        n = d.lab.value_counts().to_dict()
        lines.append(f"{q}: {n}")
        for name, fn in rules(q).items():
            k = fn(d).fillna(False)
            c = {v: int((k & (d.lab == v)).sum()) for v in ("right", "wrong", "unsure")}
            lines.append(f"  {name:24s} right {c['right']:3d}/{n.get('right', 0)}  wrong {c['wrong']:3d}/{n.get('wrong', 0)}"
                         f"  unsure {c['unsure']:3d}/{n.get('unsure', 0)}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    {"gen": gen, "report": report}[sys.argv[1]]()
