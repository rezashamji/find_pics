"""RESULTS 59, per-photo comparison on DEV truth questions: the 9B judge vs query-time scorers, both against the 27B
oracle (match = oracle P >= .8, non-match < .2, the band between dropped).
  python eval/facts_scorer/eval_photo.py name=eval/facts_scorer/models/<dir> ... [--out=file] [--set=dev]
Per method: pooled AUC, macro AUC over (library, question) with >= 1 match and >= 1 non-match, precision / recall /
F1 at cut .7, and precision at the 9B's recall (threshold where the method's recall equals the 9B's at .7, pooled).
The 9B row = eval/facts_scorer/judge_truth/Qwen3.5-9B. Breakdowns by benchmark category of the question and by
question kind (regex, KIND below; first match wins).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "eval/complex_bench")
from common import load_programs, whole_photo_q  # noqa: E402

COLOURS = ("red|orange|yellow|green|blue|purple|pink|magenta|teal|turquoise|brown|beige|grey|gray|black|white|golden|"
           "gold|silver|navy|violet|maroon|cyan|colou?r|colou?rful|neon|pastel")
KIND = [
    ("person (red box)", lambda q: q.startswith("PERSON[")),
    ("text / sign", lambda q: re.search(r"\b(text|sign|signs|signage|written|writing|reads?|says|words?|letters|logo|"
                                       r"menu|banner|label|poster|number plate)\b", q, re.I)),
    ("count", lambda q: re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|several|group of|crowd|many|"
                                 r"multiple|\d+|pair of|alone|only one|single|couple)\b", q, re.I)),
    ("photo look (quality/style)", lambda q: re.search(r"\b(blurr?y|blurred|out of focus|motion blur|black[- ]and[- ]white|"
                                                     r"monochrome|sepia|overexposed|underexposed|washed out|close-?up|"
                                                     r"long exposure|silhouette|reflection|aerial|panorama|bokeh|macro|"
                                                     r"wide|tilted|grainy|dark|bright|hazy|smog|foggy|fog|lit|lighting|"
                                                     r"light|sunlight|sunset|night)\b", q, re.I)),
    ("colour", lambda q: re.search(rf"\b({COLOURS})\b", q, re.I)),
    ("person attribute (no box)", lambda q: re.search(r"\b(person|man|woman|men|women|boy|girl|child|children|kid|kids|"
                                                     r"people|someone|bride|groom|baby)\b", q, re.I)),
    ("activity", lambda q: re.search(r"\b\w{3,}ing\b", q, re.I)),
    ("scene / place", lambda q: re.search(r"^(is|was) (this|the photo)\b|\b(taken|inside|indoors|outdoors|interior|"
                                          r"in a|at a|on a)\b", q, re.I)),
    ("object", lambda q: True),
]


def kind_of(q):
    for name, f in KIND:
        if f(q):
            return name


def question_cats(users):
    """(user, oracle key) -> set of categories whose truth programs (current or replaced versions) ask it."""
    out = defaultdict(set)
    for u in users:
        P = load_programs(u)
        for r in P["requests"]:
            progs = []
            for key in ("albums", "albums_v1"):
                progs += list(r.get(key) or [])
            for key in ("turn_programs", "turn_programs_v1"):
                for tp in r.get(key) or []:
                    progs += list(tp or [])
            for a in progs:
                if not isinstance(a, dict):
                    continue
                qs = []
                for kk in ("need", "need_any", "not"):
                    qs += [q.strip() for q in (a.get(kk) or []) if isinstance(q, str)]
                anc = a.get("anchor") or {}
                if isinstance(anc, dict):
                    qs += [q.strip() for q in (anc.get("need") or []) if isinstance(q, str)]
                for q in a.get("person_need") or []:
                    if isinstance(q, str):
                        qs.append(f"PERSON[{a['person']}]::{q.strip()}" if a.get("person") else whole_photo_q(q))
                for q in qs:
                    out[(u, q)].add(r["cat"])
    return out


def metrics(y, s, cut, groups=None, recall_at=None):
    o = {"n": int(len(y)), "pos": int(y.sum())}
    if len(np.unique(y)) < 2:
        return o
    o["auc"] = roc_auc_score(y, s)
    if groups is not None:
        a = []
        for g in np.unique(groups):
            m = groups == g
            if 0 < y[m].sum() < m.sum():
                a.append(roc_auc_score(y[m], s[m]))
        o["macro_auc"] = float(np.mean(a)) if a else np.nan
        o["n_q"] = len(a)
    yes = s >= cut
    tp = int((yes & (y == 1)).sum())
    o["P"] = tp / max(1, yes.sum()); o["R"] = tp / max(1, y.sum())
    o["F1"] = 2 * o["P"] * o["R"] / max(1e-9, o["P"] + o["R"])
    if recall_at is not None:
        thr = np.quantile(s[y == 1], 1 - recall_at) if y.sum() else np.inf
        yy = s >= thr
        o["P@R9B"] = int((yy & (y == 1)).sum()) / max(1, yy.sum())
    return o


def fmt(o):
    g = lambda k, f="{:.3f}": f.format(o[k]) if k in o and o[k] == o[k] else "-"
    return f"{g('auc')} | {g('macro_auc')} | {g('P')} | {g('R')} | {g('F1')} | {g('P@R9B')}"


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    cut = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cut=")), "0.7"))
    J = pd.concat([pd.read_parquet(f).assign(user=f.stem) for f in sorted(Path("eval/facts_scorer/judge_truth/Qwen3.5-9B").glob("*.parquet"))])
    J["item_id"] = J.item_id.astype(str)
    J = J.rename(columns={"p": "p9b"})
    base = None
    preds = {}
    def load_pred(d):
        if d.startswith("ens:"):     # equal-weight logit average of member models' predictions
            parts = [load_pred(x) for x in d[4:].split("+")]
            X = parts[0].rename(columns={"pred": "m0"})
            for k, y in enumerate(parts[1:], 1):
                X = X.merge(y[["user", "item_id", "q", "pred"]].rename(columns={"pred": f"m{k}"}), on=["user", "item_id", "q"])
            lg = sum(np.log(np.clip(X[f"m{k}"].to_numpy(), 1e-6, 1 - 1e-6) / (1 - np.clip(X[f"m{k}"].to_numpy(), 1e-6, 1 - 1e-6)))
                     for k in range(len(parts))) / len(parts)
            X["pred"] = 1 / (1 + np.exp(-lg))
            return X[["user", "item_id", "q", "p", "pred"]]
        d, fn = (d.split("::", 1) + ["pred_dev.parquet"])[:2] if "::" in d else (d, "pred_dev.parquet")
        X = pd.read_parquet(Path(d) / fn, columns=["user", "item_id", "q", "p", "pred"])
        X["item_id"] = X.item_id.astype(str)
        return X

    for name, d in runs:
        X = load_pred(d)
        X["item_id"] = X.item_id.astype(str)
        preds[name] = X.rename(columns={"pred": name})
        base = X[["user", "item_id", "q", "p"]] if base is None else base
    T = base.merge(J[["user", "item_id", "q", "p9b"]], on=["user", "item_id", "q"], how="inner")
    for name, X in preds.items():
        T = T.merge(X[["user", "item_id", "q", name]], on=["user", "item_id", "q"], how="inner")
    lab = np.where(T.p >= .8, 1, np.where(T.p < .2, 0, -1))
    T = T[lab >= 0].reset_index(drop=True); T["y"] = lab[lab >= 0]
    T["g"] = T.user + "|" + T.q
    QC = question_cats(sorted(T.user.unique()))
    T["cats"] = [QC.get((u, q), set()) for u, q in zip(T.user, T.q)]
    T["kind"] = T.q.map({q: kind_of(q) for q in T.q.unique()})
    methods = ["p9b"] + list(preds)
    lines = [f"Per-photo, DEV truth questions: {len(T)} pairs with oracle P >= .8 (match, {int(T.y.sum())}) or < .2 "
             f"(non-match), {T.g.nunique()} (library, question); cut {cut}; P@R9B = precision at the 9B's recall",
             "method | AUC | macro AUC | P | R | F1 | P@R9B"]

    def block(title, m):
        y = T.y.to_numpy()[m]; g = T.g.to_numpy()[m]
        o9 = metrics(y, T.p9b.to_numpy()[m], cut, g)
        lines.append(f"-- {title}: {int(m.sum())} pairs, {int(y.sum())} matches, {o9.get('n_q', 0)} questions with both")
        for meth in methods:
            o = metrics(y, T[meth].to_numpy()[m], cut, g, recall_at=o9.get("R"))
            lines.append(f"{'9B judge' if meth == 'p9b' else meth} | {fmt(o)}")
    block("ALL", np.ones(len(T), bool))
    for c in "abcdefgh":
        block(f"category {c}", T.cats.map(lambda s: c in s).to_numpy())
    for kname, _ in KIND:
        block(f"kind {kname}", (T.kind == kname).to_numpy())
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")
        T.drop(columns=["cats"]).to_parquet(Path(out_f).with_suffix(".parquet"))


if __name__ == "__main__":
    main()
