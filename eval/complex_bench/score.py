"""Complex-query benchmark: score a system run (runs/<split>/part*.json) against the candidate answer sets.

Per query: returned set = union of the system's albums (the DISBench eval's rule) for one-album categories; for g
(two albums) the system's albums are matched to the two truth albums (best of the two orders; a missing album
scores 0) and the query's numbers are the mean over the two. h is scored on the LAST message (each earlier turn also
reported). f additionally reports hit = the half-remembered photo is among those returned, and returned size.
Both the FIRST answer (fast) and the FINAL round (judge has looked at every in-scope photo) are scored.
Usage: python eval/complex_bench/score.py <split> [run_dir]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402


def prf(got, gt):
    got, gt = set(got), set(gt)
    tp = len(got & gt)
    p = tp / len(got) if got else 0.0
    r = tp / len(gt) if gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp


def score_turn(cat, sys_albums, truth_albums):
    if cat == "g":
        T = truth_albums
        S = list(sys_albums) + [[]] * max(0, 2 - len(sys_albums))
        best = None
        for order in ([0, 1], [1, 0]):
            rows = [prf(S[order[k]], T[k]) for k in range(2)] if len(S) >= 2 else [prf(S[0], T[0]), prf([], T[1])]
            m = tuple(np.mean([r[j] for r in rows]) for j in range(3))
            if best is None or m[2] > best[2]:
                best = m
        return dict(p=best[0], r=best[1], f1=best[2], n_got=sum(len(s) for s in sys_albums), n_gt=sum(len(t) for t in T))
    got = set().union(*map(set, sys_albums)) if sys_albums else set()
    gt = set().union(*map(set, truth_albums)) if truth_albums else set()
    p, r, f1, tp = prf(got, gt)
    return dict(p=p, r=r, f1=f1, n_got=len(got), n_gt=len(gt), tp=tp, exact=got == gt)


def main():
    split = sys.argv[1]
    run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / split
    Q = {q["qid"]: q for q in json.loads((ROOT / "queries" / f"{split}.json").read_text())["queries"]}
    R = list({r["qid"]: r for f in sorted(run_dir.glob("*part*.json")) for r in json.loads(f.read_text())}.values())
    rows = []
    for r in R:
        q = Q.get(r["qid"])
        if q is None:
            continue
        rec = dict(qid=r["qid"], cat=q["cat"], user=q["user"], words=" || ".join(q["turns"]), error=r.get("error"),
                   verified=q.get("verified", True))
        for stage in ("first", "final"):
            if r.get("error") or not r["turns"] or len(r["turns"]) < len(q["turns"]) or r["turns"][-1][stage] is None:
                s = dict(p=0.0, r=0.0, f1=0.0, n_got=0, n_gt=sum(len(a) for a in q["answers"][-1]), exact=False)
            else:
                s = score_turn(q["cat"], r["turns"][-1][stage]["albums"], q["answers"][-1])
            if q["cat"] == "f" and not r.get("error") and r["turns"] and r["turns"][-1][stage]:
                got = set().union(*map(set, r["turns"][-1][stage]["albums"]))
                tgt = set(map(str, (q["program"][0] or {}).get("ids") or []))
                s["hit"] = bool(got & tgt)
            rec[stage] = s
        if q["cat"] == "h" and not r.get("error"):
            rec["turn_f1"] = [score_turn("h", t["final"]["albums"], a)["f1"] if t.get("final") else 0.0
                              for t, a in zip(r["turns"], q["answers"])]
        rows.append(rec)
    out = [f"Complex-query benchmark, split={split}: {len(rows)} of {len(Q)} queries scored ({run_dir})",
           "Truth = candidate answer sets (truth programs + Qwen3.5-27B oracle, audited by eye: audit/quality.txt).",
           "'verified' = categories a,b,d,e,f; c,g,h are provisional (their candidate sets miss the audit bar).", ""]
    out.append("cat | n | first P / R / F1 | final P / R / F1 | final: nothing returned | errors | extra")
    by = defaultdict(list)
    for x in rows:
        by[x["cat"]].append(x)
    for c in sorted(by) + ["ALL", "ALL verified"]:
        xs = rows if c == "ALL" else [x for x in rows if x["verified"]] if c == "ALL verified" else by[c]
        if not xs:
            continue
        m = lambda st, k: np.mean([x[st][k] for x in xs])
        extra = ""
        if c == "f":
            extra = f"hit first {sum(x['first'].get('hit', False) for x in xs)}/{len(xs)}, final {sum(x['final'].get('hit', False) for x in xs)}/{len(xs)}; median returned {np.median([x['final']['n_got'] for x in xs]):.0f}"
        if c == "h":
            tf = [x.get("turn_f1") for x in xs if x.get("turn_f1")]
            if tf:
                extra = "mean F1 by turn: " + ", ".join(f"t{k + 1} {np.mean([t[k] for t in tf if len(t) > k]):.3f}" for k in range(max(map(len, tf))))
        out.append(f"{c} | {len(xs)} | {m('first', 'p'):.3f} / {m('first', 'r'):.3f} / {m('first', 'f1'):.3f} | "
                   f"{m('final', 'p'):.3f} / {m('final', 'r'):.3f} / {m('final', 'f1'):.3f} | "
                   f"{sum(x['final']['n_got'] == 0 for x in xs)} | {sum(bool(x['error']) for x in xs)} | {extra}")
    (run_dir / "score.txt").write_text("\n".join(out))
    (run_dir / "scored.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(out))


if __name__ == "__main__":
    main()
