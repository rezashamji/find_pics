"""RESULTS 69 (1) TRACE: which pass-A leaf scopes of the measured R59d cascade came from an engine step whose rows depend
on judge ANSWERS (and so on pass A's judge-everything answers). Reads the RESULTS 68 recordings (config 100_N_A = the
measured 100%-facts R59d) and the stored plans; no search-quality number.
Role of the engine call that put photos into a leaf's scope (matched by question text against the plan):
  main      the album's own question, on the library (stream: head doubles until every in-scope photo is asked ->
            rows independent of answers)
  main@win  the album's own question inside an ANCHOR window (store.subset): rows = the window = f(anchor answers)
  anchor / until  the anchor / until album (stream=False: head 600 + extensions while >= 3% yes + 300 random tail ->
            rows depend on answers through the extension rule)
  filter / exclude  asked on the album's RETURNED photos (converse._album_stream, agent.execute) -> rows = f(answers)
  person-box   person-mode look on face matches (ident + maybe, one call each) -> independent
  python eval/scope69/trace.py [--out=eval/scope69/trace.txt]
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = {"dev": "eval/complex_bench/runs/cmp/r60v2_base/plan/plans.json",
         "train": "eval/facts_scorer/replay/train_plans/plans_r60.json"}
BOX = re.compile(r"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)")


def roles(plan):
    r = {}
    for t in plan.get("turns") or []:
        for a in t["plan"]["albums"]:
            for k, q in (("exclude", a.get("exclude_question")), ("filter", a.get("filter_question")),
                         ("until", (a.get("until") or {}).get("judge_question")),
                         ("anchor", (a.get("anchor") or {}).get("judge_question"))):
                if q:
                    r.setdefault(q, set()).add(k)
            q = a.get("judge_question")
            if q:
                r.setdefault(q, set()).add("main@win" if a.get("anchor") else "main")
                if a.get("person"):
                    r.setdefault(BOX.sub("the person in the red box", q), set()).add("person-box")
    return r


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    L = ["RESULTS 69 (1) trace: pass-A judge-leaf scopes of the measured R59d (100_N_A recordings) by the engine step "
         "that asked them"]
    for s in ("dev", "train"):
        plans = json.loads((ROOT / PLANS[s]).read_text())["plans"]
        C, P, EQ = Counter(), Counter(), Counter()
        req_dep = Counter()
        for f in sorted((ROOT / f"data/public/cascade_port/record/{s}/100_N_A").glob("rec*.jsonl.gz")):
            for line in gzip.open(f, "rt"):
                r = json.loads(line)
                R = roles(plans.get(r["qid"]) or {})
                # whole question of each pass-A call -> role; a leaf inherits the roles of the calls whose tree holds it
                call_role = []
                for c in r["A"]:
                    rr = set(R.get(c["q"], set()))
                    if c["excl"]:
                        rr.add("exclude")
                    if c["sub"] != -1 and "main" in rr:
                        rr = (rr - {"main"}) | {"main@win"}
                    if c["crop"] == "box":
                        rr.add("person-box")
                    call_role.append((c["q"], frozenset(rr) or frozenset({"other"})))
                Bask = defaultdict(set)
                for c in r["B"]:
                    Bask[c["q"]].update(c["rows"])
                dep_any = False
                for key, sc in r["scope"].items():
                    lq, crop = key.rsplit("|", 1)
                    # the whole questions whose calls can hold this leaf: equal, or containing the leaf text
                    rs = set()
                    for q, rr in call_role:
                        if q == lq or lq in q or (lq.rstrip("?") in q):
                            rs |= rr
                    rs = rs or {"other"}
                    dep = bool(rs & {"main@win", "anchor", "until", "filter", "exclude", "other"})
                    dep_any |= dep
                    tag = "+".join(sorted(rs))
                    C[tag] += 1; P[tag] += len(sc)
                    asked = set()
                    for q, rows in Bask.items():
                        if q == lq or lq in q or lq.rstrip("?") in q:
                            asked |= rows
                    A_ = {x[0] for x in sc}
                    EQ[(tag, "B asks = A scope" if asked == A_ else "B asks subset" if asked < A_ else "B asks outside")] += 1
                req_dep["requests"] += 1
                req_dep["with an answer-dependent judge leaf"] += dep_any
                req_dep["with any judge leaf"] += bool(r["scope"])
        L.append(f"\n{s}: {dict(req_dep)}")
        L.append("role(s) of the leaf | leaves | scope photos (sum) | B asks = A scope / subset / outside")
        for tag, n in C.most_common():
            L.append(f"  {tag} | {n} | {P[tag]:,} | {EQ[(tag, 'B asks = A scope')]} / {EQ[(tag, 'B asks subset')]} / "
                     f"{EQ[(tag, 'B asks outside')]}")
    txt = "\n".join(L)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
