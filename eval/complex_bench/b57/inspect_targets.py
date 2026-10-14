"""RESULTS 57: what the B3 SFT targets look like before training (checked.json of sft_from_truth.py).
(1) pass / fail per category and the failure reasons; (2) plan SHAPES of the passing targets per category vs the key
(the B1 errors: one described thing split into judge + filter = AND-ed descriptions; an anchor + same_event window for a
named trip or event the key gives as days/place = wide trip windows); (3) --sample=N: N passing targets, one per request,
fixed seed, printed compactly (words, key, grounded plan) for the eye check.
Usage: python eval/complex_bench/b57/inspect_targets.py <checked.json> [--sample=25] [--seed=0] [--cmp=<old checked.json>]
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict

arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
C = json.load(open(sys.argv[1]))


def shapes(C, label):
    ok = [c for c in C if c["ok"]]
    by = defaultdict(list)
    for c in ok:
        by[c["cat"]].append(c)
    print(f"\n== {label}: {len(ok)} of {len(C)} turns pass")
    print("cat  pass/all  filter  anchor  key_anchor  anchor_wo_key  key_need>=2  plan_q>=2  excl  key_excl  place  dates")
    for cat in sorted(set(c["cat"] for c in C)):
        allc = [c for c in C if c["cat"] == cat]
        xs = by[cat]
        n = max(len(xs), 1)
        A = [a for c in xs for a in c["grounded"]["albums"]]
        K = [k for c in xs for k in c["key"] if isinstance(k, dict)]
        na = max(len(A), 1)
        f = sum(bool(a.get("filter_question")) for a in A) / na
        an = sum(bool(a.get("anchor")) for a in A) / na
        ka = sum(isinstance(k.get("anchor"), dict) for k in K) / max(len(K), 1)
        awk = sum(1 for c in xs for a, k in zip(c["grounded"]["albums"], c["key"])
                  if a.get("anchor") and not (isinstance(k, dict) and isinstance(k.get("anchor"), dict)))
        kn2 = sum(len(k.get("need") or []) >= 2 for k in K) / max(len(K), 1)
        pq2 = sum(bool(a.get("judge_question")) and bool(a.get("filter_question")) for a in A) / na
        ex = sum(bool(a.get("exclude_question")) for a in A) / na
        kex = sum(bool(k.get("not")) for k in K) / max(len(K), 1)
        pl = sum(bool(a.get("place")) for a in A) / na
        dt = sum(bool(a.get("date_from") or a.get("date_to")) for a in A) / na
        print(f"{cat}   {len(xs):4d}/{len(allc):<4d}  {f:.2f}    {an:.2f}    {ka:.2f}        {awk:3d}            "
              f"{kn2:.2f}         {pq2:.2f}       {ex:.2f}  {kex:.2f}      {pl:.2f}   {dt:.2f}")
    fails = Counter()
    for c in C:
        if not c["ok"]:
            for p in c["problems"]:
                fails[re.sub(r"'[^']*'|\d{4}-\d\d-\d\d|\[[^\]]*\]|\{[^}]*\}", "_", p)[:70]] += 1
    print("failure reasons (problems, a turn may have several):")
    for k, v in fails.most_common(15):
        print(f"  {v:4d}  {k}")


shapes(C, sys.argv[1])
if arg("cmp"):
    shapes(json.load(open(arg("cmp"))), arg("cmp"))

N = int(arg("sample", "0"))
if N:
    rng = random.Random(int(arg("seed", "0")))
    ok = [c for c in C if c["ok"]]
    qids = sorted({c["qid"] for c in ok})
    rng.shuffle(qids)
    pick = []
    for q in qids:
        cs = [c for c in ok if c["qid"] == q]
        pick.append(cs[-1])      # the last passing turn of the request (h: the follow-up)
        if len(pick) >= N:
            break
    keep = ("name", "person", "judge_question", "filter_question", "exclude_question", "anchor", "window", "place",
            "time_phrase", "date_from", "date_to", "media", "want", "with_people")
    for i, c in enumerate(pick):
        print(f"\n#{i + 1} {c['qid']} cat {c['cat']} turn {c['turn']}")
        for h in c.get("history") or []:
            print(f"   earlier: {h}")
        print(f"   WORDS: {c['message']}")
        print(f"   KEY:   {json.dumps(c['key'], ensure_ascii=False)}")
        for a in c["grounded"]["albums"]:
            g = {k: a[k] for k in keep if a.get(k) not in (None, [], "", "any", "all")}
            if isinstance(g.get("anchor"), dict):
                g["anchor"] = g["anchor"].get("judge_question")
            print(f"   PLAN:  {json.dumps(g, ensure_ascii=False)}")
