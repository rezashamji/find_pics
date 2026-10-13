"""Complex-query benchmark, RESULTS 52: blind eye audit of category h (follow-up edits) BEFORE vs AFTER the h truth fix,
on a FRESH sample of requests (none of the 20 h requests audited on 10-12). Protocol: AUDIT_PROTOCOL.txt.

Versions compared (final message's answer = union of its albums):
  h1  : queries_h1/   the truths as committed before RESULTS 52 (programs as generated, 10-12 compile rules)
  h1c : queries_h1c/  same programs, RESULTS 52 compile rules only (build.py H_FIX)
  h2  : queries_h2c09/ RESULTS 52 rewritten programs (programs_h2/) + compile rules, oracle cut 0.9 (the h cut until 10-13)
  h2c05: queries/       the same at oracle cut 0.5 = the NEW TRUTH (adopted by the rule in AUDIT_PROTOCOL.txt)
Sampling (per request; each photo belongs to exactly one stratum, sizes recorded in the key):
  pattern strata: photos in at least one version's answer, split by membership pattern (in h1, h1c, h2, h2c05):
                  up to 4 from the pattern in all versions, up to 3 from every other non-empty pattern;
  non-answers (in no version): hard_in = the 20 most text-similar (PE-Core) inside the union of the versions' time/
                  place scopes (3 sampled), rand_in = the rest of that scope (1), hard_out = the 15 most similar outside
                  it (2), rand_out = the rest of the library (1).
The labeller sees only the words (requests.txt) and the sheets; never key.json or queries.
score: per version, stratified estimate pooled over requests: TP = sum(stratum size x share labelled match) over strata
inside the version's answer, FP likewise for no_match, FN = matches in strata outside it. unsure excluded. 95% bootstrap
over requests.
Usage (main env): python eval/complex_bench/audit3.py make [N=20] | score
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import scope_mask, sheet, tile  # noqa: E402
from build import Lib  # noqa: E402
from common import ROOT, city, load_items  # noqa: E402

A3 = ROOT / "audit3"
# folders as they are AFTER RESULTS 52 adopted cut 0.5 (the audit itself was made when queries/ held the cut-0.9 build;
# key.json records every set size per version and matches these folders: 0 mismatches). Rebuild the untracked ones with
#   queries_h1   = git show 9f456d8:eval/complex_bench/queries/{test,dev}.json
#   queries_h1c  = FP_CB_H2= FP_CB_QDIR=queries_h1c python eval/complex_bench/build.py
#   queries_h2c09= FP_CB_CUTS=a=0.7,b=0.8,c=0.5,d=0.8,e=0.5,f=0.9,g=0.5,h=0.9 FP_CB_QDIR=queries_h2c09 python .../build.py
VERS = [("h1", "queries_h1"), ("h1c", "queries_h1c"), ("h2", "queries_h2c09"), ("h2c05", "queries")]


def load(folder):
    out = {}
    for s in ("test", "dev"):
        f = ROOT / folder / f"{s}.json"
        for q in json.loads(f.read_text())["queries"]:
            if q["cat"] == "h":
                out[q["qid"]] = q
    return out


def final_set(q):
    return set().union(*map(set, q["answers"][-1])) if q else set()


def make(n_req):
    from findpics import store
    from findpics.models import ImageTextEncoder
    Q = {v: load(f) for v, f in VERS}
    old_audit = {k["qid"] for k in json.loads((ROOT / "audit" / "key.json").read_text()).values() if k["cat"] == "h"}
    both = sorted(set(Q["h1"]) & set(Q["h2"]) - old_audit)
    rng = random.Random(52)
    users = defaultdict(list)
    for qid in both:
        users[Q["h2"][qid]["user"]].append(qid)
    for u in users:
        rng.shuffle(users[u])
    pick = []
    while len(pick) < n_req and any(users.values()):
        for u in sorted(users):
            if users[u] and len(pick) < n_req:
                pick.append(users[u].pop())
    it = load_items()
    idx = store.load("data/public/index_disbench")
    row_of = {str(i): k for k, i in enumerate(idx.items.item_id.astype(str))}
    enc = ImageTextEncoder(idx.clip_model, device="cpu")
    X = idx.clip.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    unit_item = idx.units.item_row.to_numpy()
    TT = enc.texts([" ".join(Q["h2"][q]["turns"]) for q in pick]).astype(np.float32)
    TT /= np.linalg.norm(TT, axis=1, keepdims=True)
    A3.mkdir(exist_ok=True); (A3 / "sheets").mkdir(exist_ok=True)
    key, libs = {}, {}
    for n, (qid, T) in enumerate(zip(pick, TT)):
        q2 = Q["h2"][qid]; u = q2["user"]
        if u not in libs:
            libs.clear(); libs[u] = Lib(u, it)
        lib = libs[u]
        rows = np.array([row_of[i] for i in lib.ids])
        um = np.isin(unit_item, rows)
        simv = pd.DataFrame({"r": unit_item[um], "s": X[um] @ T}).groupby("r").s.max().reindex(rows).fillna(-1.0).to_numpy()
        sets = {v: final_set(Q[v].get(qid)) for v, _ in VERS}
        scope = np.zeros(len(lib.ids), bool)
        for v, _ in VERS:
            qv = Q[v].get(qid)
            if qv:
                for a in qv["program"][-1]:
                    if isinstance(a, dict):
                        scope |= scope_mask(lib, a, "h") if v == "h1" else (lib.meta_mask(a, loose=True) & _anchor_person(lib, a))
        strata = defaultdict(list)
        for i in lib.ids:
            pat = "".join("1" if i in sets[v] else "0" for v, _ in VERS)
            if "1" in pat:
                strata["p" + pat].append(i)
        none = np.where(~np.isin(lib.ids, list(set().union(*sets.values()))))[0]
        ins, outs = none[scope[none]], none[~scope[none]]
        ins = ins[np.argsort(-simv[ins])]; outs = outs[np.argsort(-simv[outs])]
        strata["hard_in"] = list(lib.ids[ins[:20]]); strata["rand_in"] = list(lib.ids[ins[20:]])
        strata["hard_out"] = list(lib.ids[outs[:15]]); strata["rand_out"] = list(lib.ids[outs[15:]])
        take = dict(hard_in=3, rand_in=1, hard_out=2, rand_out=1)
        items = []
        for s, pool in sorted(strata.items()):
            k = take.get(s, 4 if s == "p" + "1" * len(VERS) else 3)
            for i in rng.sample(sorted(pool), min(k, len(pool))):
                items.append((i, s))
        rng.shuffle(items)
        tiles = [tile(lib.L.path.iloc[lib.pos[i]], str(j), f"{lib.L.t.iloc[lib.pos[i]]:%Y-%m-%d %a %H:%M} | "
                      f"{city(lib.L.place.iloc[lib.pos[i]])}") for j, (i, _) in enumerate(items, 1)]
        name = f"r{n:03d}_h"
        for s in range(0, len(tiles), 6):
            sheet(tiles[s:s + 6]).save(A3 / "sheets" / f"{name}_{s // 6}.jpg", quality=92)
        key[name] = dict(qid=qid, user=u, split=q2["split"], turns=q2["turns"], today=q2["today"],
                         items=[dict(n=j, item_id=i, stratum=s) for j, (i, s) in enumerate(items, 1)],
                         pops={s: len(p) for s, p in strata.items()}, sizes={v: len(x) for v, x in sets.items()},
                         present={v: qid in Q[v] for v, _ in VERS})
        print(name, qid, len(items), key[name]["pops"], flush=True)
    (A3 / "key.json").write_text(json.dumps(key, indent=1))
    with open(A3 / "requests.txt", "w") as fh:
        for name, k in key.items():
            fh.write(f"{name} [h] (today {k['today']}) " + " || ".join(k["turns"]) + "\n")


def _anchor_person(lib, a):
    m = np.ones(len(lib.ids), bool)
    anc = a.get("anchor")
    if isinstance(anc, dict) and anc.get("window"):
        rows = [lib.pos[str(i)] for i in anc.get("ids") or [] if str(i) in lib.pos]
        if rows:
            m &= lib.window(rows, anc["window"])
    if a.get("person"):
        m &= np.isin(lib.ids, list(lib.persons.get(a["person"], {})))
    return m


def labels(pattern="labels_*.txt"):
    L = {}
    for f in sorted(A3.glob(pattern)):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(None, 3)
            if len(p) >= 3:
                L[(p[0], p[1])] = p[2].strip().lower()
    return L


def estimate(recs, pops, v_idx, names):
    TP = FP = FN = 0.0
    for name in names:
        for s, pop in pops[name].items():
            R = [lab for (nm, st, lab) in recs if nm == name and st == s]
            if not R or not pop:
                continue
            inV = s.startswith("p") and s[1 + v_idx] == "1"
            m = sum(l == "match" for l in R) / len(R)
            if inV:
                TP += pop * m; FP += pop * (1 - m)
            else:
                FN += pop * m
    return TP, FP, FN


def score():
    key = json.loads((A3 / "key.json").read_text())
    L = labels()
    recs, pops = [], {}
    unsure = unl = 0
    raw = defaultdict(lambda: [0, 0])
    _sp = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--split=")), "all")
    _hf = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--half=")), "all")
    sel = {n for n in key if (_sp == "all" or key[n]["split"] == _sp) and (_hf == "all" or (int(n[1:4]) % 2 == 0) == (_hf == "even"))}
    for name, k in key.items():
        if name not in sel:
            continue
        pops[name] = k["pops"]
        for x in k["items"]:
            lab = L.get((name, str(x["n"])))
            if lab is None:
                unl += 1; continue
            if lab == "unsure":
                unsure += 1; continue
            recs.append((name, x["stratum"], lab))
            raw[x["stratum"]][0] += lab == "match"; raw[x["stratum"]][1] += 1
    SPLIT = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--split=")), "all")
    HALF = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--half=")), "all")   # even | odd request index
    names = sorted(n for n in key if (SPLIT == "all" or key[n]["split"] == SPLIT) and
                   (HALF == "all" or (int(n[1:4]) % 2 == 0) == (HALF == "even")))
    rng = np.random.default_rng(0)
    out = ["RESULTS 52: blind eye audit of category h on a fresh sample (audit3/), final message, stratified estimate.",
           f"{len(names)} requests ({sum(key[n]['split'] == 'test' for n in names)} test, "
           f"{sum(key[n]['split'] == 'dev' for n in names)} dev); labelled photos {len(recs)} (+{unsure} unsure excluded, "
           f"{unl} unlabelled).", "",
           "version | precision [95% CI] | recall [95% CI] | F1 | est. right / wrong / missed photos"]
    for vi, (v, _) in enumerate(VERS):
        names_v = [n for n in names if key[n]["present"][v]]   # a version that DROPPED the request is not scored on it
        TP, FP, FN = estimate(recs, pops, vi, names_v)
        P, R = TP / (TP + FP) if TP + FP else 0, TP / (TP + FN) if TP + FN else 0
        bs = []
        for _ in range(2000):
            nm = list(rng.choice(names_v, len(names_v)))
            # duplicate requests in the resample are counted with multiplicity
            tp = fp = fn = 0.0
            for x in nm:
                a, b, c = estimate(recs, pops, vi, [x]); tp += a; fp += b; fn += c
            bs.append((tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0))
        bs = np.array(bs)
        pl, ph = np.percentile(bs[:, 0], [2.5, 97.5]); rl, rh = np.percentile(bs[:, 1], [2.5, 97.5])
        out.append(f"{v} ({len(names_v)} req) | {P:.2f} [{pl:.2f}, {ph:.2f}] | {R:.2f} [{rl:.2f}, {rh:.2f}] | {2 * P * R / (P + R) if P + R else 0:.2f} | "
                   f"{TP:.0f} / {FP:.0f} / {FN:.0f}")
    out += ["", "raw labels per stratum (match / labelled; pattern = in h1, h1c, h2, h2c05):"]
    out += [f"  {s}: {m}/{n}" for s, (m, n) in sorted(raw.items())]
    ov = labels("overlap_main.txt")
    agree = tot = 0; dis = []
    for k_, lab in ov.items():
        other = L.get(k_)
        if other is None or "unsure" in (lab, other):
            continue
        tot += 1; agree += lab == other
        if lab != other:
            dis.append(f"  {k_[0]} {k_[1]}: main={lab} subagent={other}")
    out.append(f"\nmain agent vs subagents (overlap, unsure excluded): {agree}/{tot}")
    out += dis
    out[0] += f" [split: {SPLIT}; half: {HALF}]"
    tag = "" if SPLIT == "all" and HALF == "all" else f"_{SPLIT}_{HALF}"
    (A3 / f"quality{tag}.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        score()
