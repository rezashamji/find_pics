"""Complex-query benchmark, RESULTS 53: blind eye audit of category g (two albums) BEFORE vs AFTER the g truth fix, on a
FRESH sample of requests (none of the 20 g requests audited on 10-12). Protocol: AUDIT_PROTOCOL.txt (g labels: album0 |
album1 | neither | unsure; album0 = the FIRST group the request names).

Versions compared (the request's two albums):
  g1   : queries_g1/    the truths as committed before RESULTS 53 (= git 6934daa; build.py's g compile changes made in
                        RESULTS 53 change no g truth, checked on every split, so there is no separate "g1c")
  g2   : queries/       RESULTS 53 rewritten programs (programs_g2/), oracle cut 0.5 (the g cut since 10-12)
  g2c03: queries_g2c03/ the same at oracle cut 0.3 (the alternative cut; adopted only by the rule in AUDIT_PROTOCOL.txt)
Sampling per request; every photo belongs to exactly one stratum: membership pattern over the versions, one character
per version: 0 = in no album, a = album0, b = album1, c = both albums. Up to 4 photos from the pattern every version agrees
on (most common non-zero), 3 from each other non-zero pattern; non-answers of every version: 3 of the 20 most
text-similar inside the union of the versions' time/place/person scopes, 1 other in scope, 2 of the 15 most similar outside
it, 1 other. Sheets put the named person's reference crops first (tiles "R") when an album names Sam/Alex.
score: per version, stratified estimate pooled over requests: a photo in album k of V is right if labelled album k (in
both albums: right if labelled either), wrong otherwise (unsure excluded); a photo in no album of V is a miss if labelled
album0/album1. 95% bootstrap over requests. --split=test|dev, --half=even|odd filter the requests.
Usage (main env): python eval/complex_bench/audit4.py make [N=20] | score
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from audit import TILE, font, scope_mask, sheet, tile  # noqa: E402
from audit3 import _anchor_person  # noqa: E402
from build import Lib  # noqa: E402
from common import ROOT, city, load_items  # noqa: E402

A4 = ROOT / "audit4"
# folders as they are after RESULTS 53 (rebuild the untracked ones with build.py: queries_g1 = git show
# 6934daa:eval/complex_bench/queries/{test,dev}.json; queries_g2c03 = FP_CB_CUTS=...,g=0.3 FP_CB_QDIR=queries_g2c03)
VERS = [("g1", "queries_g1"), ("g2", "queries"), ("g2c03", "queries_g2c03")]


def load(folder):
    out = {}
    for s in ("test", "dev"):
        for q in json.loads((ROOT / folder / f"{s}.json").read_text())["queries"]:
            if q["cat"] == "g":
                out[q["qid"]] = q
    return out


def albums_of(q):
    return [set(s) for s in q["answers"][-1]] if q else [set(), set()]


def state(i, A):
    a, b = i in A[0], (len(A) > 1 and i in A[1])
    return "c" if a and b else "a" if a else "b" if b else "0"


def make(n_req):
    from findpics import store
    from findpics.engine import reference_crop
    from findpics.models import ImageTextEncoder
    Q = {v: load(f) for v, f in VERS}
    old_audit = {k["qid"] for k in json.loads((ROOT / "audit" / "key.json").read_text()).values() if k["cat"] == "g"}
    both = sorted(set(Q["g1"]) & set(Q["g2"]) - old_audit)
    rng = random.Random(53)
    users = defaultdict(list)
    for qid in both:
        users[Q["g2"][qid]["user"]].append(qid)
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
    TT = enc.texts([" ".join(Q["g2"][q]["turns"]) for q in pick]).astype(np.float32)
    TT /= np.linalg.norm(TT, axis=1, keepdims=True)
    A4.mkdir(exist_ok=True); (A4 / "sheets").mkdir(exist_ok=True)
    key, libs = {}, {}
    for n, (qid, T) in enumerate(zip(pick, TT)):
        q2 = Q["g2"][qid]; u = q2["user"]
        if u not in libs:
            libs.clear(); libs[u] = Lib(u, it)
        lib = libs[u]
        rows = np.array([row_of[i] for i in lib.ids])
        um = np.isin(unit_item, rows)
        simv = pd.DataFrame({"r": unit_item[um], "s": X[um] @ T}).groupby("r").s.max().reindex(rows).fillna(-1.0).to_numpy()
        alb = {v: albums_of(Q[v].get(qid)) for v, _ in VERS}
        scope = np.zeros(len(lib.ids), bool)
        persons = []
        for v, _ in VERS:
            qv = Q[v].get(qid)
            for a in (qv["program"] if qv else []):
                if isinstance(a, dict):
                    scope |= scope_mask(lib, a, "g") if v == "g1" else (lib.meta_mask(a, loose=True) & _anchor_person(lib, a))
                    if a.get("person") in ("Sam", "Alex") and a["person"] not in persons:
                        persons.append(a["person"])
        strata = defaultdict(list)
        for i in lib.ids:
            pat = "".join(state(i, alb[v]) for v, _ in VERS)
            if pat.strip("0"):
                strata["p" + pat].append(i)
        anyset = set().union(*[x for v in alb for x in alb[v]])
        none = np.where(~np.isin(lib.ids, list(anyset)))[0]
        ins, outs = none[scope[none]], none[~scope[none]]
        ins = ins[np.argsort(-simv[ins])]; outs = outs[np.argsort(-simv[outs])]
        strata["hard_in"] = list(lib.ids[ins[:20]]); strata["rand_in"] = list(lib.ids[ins[20:]])
        strata["hard_out"] = list(lib.ids[outs[:15]]); strata["rand_out"] = list(lib.ids[outs[15:]])
        agree = sorted((s for s in strata if s.startswith("p") and len(set(s[1:])) == 1), key=lambda s: -len(strata[s]))
        take = dict(hard_in=3, rand_in=1, hard_out=2, rand_out=1)
        items = []
        for s, pool in sorted(strata.items()):
            k = take.get(s, 4 if s in agree else 3)
            for i in rng.sample(sorted(pool), min(k, len(pool))):
                items.append((i, s))
        rng.shuffle(items)
        ref_tiles = []
        if persons:
            meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
            G = [g for g in meta.get("face_groups", []) if g["n_items"] >= 25][:2]
            sub = store.subset(idx, rows)
            for nm in persons:
                gi = ["Sam", "Alex"].index(nm)
                if gi >= len(G):
                    continue
                for fr in [G[gi]["rep"]] + G[gi]["face_rows"][:2]:
                    im = reference_crop(sub, int(fr)).convert("RGB")
                    k_ = 300 / max(im.size)
                    im = im.resize((max(1, int(im.width * k_)), max(1, int(im.height * k_))), Image.LANCZOS)
                    c = Image.new("RGB", (TILE, TILE + 46), "white"); c.paste(im, ((TILE - im.width) // 2, 0))
                    d = ImageDraw.Draw(c); d.rectangle([0, 0, 44, 30], fill=(255, 120, 0))
                    d.text((6, 4), "R", fill="black", font=font(20)); d.text((6, TILE + 4), f"reference: {nm}", fill="black", font=font(15))
                    ref_tiles.append(c)
        tiles = ref_tiles + [tile(lib.L.path.iloc[lib.pos[i]], str(j), f"{lib.L.t.iloc[lib.pos[i]]:%Y-%m-%d %a %H:%M} | "
                                  f"{city(lib.L.place.iloc[lib.pos[i]])}") for j, (i, _) in enumerate(items, 1)]
        name = f"g{n:03d}_g"
        for s in range(0, len(tiles), 6):
            sheet(tiles[s:s + 6]).save(A4 / "sheets" / f"{name}_{s // 6}.jpg", quality=92)
        key[name] = dict(qid=qid, user=u, split=q2["split"], turns=q2["turns"], today=q2["today"], n_ref=len(ref_tiles),
                         items=[dict(n=j, item_id=i, stratum=s) for j, (i, s) in enumerate(items, 1)],
                         pops={s: len(p) for s, p in strata.items()},
                         sizes={v: [len(x) for x in alb[v]] for v, _ in VERS}, present={v: qid in Q[v] for v, _ in VERS})
        print(name, qid, len(items), key[name]["pops"], flush=True)
    (A4 / "key.json").write_text(json.dumps(key, indent=1))
    with open(A4 / "requests.txt", "w") as fh:
        for name, k in key.items():
            fh.write(f"{name} [g] (today {k['today']}) " + " || ".join(k["turns"]) + "\n")


def labels(pattern="labels_*.txt"):
    L = {}
    for f in sorted(A4.glob(pattern)):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(None, 3)
            if len(p) >= 3:
                L[(p[0], p[1])] = p[2].strip().lower()
    return L


def estimate(recs, pops, vi, names):
    TP = FP = FN = 0.0
    for name in names:
        for s, pop in pops[name].items():
            R = [lab for (nm, st, lab) in recs if nm == name and st == s]
            if not R or not pop:
                continue
            st_ = s[1 + vi] if s.startswith("p") else "0"
            if st_ == "0":
                FN += pop * sum(l in ("album0", "album1") for l in R) / len(R)
                continue
            ok = {"a": ("album0",), "b": ("album1",), "c": ("album0", "album1")}[st_]
            r = sum(l in ok for l in R) / len(R)
            TP += pop * r; FP += pop * (1 - r)
    return TP, FP, FN


def score():
    key = json.loads((A4 / "key.json").read_text())
    L = labels()
    SPLIT = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--split=")), "all")
    HALF = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--half=")), "all")
    names = sorted(n for n in key if (SPLIT == "all" or key[n]["split"] == SPLIT) and
                   (HALF == "all" or (int(n[1:4]) % 2 == 0) == (HALF == "even")))
    recs, pops, raw = [], {}, defaultdict(Counter)
    unsure = unl = 0
    for name in names:
        k = key[name]; pops[name] = k["pops"]
        for x in k["items"]:
            lab = L.get((name, str(x["n"])))
            if lab is None:
                unl += 1; continue
            if lab == "unsure":
                unsure += 1; continue
            recs.append((name, x["stratum"], lab)); raw[x["stratum"]][lab] += 1
    rng = np.random.default_rng(0)
    out = [f"RESULTS 53: blind eye audit of category g on a fresh sample (audit4/), stratified estimate [split: {SPLIT}; "
           f"half: {HALF}].",
           f"{len(names)} requests ({sum(key[n]['split'] == 'test' for n in names)} test, "
           f"{sum(key[n]['split'] == 'dev' for n in names)} dev); labelled photos {len(recs)} (+{unsure} unsure excluded, "
           f"{unl} unlabelled).", "",
           "version | precision [95% CI] | recall [95% CI] | F1 | est. right / wrong / missed photos"]
    for vi, (v, _) in enumerate(VERS):
        nv = [n for n in names if key[n]["present"][v]]
        TP, FP, FN = estimate(recs, pops, vi, nv)
        P, R = TP / (TP + FP) if TP + FP else 0, TP / (TP + FN) if TP + FN else 0
        bs = []
        for _ in range(2000):
            tp = fp = fn = 0.0
            for x in rng.choice(nv, len(nv)):
                a, b, c = estimate(recs, pops, vi, [x]); tp += a; fp += b; fn += c
            bs.append((tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0))
        bs = np.array(bs)
        pl, ph = np.percentile(bs[:, 0], [2.5, 97.5]); rl, rh = np.percentile(bs[:, 1], [2.5, 97.5])
        out.append(f"{v} ({len(nv)} req) | {P:.2f} [{pl:.2f}, {ph:.2f}] | {R:.2f} [{rl:.2f}, {rh:.2f}] | "
                   f"{2 * P * R / (P + R) if P + R else 0:.2f} | {TP:.0f} / {FP:.0f} / {FN:.0f}")
    out += ["", "raw labels per stratum (pattern = state in g1, g2, g2c03; 0 none, a album0, b album1, c both):"]
    out += [f"  {s}: " + ", ".join(f"{l} {n}" for l, n in sorted(c.items())) for s, c in sorted(raw.items())]
    ov = labels("overlap_main.txt")
    agree = tot = 0; dis = []
    for k_, lab in ov.items():
        other = L.get(k_)
        if other is None or k_[0] not in names or "unsure" in (lab, other):
            continue
        tot += 1; agree += lab == other
        if lab != other:
            dis.append(f"  {k_[0]} {k_[1]}: main={lab} subagent={other}")
    out.append(f"\nmain agent vs subagents (overlap, unsure excluded): {agree}/{tot}")
    out += dis
    tag = "" if SPLIT == "all" and HALF == "all" else f"_{SPLIT}_{HALF}"
    (A4 / f"quality{tag}.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        score()
