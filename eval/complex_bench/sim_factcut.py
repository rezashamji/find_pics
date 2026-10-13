# Counterfactual from cached judge answers (no new judge calls): person albums whose question is a FACT (no LOOK_WORDS)
# keep only photos the judge said yes to (P >= judge_accept 0.7), instead of "top half of the person's photos OR P >= 0.7".
# Returned set new = old returned set of that album INTERSECT {P >= 0.7} (exact: the rule only removes photos).
# Usage: python simfact.py <run_search_dir> <out_dir>   -> <out_dir>/scored.json (score.py format) + counts
import json, glob, sys, re, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "eval/complex_bench")
from score import score_turn
from findpics import store
from findpics.engine import _frames_for, person_crop_boxed, LOOK_WORDS, Thresholds
from findpics.people import expand_refs, item_person_scores
run, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ACC = Thresholds().judge_accept
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {r["qid"]: r for f in sorted(run.glob("part*.json")) for r in json.load(open(f))}
S = {r["qid"]: r for r in json.load(open(run / "scored.json"))}
idx = store.load("data/public/index_disbench")
user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
NAMES = ["Sam", "Alex"]
cur, sub, C, prof = None, None, {}, {}
rows_out, changed, n_alb, miss = [], [], 0, 0
for qid in sorted(R, key=lambda q: R[q]["user"]):
    r, q = R[qid], Q[qid]
    rec = dict(S[qid])
    if r.get("error") or not r["turns"] or r["turns"][-1].get("final") is None:
        rows_out.append(rec); continue
    t = r["turns"][-1]
    albums = [list(a) for a in t["final"]["albums"]]
    plan = t["plan"]["albums"]
    touched = False
    for ai, a in enumerate(plan):
        if not (a.get("person") in NAMES and a.get("judge_question") and not LOOK_WORDS.search(a["judge_question"]) and not a.get("anchor")):
            continue
        u = r["user"]
        if u != cur:
            cur = u; sub = store.subset(idx, np.where(user_of == u)[0]); C = {}; prof = {}
            for f in glob.glob(f"eval/complex_bench/runs/cmp/jcache/*/{u}.json"):
                try: C.update(json.loads(open(f).read()))
                except Exception: pass
        lib = json.load(open(f"eval/complex_bench/libraries/{u}.json"))
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        if NAMES.index(a["person"]) >= len(G): continue
        if a["person"] not in prof:
            g = G[NAMES.index(a["person"])]
            refs = expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3)
            prof[a["person"]] = item_person_scores(sub, refs)[1]
        best = prof[a["person"]]
        qq = re.sub(r"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)", "the person in the red box", a["judge_question"])
        row = {iid: i for i, iid in enumerate(sub.items.item_id.astype(str))}
        if ai >= len(albums): continue
        keep = []
        for iid in albums[ai]:
            k = row[iid]; fr = int(best[k])
            ims = [person_crop_boxed(sub, im, fr) if fr >= 0 else im for im in _frames_for(sub, k, fr)]
            ps = [C.get(hashlib.md5(qq.encode() + im.tobytes() + str(im.size).encode()).hexdigest()) for im in ims]
            ps = [p for p in ps if p is not None]
            if not ps: miss += 1; keep.append(iid); continue      # no cached answer: keep (conservative)
            if max(ps) >= ACC: keep.append(iid)
        n_alb += 1
        if len(keep) != len(albums[ai]): touched = True
        albums[ai] = keep
    if touched:
        rec = dict(rec); rec["final"] = score_turn(q["cat"], albums, q["answers"][-1])
        changed.append((qid, q["cat"], round(S[qid]["final"]["f1"], 3), round(rec["final"]["f1"], 3), " || ".join(q["turns"])[:80]))
    rows_out.append(rec)
json.dump(rows_out, open(out / "scored.json", "w"), indent=1)
print(f"{run.name}: person fact albums {n_alb}, requests changed {len(changed)}, rows without cached answer kept {miss}")
for c in sorted(changed, key=lambda c: c[3] - c[2]): print("  ", c)
