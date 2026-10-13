# RESULTS 49b counterfactual from cached judge answers (no new judge calls): object/scene albums (no person, no anchor)
# keep only returned photos whose main question got P(yes) >= CUT (engine default judge_accept 0.7).
# Exact for raising the cut (it only removes photos); the stop rule could have judged a different tail at another cut
# (not modelled). Usage (repo root, PYTHONPATH=src): python eval/complex_bench/sim_judgecut.py <run_search_dir> <out_dir> <cut>
import json, glob, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "eval/complex_bench")
from score import score_turn
from findpics import store
from findpics.engine import _frames_for, _boxed
run, out, CUT = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {r["qid"]: r for f in sorted(run.glob("part*.json")) for r in json.load(open(f))}
S = {r["qid"]: r for r in json.load(open(run / "scored.json"))}
idx = store.load("data/public/index_disbench")
user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
cur, sub, C, row = None, None, {}, {}
rows_out, n_changed, miss, n_rows = [], 0, 0, 0
for qid in sorted(R, key=lambda q: R[q]["user"]):
    r, q = R[qid], Q[qid]
    rec = dict(S[qid])
    if r.get("error") or not r["turns"] or r["turns"][-1].get("final") is None:
        rows_out.append(rec); continue
    t = r["turns"][-1]; albums = [list(a) for a in t["final"]["albums"]]; touched = False
    for ai, a in enumerate(t["plan"]["albums"]):
        if a.get("person") or a.get("anchor") or not a.get("judge_question") or ai >= len(albums):
            continue
        if r["user"] != cur:
            cur = r["user"]; sub = store.subset(idx, np.where(user_of == cur)[0]); C = {}
            for f in glob.glob(f"eval/complex_bench/runs/cmp/jcache/*/{cur}.json"):
                try: C.update(json.loads(open(f).read()))
                except Exception: pass
            row = {iid: i for i, iid in enumerate(sub.items.item_id.astype(str))}
        qq = a["judge_question"]; keep = []
        for iid in albums[ai]:
            k = row[iid]; n_rows += 1
            ps = [C.get(hashlib.md5(qq.encode() + im.tobytes() + str(im.size).encode()).hexdigest()) for im in _frames_for(sub, k, -1)]
            ps = [p for p in ps if p is not None]
            if not ps: miss += 1; keep.append(iid); continue
            if max(ps) >= CUT: keep.append(iid)
        if len(keep) != len(albums[ai]): touched = True
        albums[ai] = keep
    if touched:
        n_changed += 1; rec = dict(rec); rec["final"] = score_turn(q["cat"], albums, q["answers"][-1])
    rows_out.append(rec)
json.dump(rows_out, open(out / "scored.json", "w"), indent=1)
print(f"{run.name} cut {CUT}: requests changed {n_changed}, returned rows checked {n_rows}, without cached answer (kept) {miss}")
