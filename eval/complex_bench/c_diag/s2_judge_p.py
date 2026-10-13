# RESULTS 49b step 2: python eval/complex_bench/c_diag/s2_judge_p.py <work dir>  (reads cdiag.json, writes cdiag2.json)
# Recover the 9B judge's P(yes) for each FP/FN/TP photo of the sampled c failures from the shared judge cache
# (key = md5(question + image bytes + size), images rebuilt exactly as engine._judge_rows builds them).
import json, glob, sys, re, hashlib
from pathlib import Path
import numpy as np
from findpics import store
from findpics.engine import _frames_for, person_crop_boxed, _boxed
from findpics.people import expand_refs, item_person_scores
SP = Path(sys.argv[1])
D = json.load(open(SP / "cdiag.json"))
idx = store.load("data/public/index_disbench")
user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
NAMES = ["Sam", "Alex"]
subs = {}
for qid, d in D.items():
    u = qid.split(":")[0]
    if u not in subs:
        subs.clear(); subs[u] = store.subset(idx, np.where(user_of == u)[0])
    sub = subs[u]
    C = {}
    for f in glob.glob(f"eval/complex_bench/runs/cmp/jcache/*/{u}.json"):
        try: C.update(json.loads(open(f).read()))
        except Exception: pass
    a = d["plan"][0]
    q = a["judge_question"]
    row = {iid: i for i, iid in enumerate(sub.items.item_id.astype(str))}
    best = None
    if a.get("person"):
        lib = json.load(open(f"eval/complex_bench/libraries/{u}.json"))
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        g = G[NAMES.index(a["person"])]
        refs = expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3)
        ps, best = item_person_scores(sub, refs)
        q = re.sub(r"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)", "the person in the red box", q)
    hit = 0
    for r in d["rows"]:
        k = row[r["id"]]
        fr = int(best[k]) if best is not None else -1
        try:
            ims = _frames_for(sub, k, fr)
            ims = [person_crop_boxed(sub, im, fr) if (best is not None and fr >= 0) else _boxed(sub, im, fr) for im in ims]
        except Exception as e:
            ims = []
        ps_ = [C.get(hashlib.md5(q.encode() + im.tobytes() + str(im.size).encode()).hexdigest()) for im in ims]
        ps_ = [p for p in ps_ if p is not None]
        r["judge_p"] = round(max(ps_), 3) if ps_ else None
        r["face_row"] = fr
        if best is not None: r["face_sim"] = round(float(ps[k]), 3)
        hit += r["judge_p"] is not None
    d["judge_q"] = q
    print(qid, "| q:", q[:90], "| cache hits", hit, "/", len(d["rows"]), flush=True)
json.dump(D, open(SP / "cdiag2.json", "w"), indent=1, default=str)
