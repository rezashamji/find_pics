# RESULTS 49b: one person-fact request: judge P per returned photo + sheets of truth photos a 0.7 cut drops.
# python eval/complex_bench/c_diag/s4_fact_lost.py "<qid>" eval/complex_bench/runs/cmp/A4old2_search .cache/tmp/c49b_sheets
# For one person-fact request: judge P per returned photo; sheet of truth photos the 0.7 cut would drop (native px, red box)
import json, glob, sys, re, hashlib, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from findpics import store
from findpics.engine import _frames_for, person_crop_boxed, _boxed
from findpics.people import expand_refs, item_person_scores
qid, run, outd = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); outd.mkdir(parents=True, exist_ok=True)
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {r["qid"]: r for f in sorted(run.glob("part*.json")) for r in json.load(open(f))}
q, r = Q[qid], R[qid]; u = q["user"]
idx = store.load("data/public/index_disbench")
sub = store.subset(idx, np.where(idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy() == u)[0])
C = {}
for f in glob.glob(f"eval/complex_bench/runs/cmp/jcache/*/{u}.json"): C.update(json.load(open(f)))
a = r["turns"][-1]["plan"]["albums"][0]
lib = json.load(open(f"eval/complex_bench/libraries/{u}.json"))
G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
g = G[["Sam", "Alex"].index(a["person"])]
best = item_person_scores(sub, expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3))[1]
qq = re.sub(r"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)", "the person in the red box", a["judge_question"])
row = {iid: i for i, iid in enumerate(sub.items.item_id.astype(str))}
gt = set().union(*map(set, q["answers"][-1])); got = r["turns"][-1]["final"]["albums"][0]
recs = []
for iid in got:
    k = row[iid]; fr = int(best[k])
    ims = [person_crop_boxed(sub, im, fr) for im in _frames_for(sub, k, fr)]
    p = max(C.get(hashlib.md5(qq.encode() + im.tobytes() + str(im.size).encode()).hexdigest(), -1) for im in ims)
    recs.append((iid, iid in gt, p, k, fr))
print(qid, q["turns"], "| q:", qq)
print("returned", len(got), "truth", len(gt), "| TP P>=0.7:", sum(t and p >= .7 for _, t, p, _, _ in recs), "TP P<0.7:", sum(t and p < .7 for _, t, p, _, _ in recs),
      "| FP P>=0.7:", sum((not t) and p >= .7 for _, t, p, _, _ in recs), "FP P<0.7:", sum((not t) and p < .7 for _, t, p, _, _ in recs))
lost = [x for x in recs if x[1] and x[2] < .7]; random.Random(0).shuffle(lost)
keptfp = [x for x in recs if (not x[1]) and x[2] >= .7]
font = ImageFont.load_default()
for name, L in (("lostTP", lost[:9]), ("keptFP", keptfp[:6])):
    for s in range(0, len(L), 6):
        T = L[s:s + 6]
        ims = [_boxed(sub, Image.open(sub.items.path.iloc[k]).convert("RGB"), fr) for _, _, _, k, fr in T]
        H = sum(max(im.size[1] for im in ims[j:j + 3]) + 20 for j in range(0, len(ims), 3))
        S = Image.new("RGB", (3 * 504, H + 20), "white"); d = ImageDraw.Draw(S); d.text((4, 4), f"{qid} {name}: {q['turns'][-1]}", fill="black", font=font)
        y = 20
        for j in range(0, len(ims), 3):
            for c, (im, x) in enumerate(zip(ims[j:j + 3], T[j:j + 3])):
                d.text((c * 504 + 2, y), f"{x[0]} judge={x[2]:.3f} {'TRUTH' if x[1] else 'not-truth'}", fill="blue", font=font); S.paste(im, (c * 504, y + 14))
            y += max(im.size[1] for im in ims[j:j + 3]) + 20
        f = outd / f"{qid.replace('@', '_').replace(':', '_')}_{name}_{s // 6}.jpg"; S.save(f, quality=93); print(f)
