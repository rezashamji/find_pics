"""Fast-stage retrieval evaluation on the public test library (no judge yet).

1. Concepts (Open Images verified labels): for each image encoder, rank all items by cos(image, "a photo of <c>").
   Report recall@|pos|, recall@5|pos| (denominator = verified positives), and the "Apple-style" operating point:
   the score cutoff that gives 90% precision on verified pos/neg, and the recall left at that cutoff.
2. People (IMDB family; references = the simulated Apple-tagged half): how many UNTAGGED photos of each person
   do face vectors recover, at several cosine thresholds, and how many wrong items come along.
Writes eval/results_retrieval.json and prints a table. Usage: python eval/eval_retrieval.py <index_dir> <testlib_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from findpics import store
from findpics.models import ImageTextEncoder
from findpics.people import item_person_scores, refs_from_items
from findpics.store import per_item_max

idx_dir, tl = Path(sys.argv[1]), Path(sys.argv[2])
idx = store.load(idx_dir)
gt = json.load(open(tl / "ground_truth.json"))
row = {iid: i for i, iid in enumerate(idx.items.item_id)}
stats = json.load(open(next((idx_dir / "shards").glob("*/stats.json"))))
names = stats.get("clip_models") or [stats.get("clip_model")]
out = {"n_items": idx.n_items, "n_units": len(idx.units), "n_faces": len(idx.faces), "errors": len(idx.errors),
       "concepts": {}, "people": {}}
print(f"index: {idx.n_items} items, {len(idx.units)} units, {len(idx.faces)} faces, {len(idx.errors)} decode errors")

# ---------- concepts
for k, name in enumerate(names):
    C = idx.clip if k == 0 else np.concatenate([np.load(sd.parent / f"clip_{k}.npy") for sd in sorted((idx_dir / "shards").glob("*/DONE"))])
    enc = ImageTextEncoder(name)
    res = {}
    for key, v in gt.items():
        if not key.startswith("concept:"):
            continue
        c = key.split(":", 1)[1]
        pos = np.array([row[u] for u in v["pos"] if u in row]); neg = np.array([row[u] for u in v["neg"] if u in row])
        if len(pos) == 0:
            continue
        t = enc.texts([f"a photo of {c.lower()}"])
        s = per_item_max((C @ t.T.astype(np.float16)).astype(np.float32)[:, 0], idx.units.item_row.to_numpy(), idx.n_items)
        order = np.argsort(-s)
        rank = np.empty(len(s), int); rank[order] = np.arange(len(s))
        r1 = float((rank[pos] < len(pos)).mean()); r5 = float((rank[pos] < 5 * len(pos)).mean())
        # Apple-style cutoff: highest-recall threshold with >=90% precision among verified pos+neg
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]); sc = np.concatenate([s[pos], s[neg]])
        o = np.argsort(-sc); tp = np.cumsum(lab[o]); prec = tp / np.arange(1, len(o) + 1)
        ok = np.where(prec >= 0.9)[0]
        rec90 = float(tp[ok[-1]] / len(pos)) if len(ok) else 0.0
        res[c] = dict(n_pos=int(len(pos)), n_neg_verified=int(len(neg)), recall_at_npos=r1, recall_at_5npos=r5, recall_at_p90_cutoff=rec90)
    out["concepts"][name] = res
    print(f"\n== {name}")
    print(f"{'concept':16s} {'pos':>5s} {'R@|pos|':>8s} {'R@5|pos|':>9s} {'R@P90cut':>9s}")
    for c, r in res.items():
        print(f"{c:16s} {r['n_pos']:5d} {r['recall_at_npos']:8.2f} {r['recall_at_5npos']:9.2f} {r['recall_at_p90_cutoff']:9.2f}")
    del enc

# ---------- people
tagged = {}
for i, ps in enumerate(idx.items.apple_persons):
    for p in (ps if ps is not None else []):
        tagged.setdefault(p, []).append(i)
print(f"\n== people (references = simulated Apple-tagged half; targets = untagged photos)")
print(f"{'person':16s} {'tagged':>6s} {'untagged':>8s} " + " ".join(f"{'R@'+str(t):>7s} {'FP@'+str(t):>7s}" for t in (0.5, 0.4, 0.3, 0.25, 0.2)))
for key, uu in gt.items():
    if not key.startswith("person:"):
        continue
    p = key.split(":", 1)[1]
    allrows = np.array([row[u] for u in uu if u in row])
    trows = np.array(tagged.get(p, []))
    untag = np.setdiff1d(allrows, trows)
    refs = refs_from_items(idx, trows)
    s, _ = item_person_scores(idx, refs)
    cand_pool = np.setdiff1d(np.arange(idx.n_items), trows)
    rr = {}
    for t in (0.5, 0.4, 0.3, 0.25, 0.2):
        hit = cand_pool[s[cand_pool] >= t]
        tp = np.intersect1d(hit, untag)
        rr[str(t)] = dict(recall=float(len(tp) / max(len(untag), 1)), returned=int(len(hit)), false_pos=int(len(hit) - len(tp)))
    out["people"][p] = dict(tagged=int(len(trows)), untagged=int(len(untag)), n_ref_faces=int(len(refs)), by_threshold=rr,
                            untagged_no_face_detected=int(np.sum(s[untag] < -0.5)))
    print(f"{p:16s} {len(trows):6d} {len(untag):8d} " + " ".join(f"{rr[str(t)]['recall']:7.2f} {rr[str(t)]['false_pos']:7d}" for t in (0.5, 0.4, 0.3, 0.25, 0.2))
          + f"   no-face:{out['people'][p]['untagged_no_face_detected']}")

(Path(__file__).parent / "results_retrieval.json").write_text(json.dumps(out, indent=1))
print("\nwrote eval/results_retrieval.json")
