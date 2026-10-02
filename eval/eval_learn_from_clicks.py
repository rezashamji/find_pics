"""Learning from the user's clicks (no model weights change): text ranking -> user marks the top R results right/wrong
(simulated with Open Images verified labels; unlabeled items are skipped, as a user could skip unsure ones) -> logistic
regression on the stored image vectors (+ text score as one feature) -> re-rank. Metric: recall of the NOT-yet-reviewed
verified positives within the top 2x, text vs text+clicks. Usage: python eval/eval_learn_from_clicks.py (CPU)."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from findpics import store
from findpics.models import ImageTextEncoder
from findpics.store import per_item_max

idx = store.load("data/public/index_testlib")
gt = json.load(open("data/public/testlib/ground_truth.json"))
row = {u: i for i, u in enumerate(idx.items.item_id)}
enc = ImageTextEncoder(idx.clip_model, device="cpu")
C = idx.clip.astype(np.float32); ur = idx.units.item_row.to_numpy()
X = np.zeros((idx.n_items, C.shape[1]), np.float32); np.add.at(X, ur, C)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
out = {}
for R in (30, 60):
    print(f"\n== user reviews top {R}")
    print(f"{'concept':16s} {'labels(+/-)':>11s}  text  +clicks")
    rows = []
    for key, v in gt.items():
        if not key.startswith("concept:"):
            continue
        c = key.split(":", 1)[1]
        pos = set(row[u] for u in v["pos"] if u in row); neg = set(row[u] for u in v["neg"] if u in row)
        if len(pos) < 15:
            continue
        t = per_item_max((C @ enc.texts([f"a photo of {c.lower()}"]).T)[:, 0], ur, idx.n_items)
        top = np.argsort(-t)[:R]
        lab = [(i, 1) for i in top if i in pos] + [(i, 0) for i in top if i in neg]
        # a user marks wrong photos too; if top-R had no verified negatives, add the R lowest-ranked of the top-R as "skip"
        y = np.array([l for _, l in lab]); ii = np.array([i for i, _ in lab])
        reviewed = set(top.tolist()); remaining = np.array(sorted(pos - reviewed)); K = 2 * len(remaining)
        mask = np.ones(idx.n_items, bool); mask[list(reviewed)] = False
        def rec(s):
            s = np.where(mask, s, -np.inf); return float(np.isin(remaining, np.argsort(-s)[:K]).mean())
        r_text = rec(t)
        if y.sum() >= 2 and (y == 0).sum() >= 2:
            feats = np.c_[X, (t - t.mean()) / t.std()]
            # unlabeled bottom-of-library items as extra negatives (a random 500: the user never sees them; mostly negative)
            rnd = np.random.default_rng(0).choice(np.argsort(t)[: idx.n_items // 2], 500, replace=False)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
                np.r_[feats[ii], feats[rnd]], np.r_[y, np.zeros(500)])
            r_click = rec(clf.decision_function(feats))
        else:
            r_click = float("nan")
        rows.append((c, r_text, r_click))
        out.setdefault(str(R), {})[c] = dict(text=r_text, clicks=r_click, labeled_pos=int(y.sum()), labeled_neg=int((y == 0).sum()), remaining=int(len(remaining)))
        print(f"{c:16s} {int(y.sum()):5d}/{int((y==0).sum()):<5d}  {r_text:.2f}  {r_click:.2f}")
    ok = [r for r in rows if r[2] == r[2]]
    print("mean over concepts with both labels:", round(np.mean([r[1] for r in ok]), 3), "->", round(np.mean([r[2] for r in ok]), 3), f"(n={len(ok)})")
json.dump(out, open("eval/results_learn_from_clicks.json", "w"), indent=1)
