"""Does a red box around the person's face (inside the person crop) stop the judge from rating a NEIGHBOUR?
Reza's sample (his OK, 10-04): his face-matched photos; 'heavier' asked with the current crop vs crop + red box.
Proxy truth: era (2023 = heavier, 2026 = fit, Reza to confirm). Reports AUC per variant on all photos and on photos
with 2+ faces, and writes a 900 px sheet of the photos where the variants disagree most. Outputs stay in data/private.
Usage (vLLM env, GPU): python eval/eval_crop_box.py
"""
import json
import sys

import numpy as np
import pandas as pd


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main():
    sys.path.insert(0, "src")
    from findpics import store
    from findpics.engine import Thresholds, _judge_rows, person_crop, person_crop_boxed, _frame_for
    from findpics.people import item_person_scores, expand_refs
    from findpics.vlm import VLLMJudge
    idx = store.load("data/private/index_sample")
    named = json.load(open("data/private/index_sample/named_people.json"))
    refs = expand_refs(idx, idx.face_emb[np.array(named["Reza"]["faces"])], accept=0.55, rounds=3)
    score, best = item_person_scores(idx, refs)
    rows = np.where((score >= Thresholds().person_accept) & (idx.items.media == "photo").to_numpy())[0]
    year = pd.to_datetime(idx.items.taken, utc=True, errors="coerce", format="ISO8601").dt.year.to_numpy()
    nfaces = idx.faces.groupby("item_row").size().reindex(range(idx.n_items)).fillna(0).to_numpy()
    J = VLLMJudge(gpu_mem=0.8)
    p_crop = _judge_rows(idx, J, rows, best[rows], "Is the person in this photo heavier than usual?", crop_person=True)
    p_box = _judge_rows(idx, J, rows, best[rows], "Is the person in the red box heavier than usual?", crop_person="box")
    res = {}
    for name, m in [("all", np.ones(len(rows), bool)), ("2+ faces", nfaces[rows] >= 2), ("1 face", nfaces[rows] == 1)]:
        h, l = m & (year[rows] == 2023), m & (year[rows] == 2026)
        res[name] = dict(n_2023=int(h.sum()), n_2026=int(l.sum()), auc_crop=round(auc(p_crop[h], p_crop[l]), 3),
                         auc_box=round(auc(p_box[h], p_box[l]), 3))
    print(json.dumps(res, indent=1))
    json.dump(res, open("data/private/audits/crop_box.json", "w"), indent=1)
    # the 8 multi-face photos where the two variants disagree most, both views side by side
    from PIL import Image, ImageDraw
    multi = np.where(nfaces[rows] >= 2)[0]
    order = multi[np.argsort(-np.abs(p_crop[multi] - p_box[multi]))][:8]
    for s in range(0, len(order), 2):
        sheet = Image.new("RGB", (1800, 1800), "white"); d = ImageDraw.Draw(sheet)
        for j, k in enumerate(order[s:s + 2]):
            im = _frame_for(idx, int(rows[k]), int(best[rows[k]]))
            for c, (f, lab) in enumerate([(person_crop, f"crop p={p_crop[k]:.2f}"), (person_crop_boxed, f"box p={p_box[k]:.2f}")]):
                v = f(idx, im, int(best[rows[k]])).convert("RGB"); v.thumbnail((890, 890))
                sheet.paste(v, (c * 900, j * 900)); d.text((c * 900 + 6, j * 900 + 6), f"{lab} year {year[rows[k]]}", fill="red")
        sheet.save(f"data/private/audits/crop_box_{s // 2}.jpg", quality=85)


if __name__ == "__main__":
    main()
