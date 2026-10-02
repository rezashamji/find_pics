"""Within-person ranking: for 132 CelebA identities that have >=2 'Chubby' and >=2 not-'Chubby' photos, does the judge
rank the same person's chubby-labeled photos above their other photos? This is the transformation-video question
(heavier-era vs leaner-era photos of ONE person), unlike the across-people AUC in eval_attribute.py.
Reports mean per-person AUC and how many people have AUC > 0.5. Usage (vLLM env): python eval/eval_within_person.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from findpics.media import load_image
from findpics.vlm import VLLMJudge


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    lab = pd.read_csv("data/public/celeba_within/labels.csv")
    ims = [load_image(p) for p in lab.path]
    J = VLLMJudge()
    res = {}
    for name, q in [("overweight", "Does this person look overweight?"),
                    ("heavier_vs_usual", "Does this person's face look heavier or fuller than average?")]:
        p = np.array(J.p_yes(ims, q)); lab[f"p_{name}"] = p
        per = []
        for cid, g in lab.groupby("celeb_id"):
            y = g.Chubby.to_numpy().astype(bool)
            per.append(auc(g[f"p_{name}"].to_numpy(), y))
        per = np.array(per)
        res[name] = dict(question=q, people=len(per), mean_within_auc=float(per.mean()),
                         people_auc_gt_05=int((per > 0.5).sum()), people_auc_eq_05=int((per == 0.5).sum()),
                         overall_auc=auc(p, lab.Chubby.to_numpy().astype(bool)))
    print(json.dumps(res, indent=1))
    Path("eval/results_within_person.json").write_text(json.dumps(res, indent=1))
    lab.to_csv("eval/attribute/within_person_scores.csv", index=False)


if __name__ == "__main__":
    main()
