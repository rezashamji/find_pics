"""How well do the fast encoder and the VLM judge agree with human 'Chubby' labels (CelebA, 600 pos / 1400 neg)?

Reports AUC (probability a random positive scores above a random negative) for:
  - PE-Core text-image score: cos("a photo of an overweight person with a heavy, round face") - cos("a photo of a slim, fit person")
  - VLM P(yes) on "Does this person look overweight?"
plus judge accuracy at P(yes) >= 0.5 and contact sheets of the most confident disagreements, so they can be looked at.
Usage (vLLM env): python eval/eval_attribute.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from findpics.contact import sheet
from findpics.media import load_image
from findpics.models import ImageTextEncoder
from findpics.vlm import VLLMJudge


def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    D = Path("data/public/celeba_eval"); OUT = Path("eval/attribute"); OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(D / "labels.csv")
    y = lab.Chubby.to_numpy().astype(bool)
    ims = [load_image(p) for p in lab.path]

    enc = ImageTextEncoder()
    V = np.concatenate([enc.images(ims[i:i + 128]) for i in range(0, len(ims), 128)]).astype(np.float32)
    T = enc.texts(["a photo of an overweight person with a heavy, round face", "a photo of a slim, fit person"])
    clip_score = V @ T[0] - V @ T[1]
    del enc
    import torch; torch.cuda.empty_cache()

    J = VLLMJudge(gpu_mem=0.80)
    Q = "Does this person look overweight?"
    pj = np.array(J.p_yes(ims, Q))
    res = dict(n=len(y), n_pos=int(y.sum()), clip_auc=auc(clip_score, y), judge_auc=auc(pj, y),
               judge_acc_at_05=float(((pj >= 0.5) == y).mean()), judge_yes_rate=float((pj >= 0.5).mean()),
               judge_tpr=float((pj[y] >= 0.5).mean()), judge_fpr=float((pj[~y] >= 0.5).mean()),
               by_sex={s: dict(judge_auc=auc(pj[m], y[m]), n=int(m.sum()), n_pos=int(y[m].sum()))
                       for s, m in (("male", lab.Male.to_numpy().astype(bool)), ("female", ~lab.Male.to_numpy().astype(bool)))})
    print(json.dumps(res, indent=1))
    (Path("eval") / "results_attribute.json").write_text(json.dumps(res, indent=1))
    lab["p_judge"] = pj; lab["clip"] = clip_score
    fp = lab[~lab.Chubby].sort_values("p_judge", ascending=False).head(24)
    fn = lab[lab.Chubby].sort_values("p_judge").head(24)
    sheet(list(fp.path), OUT / "judge_yes_label_no.jpg", labels=[f"p={p:.2f}" for p in fp.p_judge], title="Judge says overweight, CelebA says not Chubby (top 24)")
    sheet(list(fn.path), OUT / "judge_no_label_yes.jpg", labels=[f"p={p:.2f}" for p in fn.p_judge], title="CelebA says Chubby, judge says no (top 24)")
    lab.to_csv(OUT / "scores.csv", index=False)



if __name__ == "__main__":
    main()
