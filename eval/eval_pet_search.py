"""End-to-end "find my dog" from 3 reference photos, the product's subject path (converse.SubjectRefs logic):
candidates = max PE-Core image similarity to the refs; judge = [ref | candidate] side by side, strict same-individual
question, yes at p >= 0.5. Library = all DogFaceNet photos (dedup, decodable), ~8-11k photos of >1,000 dogs.
For 40 dogs with >= 6 photos: 3 refs, the dog's other photos are the targets; judge the top K=300 candidates.
Reports per dog: targets, found in top K (candidate recall), returned, correct (precision, recall).
Usage (vLLM env, GPU): python eval/eval_pet_search.py
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def main():
    from findpics.converse import SUBJECT_Q
    from findpics.engine import side_by_side
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    fs = sorted(Path("data/public/raw/dogfacenet/data").glob("*.parquet"))
    df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
    col = [c for c in df.columns if c != "label"][0]
    raw = lambda v: v["bytes"] if isinstance(v, dict) else v
    df["h"] = [hashlib.md5(raw(v)).hexdigest() for v in df[col]]
    multi = df.groupby("h").label.nunique(); df = df[~df.h.isin(set(multi[multi > 1].index))].drop_duplicates("h")
    ims, keep = [], []
    for v in df[col]:
        try:
            ims.append(Image.open(io.BytesIO(raw(v))).convert("RGB")); keep.append(True)
        except Exception:
            keep.append(False)
    df = df[keep].reset_index(drop=True); lab = df.label.to_numpy()
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336")
    V = np.concatenate([enc.images(ims[i:i + 128]) for i in range(0, len(ims), 128)]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    del enc
    import torch; torch.cuda.empty_cache()
    J = VLLMJudge(gpu_mem=0.75)
    rng = np.random.default_rng(0)
    dogs = [d for d in pd.unique(lab) if (lab == d).sum() >= 6]
    out, K = [], 300
    for d in rng.choice(dogs, min(40, len(dogs)), replace=False):
        mine = np.where(lab == d)[0]; refs = rng.choice(mine, 3, replace=False); targets = set(mine) - set(refs)
        s = (V @ V[refs].T).max(1); s[refs] = -9
        cand = np.argsort(-s)[:K]
        q = SUBJECT_Q.format(name="this dog", kind="dog")
        p = np.array(J.p_yes([side_by_side(ims[refs[0]], ims[c]) for c in cand], q))
        ret = set(cand[p >= 0.5])
        tp = len(ret & targets)
        out.append(dict(dog=int(d), targets=len(targets), in_topK=len(set(cand) & targets), returned=len(ret), correct=tp,
                        precision=tp / max(len(ret), 1), recall=tp / len(targets)))
        print(out[-1], flush=True)
    r = pd.DataFrame(out)
    summ = dict(dogs=len(r), library=len(ims), candidate_recall=float(r.in_topK.sum() / r.targets.sum()),
                recall=float(r.correct.sum() / r.targets.sum()), precision=float(r.correct.sum() / max(r.returned.sum(), 1)),
                median_returned=float(r.returned.median()))
    print(json.dumps(summ, indent=1))
    r.to_json("eval/results_pet_search.json", orient="records", indent=1)
    json.dump(summ, open("eval/results_pet_search_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
