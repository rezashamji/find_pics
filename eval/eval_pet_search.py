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
    import sys
    mixed = "mixed" in sys.argv
    if mixed:   # a realistic library: 19,218 everyday photos (incl. ~1,600 of random dogs) + the query dog's own photos
        from findpics import store
        from findpics.media import load_image
        tl = store.load("data/public/index_testlib")
        first = tl.units.reset_index().groupby("item_row")["index"].first().to_numpy()                  # one vector per item
        BG = tl.clip[first].astype(np.float32); BG /= np.linalg.norm(BG, axis=1, keepdims=True)
        bg_paths = tl.items.path.to_numpy()
    J = VLLMJudge(gpu_mem=0.75)
    rng = np.random.default_rng(0)
    dogs = [d for d in pd.unique(lab) if (lab == d).sum() >= 6]
    out, K, allp = [], 300, []
    for d in rng.choice(dogs, min(40, len(dogs)), replace=False):
        mine = np.where(lab == d)[0]; refs = rng.choice(mine, 3, replace=False); targets = set(mine) - set(refs)
        q = SUBJECT_Q.format(name="this dog", kind="dog")
        if mixed:   # candidates come from [everyday library + this dog's other photos]; index >= 0: dog photo, < 0: library
            own = np.array(sorted(targets)); sv = np.r_[(BG @ V[refs].T).max(1), (V[own] @ V[refs].T).max(1)]
            order = np.argsort(-sv)[:K]
            cand = np.where(order >= len(BG), own[np.clip(order - len(BG), 0, len(own) - 1)], -1 - order)
            s = np.full(len(V), -9.0); s_c = sv[order]
            pic = lambda c: ims[c] if c >= 0 else load_image(bg_paths[-1 - c])
            P3 = np.stack([np.array(J.p_yes([side_by_side(ims[rf], pic(c)) for c in cand], q)) for rf in refs], 1)
            allp.append(pd.DataFrame(dict(dog=int(d), cand=cand, target=[c in targets for c in cand], vec=s_c,
                                          p0=P3[:, 0], p1=P3[:, 1], p2=P3[:, 2])))
            ret = set(cand[P3[:, 0] >= 0.5]); tp = len(ret & targets)
            out.append(dict(dog=int(d), targets=len(targets), in_topK=len(set(cand) & targets), returned=len(ret),
                            correct=tp, precision=tp / max(len(ret), 1), recall=tp / len(targets)))
            print(out[-1], flush=True)
            continue
        s = (V @ V[refs].T).max(1); s[refs] = -9
        cand = np.argsort(-s)[:K]
        P3 = np.stack([np.array(J.p_yes([side_by_side(ims[rf], ims[c]) for c in cand], q)) for rf in refs], 1)  # [K, 3]
        p = P3[:, 0]
        ret = set(cand[p >= 0.5])
        allp.append(pd.DataFrame(dict(dog=int(d), cand=cand, target=[c in targets for c in cand], vec=s[cand],
                                      p0=P3[:, 0], p1=P3[:, 1], p2=P3[:, 2])))
        tp = len(ret & targets)
        out.append(dict(dog=int(d), targets=len(targets), in_topK=len(set(cand) & targets), returned=len(ret), correct=tp,
                        precision=tp / max(len(ret), 1), recall=tp / len(targets)))
        print(out[-1], flush=True)
    r = pd.DataFrame(out)
    summ = dict(dogs=len(r), library=len(ims), candidate_recall=float(r.in_topK.sum() / r.targets.sum()),
                recall=float(r.correct.sum() / r.targets.sum()), precision=float(r.correct.sum() / max(r.returned.sum(), 1)),
                median_returned=float(r.returned.median()))
    print(json.dumps(summ, indent=1))
    tag = "_mixed" if mixed else ""
    r.to_json(f"eval/results_pet_search{tag}.json", orient="records", indent=1)
    A = pd.concat(allp, ignore_index=True); A.to_parquet(f"eval/results_pet_search_cands{tag}.parquet")
    for name, score in (("1 ref", A.p0), ("mean of 3 refs", A[["p0", "p1", "p2"]].mean(1)), ("min of 3 refs", A[["p0", "p1", "p2"]].min(1))):
        for cut in (0.5, 0.7, 0.9):
            j = score >= cut; tp = int((j & A.target).sum())
            print(f"{name:15s} cut {cut}: precision {tp}/{int(j.sum())} = {tp / max(int(j.sum()), 1):.3f}, recall {tp}/{int(A.target.sum())}", flush=True)
    json.dump(summ, open(f"eval/results_pet_search_summary{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
