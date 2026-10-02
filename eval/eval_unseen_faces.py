"""Face matching on people NO model can have trained on: DigiFace-1M synthetic identities (rendered, varied pose /
lighting / expression / accessories). Answers Reza's question: does person-finding depend on the face model already
knowing the person (e.g. a celebrity)?
Protocol (same metric as the IMDB test): 300 identities x 72 images. Per identity: 8 random images = references (like
Apple tags), the other 64 = targets; every other identity's images = distractors. Same reference purification as the
product (refs_from_items consensus) is not needed here (each image has one face). Report recall and wrong matches at
cosine >= 0.40 / 0.30. Usage (main env, GPU): python eval/eval_unseen_faces.py"""
import io, json, zipfile
import numpy as np
from PIL import Image
from findpics.models import FaceEncoder

Z = "data/public/raw/digiface/subjects_0-1999_72_imgs.zip"
N_ID, N_REF = 300, 8
rng = np.random.default_rng(0)
fe = FaceEncoder()
z = zipfile.ZipFile(Z)
emb, who, miss = [], [], 0
for i in range(N_ID):
    for k in range(72):
        im = Image.open(io.BytesIO(z.read(f"{i}/{k}.png"))).convert("RGB")
        canvas = Image.new("RGB", (448, 448), (127, 127, 127)); im = im.resize((224, 224))
        canvas.paste(im, (112, 112))                      # tight crop -> put it in a scene-sized canvas for the detector
        f = fe.faces(canvas)
        if not f:
            miss += 1; continue
        best = max(f, key=lambda d: d["det_score"])
        emb.append(best["emb"].astype(np.float32)); who.append(i)
E = np.stack(emb); who = np.array(who)
print(f"faces detected: {len(E)} of {N_ID*72} images ({miss} not detected)")
res = {t: dict(tp=0, targets=0, fp=0) for t in (0.4, 0.3)}
per_id = []
for i in range(N_ID):
    mine = np.where(who == i)[0]
    if len(mine) < N_REF + 5:
        continue
    ref = rng.choice(mine, N_REF, replace=False); tgt = np.setdiff1d(mine, ref); others = np.where(who != i)[0]
    s = (E @ E[ref].T).max(1)
    for t in res:
        res[t]["tp"] += int((s[tgt] >= t).sum()); res[t]["targets"] += len(tgt); res[t]["fp"] += int((s[others] >= t).sum())
    per_id.append(float((s[tgt] >= 0.4).mean()))
out = {str(t): dict(recall=r["tp"] / r["targets"], tp=r["tp"], targets=r["targets"], wrong_matches=r["fp"],
                    wrong_per_person=r["fp"] / N_ID, distractor_images_per_person=int(len(E) * (N_ID - 1) / N_ID))
       for t, r in res.items()}
out["per_person_recall_at_0.4"] = dict(min=float(np.min(per_id)), p10=float(np.percentile(per_id, 10)), median=float(np.median(per_id)))
out["detected"] = int(len(E)); out["not_detected"] = miss
print(json.dumps(out, indent=1)); json.dump(out, open("eval/results_unseen_faces.json", "w"), indent=1)
# contact sheet input for the raw look: the 3 worst identities' refs and missed targets
worst = np.argsort(per_id)[:3]
json.dump([int(w) for w in worst], open("eval/unseen_worst_ids.json", "w"))
