"""RESULTS 58 step 2b: (question, photo[, face]) TRAINING pairs for the phone judge. Public data only; no eye-labelled
photo, no near-duplicate of one, no complex_bench dev/test library, no section-26 library.

Photo pools
  dis   DISBench, the 16 ev16 libraries only (eyeset.library_tags 'ev16'), photos. Excluded: every item_id in the eye
        set (any library) and every photo whose PE-Core-L vector has cosine >= NEAR (0.92) with an eye-labelled photo
        of the same library.
  oi    data/public/index_testlib photos (15,000 Open Images + ~4,100 IMDB scene photos); minus the 39 photos of the
        10-03 eye check (eval/adjudicate/look/verdicts.csv).
  cm    Wikimedia Commons iPhone photos (data/public/raw/commons_iphone; PE-Core-B-16 vectors of the selfie study).
Per question (look text -> PE-Core text vector; L-14-336 for dis/oi, B-16 for cm):
  whole-photo questions: top 24 dis + 16 oi + 8 cm by cosine, + 8 / 4 / 4 uniform random  (64)
  person ("the person in the red box"): photos with a face >= 24 px in dis/oi; top 28 dis + 12 oi by cosine + 10 / 6
        random (56); the boxed face = the photo's largest face (random other face for 1 in 3 photos).
Usage (GPU or CPU, vLLM env): python eval/judge_teach/pairs.py -> data/public/judge_teach/pairs.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/judge_teach"))
OUT = ROOT / "data/public/judge_teach"
NEAR = 0.92
N_WHOLE = dict(dis=(24, 8), oi=(16, 4), cm=(8, 4))
N_PERSON = dict(dis=(28, 10), oi=(12, 6))
MIN_FACE = 24
# the product's most frequent questions get 6x (whole photo) / 4x (person) the photos (still library-disjoint from eval)
CORE = {"Is a beach clearly visible in this photo?", "Is this a photo of food?", "Is this a photo of a sunset?",
        "Is this a photo of flowers?", "Is this a photo of a cat?", "Is this a photo of a church?",
        "Is this a photo of a beach?", "Is this photo taken at night?", "Is this a selfie?"} | {
    f"Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
    for x in ("dog", "car", "bicycle", "boat")}
CORE_P = {f"Is the person in the red box {x}?" for x in ("smiling", "at the beach", "outdoors", "wearing sunglasses",
                                                          "wearing a hat", "in the snow", "eating")}


def item_vectors(idx):
    """Photo rows and their unit vector (photos have one unit)."""
    u = idx.units
    photo = (idx.items.media == "photo").to_numpy()
    first = u.groupby("item_row").head(1)
    rows = first.item_row.to_numpy()
    V = np.asarray(idx.clip[first.index.to_numpy()], np.float32)
    keep = photo[rows]
    return rows[keep], V[keep]


def main():
    from findpics import store
    from findpics.models import ImageTextEncoder
    from eyeset import library_tags
    rng = np.random.default_rng(58)
    Q = [json.loads(l) for l in open(OUT / "questions.jsonl")]
    E = pd.read_parquet(ROOT / "eval/judge_teach/eyeset.parquet")
    eye_ids = set(E.item_id.astype(str))
    tag, _ = library_tags()
    pools = {}
    # ---- dis
    di = store.load(str(ROOT / "data/public/index_disbench"))
    rows, V = item_vectors(di)
    ids = di.items.item_id.astype(str).to_numpy()
    user = di.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    ok = np.array([tag.get(user[r]) == "ev16" for r in rows])
    rows, V = rows[ok], V[ok]
    lab = np.array([ids[r] in eye_ids for r in rows])
    near = np.zeros(len(rows), bool)
    for u in sorted(set(user[rows])):
        m = user[rows] == u
        L = V[m & lab]
        if len(L):
            near[np.where(m)[0]] = (V[m] @ L.T).max(1) >= NEAR
    print(f"dis: {ok.sum()} ev16 photos, eye-labelled {lab.sum()}, near-duplicates (incl. themselves) {near.sum()}", flush=True)
    keep = ~near & ~lab
    pools["dis"] = dict(idx=di, rows=rows[keep], V=V[keep], ids=ids)
    # ---- oi (testlib)
    ti = store.load(str(ROOT / "data/public/index_testlib"))
    rows, V = item_vectors(ti)
    bad = set(pd.read_csv(ROOT / "eval/adjudicate/look/verdicts.csv").item_row.astype(int))
    keep = np.array([r not in bad for r in rows])
    print(f"oi: {len(rows)} photos, minus {(~keep).sum()} eye-checked", flush=True)
    pools["oi"] = dict(idx=ti, rows=rows[keep], V=V[keep], ids=ti.items.item_id.astype(str).to_numpy())
    # ---- cm
    f = np.load(ROOT / "data/public/raw/commons_iphone/features.npz", allow_pickle=True)
    cm_ids = f["pageid"].astype(str)
    cm_path = np.array([str(ROOT / f"data/public/raw/commons_iphone/img/{p}.jpg") for p in cm_ids])
    ex = np.array([Path(p).exists() for p in cm_path])
    Vc = f["vec"].astype(np.float32)
    Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
    pools["cm"] = dict(rows=np.where(ex)[0], V=Vc[ex], ids=cm_ids, path=cm_path)
    print(f"cm: {ex.sum()} photos", flush=True)
    # ---- faces per photo (dis / oi)
    face_of = {}
    for name in ("dis", "oi"):
        F = pools[name]["idx"].faces
        F = F[F.face_px >= MIN_FACE]
        d = {}
        for fr, ir, px in zip(F.index.to_numpy(), F.item_row.to_numpy(), F.face_px.to_numpy()):
            d.setdefault(int(ir), []).append((float(px), int(fr)))
        face_of[name] = {k: sorted(v, reverse=True) for k, v in d.items()}
    # ---- text vectors
    looks = [q["look"] for q in Q]
    TL = ImageTextEncoder("hf-hub:timm/PE-Core-L-14-336").texts(looks).astype(np.float32)
    TB = ImageTextEncoder("hf-hub:timm/PE-Core-B-16").texts(looks).astype(np.float32)
    out = []
    for qi, q in enumerate(Q):
        person = q["fam"] == "person"
        spec = N_PERSON if person else N_WHOLE
        mult = 6 if q["q"] in CORE else 4 if q["q"] in CORE_P else 1
        for name, (top, rnd) in spec.items():
            top, rnd = top * mult, rnd * mult
            P = pools[name]
            rows, V = P["rows"], P["V"]
            if person:
                m = np.array([int(r) in face_of[name] for r in rows])
                rows, V = rows[m], V[m]
            s = V @ (TB[qi] if name == "cm" else TL[qi])
            order = np.argsort(-s)
            pick = list(order[:top])
            rest = np.setdiff1d(np.arange(len(rows)), pick)
            pick += list(rng.choice(rest, size=min(rnd, len(rest)), replace=False))
            for j, k in enumerate(pick):
                r = int(rows[k])
                fr = -1
                if person:
                    fs = face_of[name][r]
                    fr = fs[0][1] if (len(fs) == 1 or rng.random() > 1 / 3) else fs[int(rng.integers(1, len(fs)))][1]
                path = P["path"][r] if name == "cm" else P["idx"].items.path.iloc[r]
                out.append(dict(qid=q["qid"], q=q["q"], fam=q["fam"], pool=name, item_row=r, face_row=fr,
                                item_id=str(P["ids"][r]), path=path, sel="top" if j < top else "rand", sim=float(s[k])))
        if qi % 500 == 0:
            print(qi, len(out), flush=True)
    D = pd.DataFrame(out)
    D["img_key"] = D.pool + ":" + D.item_row.astype(str) + ":" + D.face_row.astype(str)
    # question-level validation split (5%), for checkpoint choice against the TEACHER only
    D["split"] = np.where(D.qid.map(lambda x: (x * 2654435761) % 100) < 5, "val", "train")
    D = D.drop_duplicates(["qid", "img_key"]).reset_index(drop=True)
    D.to_parquet(OUT / "pairs.parquet")
    print("pairs", len(D), "images", D.img_key.nunique(), D.groupby(["fam", "pool"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
