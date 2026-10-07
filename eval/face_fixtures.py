"""Golden fixtures for the phone's face path, for the shipped face model (findpics.face_profiles.SHIPPED).

  crops      aligned 112x112 RGB crops of 8 DigiFace rendered faces + the self-check face
             -> data/public/derived/face_check_crops.npz (input of scripts/convert_face_coreml.py's conversion check)
  selfcheck  the server's fingerprint of SelfCheck/face.png with the shipped model -> SelfCheck/refs.json "face"
             (the app's Self-check recomputes it with Core ML + Apple Vision landmarks and shows the cosine)
  golden     FindPicsCore fixtures: Fixtures/faces.json (people.py on synthetic faces, AT THE SHIPPED PROFILE'S CUTS)
             and Fixtures/face_profiles.json (the Python profile table; Swift's FaceProfile table must equal it)
Public data only (DigiFace-1M renders: people who do not exist). Usage (main env): python eval/face_fixtures.py <stage>...
"""
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics")
sys.path.insert(0, str(ROOT / "src"))
from findpics.face_profiles import FACE_PROFILES, SHIPPED  # noqa: E402

SELF = ROOT / "ios/FindPicsApp.swiftpm/Sources/SelfCheck"
FIX = ROOT / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures"


def canvas(face):
    from PIL import Image
    c = Image.new("RGB", (448, 448), (127, 127, 127)); c.paste(face.resize((224, 224)), (112, 112)); return c


def crops():
    from PIL import Image
    from insightface.utils.face_align import norm_crop
    from findpics.models import FaceEncoder
    fe = FaceEncoder(SHIPPED)
    z = zipfile.ZipFile(ROOT / "data/public/raw/digiface/subjects_0-1999_72_imgs.zip")
    ims = [canvas(Image.open(io.BytesIO(z.read(f"{i}/{j}.png"))).convert("RGB")) for i, j in
           [(1, 0), (2, 5), (3, 9), (11, 1), (17, 4), (25, 2), (31, 7), (40, 3)]]
    ims.append(Image.open(SELF / "face.png").convert("RGB"))
    out = []
    for im in ims:
        bgr = np.asarray(im)[:, :, ::-1].copy()
        fs = fe.app.get(bgr)
        if not fs:
            continue
        f = max(fs, key=lambda d: d.det_score)
        out.append(norm_crop(bgr, f.kps, 112)[:, :, ::-1])     # RGB, as the phone feeds it
    np.savez(ROOT / "data/public/derived/face_check_crops.npz", rgb=np.stack(out).astype(np.uint8))
    print("crops", len(out))


def selfcheck():
    from PIL import Image
    from findpics.models import FaceEncoder
    im = Image.open(SELF / "face.png").convert("RGB")
    f = max(FaceEncoder(SHIPPED).faces(im), key=lambda x: x["det_score"])
    refs = json.load(open(SELF / "refs.json"))
    refs["face"] = dict(file="face.png", model=SHIPPED, box=f["bbox"], embedding=[round(float(x), 6) for x in f["emb"].astype(np.float32)])
    json.dump(refs, open(SELF / "refs.json", "w"))
    print("selfcheck face", SHIPPED, "box", [round(v) for v in f["bbox"]])


def golden():
    """people.py on 600 synthetic faces of 40 made-up identities (no real face fingerprints in the repo), with the
    shipped profile's cuts. The spread is chosen so the shipped cuts matter (groups, expansion, other people)."""
    import pandas as pd
    from findpics import people as P
    pr = FACE_PROFILES[SHIPPED]
    rng = np.random.default_rng(7)
    n_id, d = 40, 512
    C = rng.normal(size=(n_id, d)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    C[1] = 0.6 * C[0] + 0.8 * C[1]; C[1] /= np.linalg.norm(C[1])                 # person 1 looks like person 0
    who = np.r_[np.repeat(np.arange(8), 30), rng.integers(8, n_id, 360)]          # 8 frequent people + the rest
    noise = rng.uniform(0.5, 0.95, len(who))[:, None]
    E = C[who] + noise * rng.normal(size=(len(who), d)) / np.sqrt(d)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    E = E.astype(np.float16).astype(np.float32)
    n_items = 400
    item = np.r_[rng.permutation(n_items), rng.integers(0, n_items, len(who) - n_items)]    # every item >= 1 face
    px = rng.uniform(20, 200, len(who)).astype(np.float32); det = rng.uniform(0.5, 1.0, len(who)).astype(np.float32)
    idx = SimpleNamespace(face_emb=E.astype(np.float16), n_items=n_items,
                          faces=pd.DataFrame({"item_row": item, "face_px": px, "det_score": det}))
    G = P.face_groups(idx, top=12, accept=pr["group"])
    refs = E[np.where(who == 0)[0][:5]]
    exp = P.expand_refs(idx, refs, accept=pr["expand"], rounds=3)
    groups = [E[np.array(g["faces"])] for g in G]
    R = exp.astype(np.float32).T
    keep = [g for g in groups if float((g @ R).max(1).mean()) < pr["other"]]
    others = np.concatenate(keep) if keep else np.zeros((0, d), np.float32)
    s, b = P.item_person_scores(idx, refs, others=others)
    s0, _ = P.item_person_scores(idx, refs, others=np.zeros((0, d), np.float32))
    fx = dict(profile=SHIPPED, faces=E.tolist(), faceItem=item.tolist(), nItems=n_items, facePx=px.tolist(),
              det=det.tolist(), groups=[dict(faces=g["faces"], items=g["items"], rep=g["rep"]) for g in G],
              refs=refs.tolist(), expanded_n=int(len(exp)), others=others.astype(np.float32).tolist(),
              others_n=int(len(others)), scores=s.tolist(), best=b.tolist(), scores_plain=s0.tolist())
    json.dump(fx, open(FIX / "faces.json", "w"))
    json.dump(FACE_PROFILES, open(FIX / "face_profiles.json", "w"), indent=1)
    print("golden: groups", len(G), "sizes", [len(g["faces"]) for g in G], "expanded", len(exp), "others", len(others),
          "person items >= accept", int((s >= pr["accept"]).sum()), "plain", int((s0 >= pr["accept"]).sum()),
          "items changed by the other-person rule", int((s != s0).sum()))


if __name__ == "__main__":
    for st in sys.argv[1:]:
        {"crops": crops, "selfcheck": selfcheck, "golden": golden}[st]()
    print("FACE_FIXTURES_OK")
