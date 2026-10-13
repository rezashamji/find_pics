# RESULTS 49b step 3: python eval/complex_bench/c_diag/s3_sheets.py <work dir> .cache/tmp/c49b_sheets  (native-pixel sheets, not committed)
# Contact sheets (native pixels, <= 500 px DISBench photos) of TP / FP / FN photos for the sampled c failures.
# Private? No: DISBench public Flickr photos. Written under .cache/tmp (inside the project, not committed).
import json, sys, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from findpics import store
from findpics.engine import _boxed, reference_crop
SP = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
D = json.load(open(SP / "cdiag2.json"))
idx = store.load("data/public/index_disbench")
user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
try: font = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception: font = ImageFont.load_default()
NAMES = ["Sam", "Alex"]
sub, cu = None, None
index = []
for qid, d in D.items():
    u = qid.split(":")[0]
    if u != cu:
        sub, cu = store.subset(idx, np.where(user_of == u)[0]), u
    row = {iid: i for i, iid in enumerate(sub.items.item_id.astype(str))}
    rnd = random.Random(qid)
    by = {k: [r for r in d["rows"] if r["kind"] == k] for k in ("TP", "FP", "FN")}
    pick = rnd.sample(by["TP"], min(2, len(by["TP"]))) + rnd.sample(by["FP"], min(7, len(by["FP"]))) + rnd.sample(by["FN"], min(6, len(by["FN"])))
    tiles = []
    a = d["plan"][0]
    if a.get("person"):
        lib = json.load(open(f"eval/complex_bench/libraries/{u}.json"))
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        g = G[NAMES.index(a["person"])]
        rep_row = int(np.where(sub.faces.index == g["rep"])[0][0]) if g["rep"] in sub.faces.index else None
        try:
            ref = reference_crop(idx, int(g["rep"])); ref.thumbnail((300, 300))
            tiles.append((ref, f"REFERENCE {a['person']} (rep face)"))
        except Exception as e:
            print("ref fail", e)
    for r in pick:
        k = row[r["id"]]
        im = Image.open(sub.items.path.iloc[k]).convert("RGB")
        if a.get("person") and r.get("face_row", -1) >= 0:
            im = _boxed(sub, im, int(r["face_row"]))
        lab = (f"{r['kind']} {r['id']} {'IN' if r['in_scope'] else 'OUT'}-scope {r['day'][:10]} judge={r.get('judge_p')}"
               f"\noracle={' '.join(f'{v}' for v in r['oracle'].values())} face={r.get('face_sim')} {im.size[0]}x{im.size[1]}")
        r["sheet_tile"] = len(tiles)
        tiles.append((im, lab))
    sheets = [tiles[i:i + 6] for i in range(0, len(tiles), 6)]
    for si, T in enumerate(sheets):
        W = 3 * 504; rows = (len(T) + 2) // 3
        H = sum(max(t[0].size[1] for t in T[j:j + 3]) + 40 for j in range(0, len(T), 3))
        S = Image.new("RGB", (W, H + 30), "white"); dr = ImageDraw.Draw(S)
        dr.text((4, 4), f"{qid}: {d['words'][:150]}", fill="black", font=font)
        y = 30
        for j in range(0, len(T), 3):
            hmax = 0
            for c, (im, lab) in enumerate(T[j:j + 3]):
                x = c * 504
                dr.text((x + 2, y), lab, fill="blue", font=font)
                S.paste(im, (x, y + 36)); hmax = max(hmax, im.size[1])
            y += hmax + 40
        f = OUT / f"{qid.replace('@', '_').replace(':', '_')}_{si}.jpg"
        S.save(f, quality=93)
        index.append(str(f))
    print(qid, len(tiles), "tiles", len(sheets), "sheets", flush=True)
json.dump(D, open(SP / "cdiag3.json", "w"), indent=1, default=str)
print("\n".join(index))
