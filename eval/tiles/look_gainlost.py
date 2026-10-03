import numpy as np, pandas as pd, random
from PIL import Image, ImageOps
from findpics import store
from findpics.models import ImageTextEncoder
idx = store.load("data/public/index_testlib")
paths = list(idx.items.path)
TV = np.load("eval/tiles/tile_vectors.npy", mmap_mode="r")
V = np.asarray(TV[:, :5], dtype="float32")
enc = ImageTextEncoder(idx.clip_model, device="cpu")
K = 1600
for c, kind in [("christmas_tree","gained"),("sunglasses","gained"),("guitar","lost")]:
    t = enc.texts([f"a photo of a {c.replace('_',' ')}"]).astype("float32")[0]
    S = V @ t
    whole = set(np.argsort(-S[:,0])[:K]); tiles = set(np.argsort(-S.max(1))[:K])
    p = pd.read_parquet(f"eval/oracle/{c}.parquet")["p"].to_numpy()
    yes = set(np.where(p >= 0.7)[0])
    G = sorted(yes & (tiles - whole)); L = sorted(yes & (whole - tiles))
    print(c, "judge_yes", len(yes), "whole_hits", len(yes & whole), "tile_hits", len(yes & tiles), "GAINED", len(G), "LOST", len(L))
    pool = G if kind == "gained" else L
    pick = random.Random(0).sample(pool, min(4, len(pool)))
    ims = []
    for i in pick:
        im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
        s = 900 / max(im.size); im = im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS)
        ims.append(im)
        print("  pick", i, paths[i], "p=%.3f" % p[i], "whole_rank", int((S[:,0] > S[i,0]).sum()), "tile_rank", int((S.max(1) > S[i].max()).sum()), "argmax_view", int(S[i].argmax()), "size", im.size)
    W = sum(m.width for m in ims[:2]) ; 
    # 2x2 grid, each cell 900x900
    sheet = Image.new("RGB", (1800, 1800), "white")
    for k, m in enumerate(ims):
        sheet.paste(m, ((k%2)*900, (k//2)*900))
    sheet.save(f"eval/tiles/look_{c}_{kind}.jpg", quality=92)
