import numpy as np, pandas as pd, random
from PIL import Image, ImageOps
from findpics import store
from findpics.models import ImageTextEncoder
idx = store.load("data/public/index_testlib"); paths = list(idx.items.path)
V = np.asarray(np.load("eval/tiles/tile_vectors.npy", mmap_mode="r")[:, :5], dtype="float32")
enc = ImageTextEncoder(idx.clip_model, device="cpu"); K=1600
c="christmas_tree"; t = enc.texts([f"a photo of a christmas tree"]).astype("float32")[0]; S = V @ t
whole=set(np.argsort(-S[:,0])[:K]); tiles=set(np.argsort(-S.max(1))[:K])
p = pd.read_parquet(f"eval/oracle/{c}.parquet")["p"].to_numpy(); yes=set(np.where(p>=0.7)[0])
G = sorted(yes & (tiles - whole)); L = sorted(yes & (whole - tiles))
pick = random.Random(0).sample(G, 4); rest=[i for i in G if i not in pick]
print("rest", rest, "lost", L)
for k,i in enumerate(rest + L):
    im = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB"); s=900/max(im.size)
    im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS).save(f"eval/tiles/look_christmas_tree_extra{k}.jpg", quality=92)
    print(k, i, paths[i].split('/')[-1], "p=%.3f"%p[i], "in", "G" if i in G else "L")
