"""M82 cluster anchor: PE-Core-B-16 vectors as the PHONE index computes them (main env, GPU).
Image: the ~448 px local rendition with its JPEG round trip (= data/public/device_anchor82/r448/<key>.jpg, the r448q80
simulation of eval/selfie_detector/extract_phone_sim.py), then open_clip's PE-Core-B-16 transform (Pillow bilinear squash
to 224, no crop) and the fp16 image tower; unit vectors (findpics.models.ImageTextEncoder). Text: the 3 scorer questions
through the same model's text tower (eval/facts_scorer/train.b16_text).
  python eval/device_anchor82/pe.py <manifest>[,...] <out.npz>   -> keys, img [N,1024] fp16, qs, txt [3,1024] fp32
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/device_anchor82")
from judge import QUESTIONS  # noqa: E402


def main():
    M = pd.concat([pd.read_parquet(m, columns=["key", "r448"]) for m in sys.argv[1].split(",")], ignore_index=True)
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from findpics.models import ImageTextEncoder
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    print("PE-Core-B-16", enc.device, enc.dtype, len(M), flush=True)

    def load(p):
        with Image.open(p) as im:
            return im.convert("RGB")
    pool, V, t0 = ThreadPoolExecutor(16), [], time.time()
    for s in range(0, len(M), 512):
        V.append(enc.images(list(pool.map(load, M.r448.iloc[s:s + 512]))))
        print(s + 512, f"{time.time() - t0:.0f}s", flush=True)
    T = enc.texts(QUESTIONS).astype(np.float32)
    np.savez(sys.argv[2], keys=np.array(M.key.tolist()), img=np.concatenate(V).astype(np.float16), qs=np.array(QUESTIONS),
             txt=T)
    print("PE_DONE", len(M), flush=True)


if __name__ == "__main__":
    main()
