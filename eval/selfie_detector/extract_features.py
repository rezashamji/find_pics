"""M46 step 2: PE-Core-B-16 image vector (the phone's model, same open_clip squash-224 preprocessing) + SCRFD face
boxes (the cluster face pipeline, findpics.models.FaceEncoder) for every downloaded thumbnail. GPU job.

Usage: python eval/selfie_detector/extract_features.py LABELS.jsonl IMG_DIR OUT.npz
OUT.npz: pageid[N], vec[N,1024] f16, W[N], H[N], faces_json (list per photo of [x1,y1,x2,y2,score]).
"""
from __future__ import annotations

import faulthandler
faulthandler.dump_traceback_later(300, repeat=True)   # a hung import / model load shows up in the log
print("start", flush=True)
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from findpics.media import load_image
from findpics.models import FaceEncoder, ImageTextEncoder


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    img_dir, out = sys.argv[2], sys.argv[3]
    rows = [r for r in rows if os.path.exists(os.path.join(img_dir, f"{r['pageid']}.jpg"))]
    print(len(rows), "images", flush=True)
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    print("PE-Core loaded", enc.device, flush=True)
    face = FaceEncoder("auraface_flip")
    print("face loaded", flush=True)

    def load(r):
        try:
            return r["pageid"], load_image(os.path.join(img_dir, f"{r['pageid']}.jpg"))
        except Exception as e:
            print("bad", r["pageid"], e, flush=True)
            return r["pageid"], None

    ids, vecs, W, H, faces = [], [], [], [], []
    t0 = time.time()
    B = 64
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(rows), B):
            batch = [x for x in ex.map(load, rows[s:s + B]) if x[1] is not None]
            if not batch:
                continue
            v = enc.images([im for _, im in batch])
            for (pid, im), vv in zip(batch, v):
                fs = face.faces(im)
                ids.append(pid); vecs.append(vv); W.append(im.size[0]); H.append(im.size[1])
                faces.append([[*f["bbox"], f["det_score"]] for f in fs])
            if s % (B * 20) == 0:
                print(f"{time.time() - t0:6.0f}s {s}/{len(rows)}", flush=True)
    np.savez(out, pageid=np.array(ids), vec=np.stack(vecs).astype(np.float16), W=np.array(W), H=np.array(H),
             faces_json=json.dumps(faces))
    print("saved", out, len(ids), flush=True)


if __name__ == "__main__":
    main()
