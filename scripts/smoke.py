"""GPU smoke test: CUDA, SigLIP2 forward, InsightFace on a real image, ORT providers."""
import time, glob
import torch, onnxruntime
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("ort", onnxruntime.get_available_providers())
from findpics.models import ImageTextEncoder, FaceEncoder
from findpics.media import load_image
ims = [load_image(p) for p in sorted(glob.glob("data/public/raw/openimages/images/*.jpg"))[:64]]
enc = ImageTextEncoder()
t = time.time(); v = enc.images(ims); torch.cuda.synchronize(); print("clip", v.shape, f"{len(ims)/(time.time()-t):.1f} img/s (incl. warmup)")
t = time.time(); v = enc.images(ims); torch.cuda.synchronize(); print("clip warm", f"{len(ims)/(time.time()-t):.1f} img/s")
tv = enc.texts(["a photo of bread", "a dog"]); print("text", tv.shape, "bread-sims top", (v.astype("float32") @ tv.T)[:, 0].max())
fe = FaceEncoder()
t = time.time(); n = sum(len(fe.faces(im)) for im in ims); print("faces", n, f"{len(ims)/(time.time()-t):.1f} img/s")
print("SMOKE_OK")
