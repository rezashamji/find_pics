"""Reference answers for the app's on-device self-check: the server's PE-Core-B-16 image vector for two synthetic test
images, text vectors for 3 phrases, and the shipped face model's fingerprint (findpics.face_profiles.SHIPPED, AuraFace + flip
since 10-07) for one DigiFace rendered face (a person who does not exist; DigiFace-1M is for non-commercial research:
dev builds only). Face part only: python eval/face_fixtures.py selfcheck. The app recomputes these with its Core ML
models and shows the cosine (should be > 0.99; a flipped or wrongly normalized image shows up as a low number).
Usage (main env): python eval/selfcheck_refs.py"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw

OUT = Path("ios/FindPicsApp.swiftpm/Sources/SelfCheck")
model, _, pre = open_clip.create_model_and_transforms("hf-hub:timm/PE-Core-B-16"); model.eval()
tok = open_clip.get_tokenizer("hf-hub:timm/PE-Core-B-16")
ims = {}
a = Image.new("RGB", (640, 480), (40, 120, 200)); d = ImageDraw.Draw(a)
d.rectangle([60, 300, 600, 470], fill=(230, 210, 120)); d.ellipse([420, 40, 560, 180], fill=(255, 220, 0))
ims["scene.png"] = a                                     # sky / sand / sun: asymmetric, so a flip changes the vector
b = Image.new("RGB", (480, 640), (255, 255, 255)); d = ImageDraw.Draw(b)
for k in range(8):
    d.rectangle([20, 20 + 75 * k, 460, 80 + 75 * k], fill=(200 - 20 * k, 30 * k % 255, 60 + 20 * k))
ims["stripes.png"] = b
refs = {"images": {}, "texts": {}}
with torch.no_grad():
    for n, im in ims.items():
        im.save(OUT / n)
        v = model.encode_image(pre(im).unsqueeze(0)); v = torch.nn.functional.normalize(v, dim=-1)[0]
        refs["images"][n] = [round(float(x), 6) for x in v]
    for t in ["a photo of a beach", "a photo of a dog", "red and blue stripes"]:
        v = torch.nn.functional.normalize(model.encode_text(tok([t])), dim=-1)[0]
        refs["texts"][t] = [round(float(x), 6) for x in v]
from findpics.face_profiles import SHIPPED
from findpics.models import FaceEncoder
z = zipfile.ZipFile("data/public/raw/digiface/subjects_0-1999_72_imgs.zip")
face = Image.open(io.BytesIO(z.read("7/3.png"))).convert("RGB").resize((224, 224))
canvas = Image.new("RGB", (448, 448), (127, 127, 127)); canvas.paste(face, (112, 112)); canvas.save(OUT / "face.png")
f = max(FaceEncoder(SHIPPED).faces(canvas), key=lambda x: x["det_score"])
refs["face"] = dict(file="face.png", model=SHIPPED, box=f["bbox"], embedding=[round(float(x), 6) for x in f["emb"].astype(np.float32)])
json.dump(refs, open(OUT / "refs.json", "w"))
print("SELFCHECK_REFS_OK", list(refs["images"]), len(refs["texts"]))
