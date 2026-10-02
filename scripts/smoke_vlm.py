"""vLLM + Qwen3.5 smoke: planner on Reza's real request; P(yes) on 8 bread-positive and 8 verified bread-negative images."""
import json, time
from datetime import date
from findpics.vlm import VLLMJudge
from findpics.planner import plan
from findpics.media import load_image

t = time.time(); J = VLLMJudge(); print(f"load {time.time()-t:.0f}s", flush=True)
REQ = ("Find every photo and video of Reza where he looks heavier or out of shape, and the best photos of him from the "
       "past 6 months where he looks fit and athletic. Put them in two albums: Reza heavier, and Reza fit.")
t = time.time(); P = plan(REQ, J.text, owner="Reza", people=["Reza", "Mom", "Dad"], today=date(2026, 10, 2))
print(f"plan {time.time()-t:.1f}s\n", P.model_dump_json(indent=1), flush=True)
gt = json.load(open("data/public/testlib/ground_truth.json"))["concept:Bread"]
pos = [f"data/public/testlib/library/openimages/{u}.jpg" for u in gt["pos"][:8]]
neg = [f"data/public/testlib/library/openimages/{u}.jpg" for u in gt["neg"][:8]]
ims = [load_image(p) for p in pos + neg]
t = time.time(); ps = J.p_yes(ims, "Is there bread (a loaf, slices, rolls, baguette or buns) visible in this image?")
print(f"judge {len(ims)} imgs in {time.time()-t:.1f}s")
for p, v in zip(pos + neg, ps): print(("POS " if p in pos else "NEG ") + f"{v:.3f} {p.split('/')[-1]}")
t = time.time(); ps = J.p_yes(ims * 16, "Is there bread visible in this image?"); print(f"throughput: {len(ps)/(time.time()-t):.1f} img/s")
print("VLM_SMOKE_OK")
