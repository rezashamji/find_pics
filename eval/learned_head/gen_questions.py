"""M48-scale step 2a: ~10^4 diverse planner-style judge questions, written by an open VLM (Qwen3.5-9B, 16-bit) from
public photos of the 37 TRAINING DISBench libraries (the 20 test libraries of EV16 / audit4 are not in the pack).
The model first captions the photo, then writes 6 yes/no judge questions of fixed kinds (object, scene, attribute,
person, fact, text) in the planner's own question forms, each one general enough to be a library search.
The prompt names the held-out everyday concepts AND their semantic neighbourhoods (vehicles, pets, food and drink,
shore and sea, places of worship, night/dusk...) so few questions are wasted; filter.py enforces it afterwards.

  python eval/learned_head/gen_questions.py --shard=k/K [--n=8000] -> eval/learned_head/scale/qgen/part{k}.jsonl
Public data only.
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "eval/learned_head")
OUT = Path("eval/learned_head/scale/qgen")
MODEL = "Qwen/Qwen3.5-9B"

PROMPT = """You are helping build a search engine for personal photo libraries. A search is answered by asking a vision model one yes/no question about every photo.

Look at this photo.
1. Write a one-sentence caption of what it shows.
2. Write 6 yes/no questions for which THIS photo's answer is YES, one of each kind below. Each question must be a reusable search: general enough that many other photos could also match (name kinds of things, not this exact scene; no names of people or places, no dates, no "this specific").
   - object: a concrete thing visible in the photo (prefer a secondary or small thing, not always the main subject). Form: "Is there a real <thing> anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?" or "Is there a <thing> visible in this photo?"
   - scene: the kind of place, event or main subject. Form: "Is this a photo of <...>?" or "Is this photo taken at/in <...>?"
   - attribute: a property of something in the photo (color, material, pattern, shape, size, condition). Example: "Is there a red umbrella in this photo?", "Is the building made of brick?"
   - person: how a person looks or what they wear or do (age group, clothing, hair, expression, pose, activity). If nobody is in the photo, ask about an animal or a second object instead.
   - fact: a relation, count, action, viewpoint, weather, season, indoor/outdoor or photo style. Example: "Are there exactly two people in this photo?", "Is someone holding an umbrella?", "Is this an aerial view?", "Is it snowing in this photo?"
   - text: visible text, a sign, a document, a screen, a label or a logo. If there is none, write another fact question.
Do NOT write any question about these topics (they are reserved for a test): dogs, cats or any pets; cars, trucks, buses, vans, motorcycles, bicycles, scooters or any road vehicle, traffic or parking; boats, ships, sailing, harbors or docks; beaches, sand, the sea, the ocean, waves or the shore; food, meals, dishes, snacks, desserts, fruit, drinks, eating, drinking, cooking or restaurants; flowers, bouquets, blossoms or gardens; sunsets, sunrises, dusk, dawn or golden hour; churches, chapels, cathedrals, temples, mosques or any place of worship; selfies; night, darkness, evening, city lights or fireworks.

{extra}Answer with JSON only: {"caption": "...", "questions": [{"kind": "object", "q": "..."}, {"kind": "scene", "q": "..."}, {"kind": "attribute", "q": "..."}, {"kind": "person", "q": "..."}, {"kind": "fact", "q": "..."}, {"kind": "text", "q": "..."}]}"""


EXTRA_V2 = ("Avoid generic questions that would match most photos (outdoors, indoors, daytime, sky, clouds, trees, grass, "
            "a building, a wall, a window, a person visible, any text visible). Prefer specific, distinctive things a "
            "person would actually search their library for, e.g. a hot air balloon, a stone arch bridge, a snow-covered "
            "peak, a person in a graduation gown, a child on a swing, a whiteboard with writing, a red phone box, "
            "a mural, a stack of books, a carousel.\n")


def main():
    import numpy as np
    from pack import load, image, PACK
    from findpics.media import load_image
    from findpics.vlm import _data_url
    import io
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    n_total = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--n=")), "8000"))
    k, K = map(int, shard.split("/"))
    v2 = "--v2" in sys.argv
    tag = "v2_" if v2 else ""
    prompt = PROMPT.replace("{extra}", EXTRA_V2 if v2 else "")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"part{tag}{k}.jsonl").exists():
        print("exists", flush=True); return
    # seed photos: n_total spread evenly over the 16 packs; this shard takes every K-th
    rng = random.Random(1 if v2 else 0)
    per = n_total // 16
    seeds = []
    for p in range(16):
        ids = (PACK / f"pack{p}_ids.txt").read_text().split()
        seeds += [(p, i) for i in sorted(rng.sample(range(len(ids)), per))]
    seeds = seeds[k::K]
    t0 = time.time()
    urls, sid = [], []
    for p in sorted({p for p, _ in seeds}):
        ids, off, buf = load(p)
        for pp, i in seeds:
            if pp == p:
                im = load_image(io.BytesIO(buf[off[i]:off[i + 1]]))
                urls.append(_data_url(im)); sid.append(ids[i])
    print(f"{len(urls)} seed photos loaded {time.time() - t0:.0f}s", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True)
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)
    msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                          {"type": "text", "text": prompt}]}] for u in urls]
    sp = SamplingParams(temperature=0.8 if v2 else 0.7, top_p=0.9, max_tokens=500, seed=k + 100 * v2)
    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    tmp = OUT / f".part{tag}{k}.tmp"
    bad = 0
    with open(tmp, "w") as fh:
        for u, o in zip(sid, outs):
            t = o.outputs[0].text
            m = re.search(r"\{.*\}", t, re.S)
            try:
                P = json.loads(m.group(0))
                qs = [(str(x["kind"]), str(x["q"]).strip()) for x in P["questions"]]
            except Exception:
                bad += 1; continue
            fh.write(json.dumps(dict(item_id=u, caption=P.get("caption", ""), questions=qs)) + "\n")
    os.replace(tmp, OUT / f"part{tag}{k}.jsonl")
    print(f"QGEN_DONE {len(outs)} photos, unparsable {bad}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
