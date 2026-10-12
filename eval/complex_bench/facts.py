"""Complex-query benchmark, step 1: dense per-photo FACTS from a strong open VLM (Qwen3.5-27B-FP8, vLLM).

Public data only: the 57 DISBench Flickr users' own libraries (109,467 photos, native 500 px).
One JSON object per photo -> eval/complex_bench/facts/<user>.jsonl (restartable: photos already written are skipped).
Usage (vLLM env, GPU with FP8: H100/H200/RTX PRO 6000):
  python eval/complex_bench/facts.py --shard=k/K        (libraries sorted by user id, every K-th from k)
"""
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

MODEL = os.environ.get("FP_FACTS_MODEL", "Qwen/Qwen3.5-27B-FP8")
OUT = Path("eval/complex_bench/facts")

PROMPT = """Describe this photo as structured facts for a photo-search index. Be literal and specific: list what is
actually visible, including small things in the background. Return ONLY one JSON object with these keys:
{"caption": "1-2 specific sentences: what the photo shows",
 "image_type": one of "photo","selfie","group_selfie","screenshot","document","photo_of_screen","photo_of_print_or_photo",
               "artwork_or_drawing","map","poster_or_sign","collage","other",
 "main_subject": "what the photo is mainly of, 1-5 words",
 "setting": one of "indoor","outdoor","inside_vehicle","unclear",
 "scene": ["1-4 scene words, e.g. beach, city street, restaurant, church interior, mountain trail, stadium, airport"],
 "objects": ["every distinct kind of visible object, singular nouns, up to 30, small ones too: cup, bicycle, sign"],
 "animals": ["real live animals visible, e.g. dog, cow, pigeon"],
 "depictions": ["animals/people/things shown only as statues, paintings, drawings, toys, models, photos, e.g. horse statue"],
 "people_count": integer number of real people visible (estimate; 20 for a crowd),
 "people": ["up to 3 most prominent people: age group, sex if clear, clothing colours, pose/expression, what they do"],
 "activities": ["what people/animals are doing, e.g. eating, hiking, dancing, playing guitar"],
 "text": "visible text, transcribed, at most 15 words, or empty",
 "colours": ["2-5 dominant colours, precise names: magenta, teal, navy, beige"],
 "notable_colour": "a striking or unusual colour feature (e.g. magenta stage light), or empty",
 "lighting": one of "daylight","overcast","golden_hour","sunset_or_sunrise","blue_hour","night","indoor_artificial",
             "flash","neon_or_coloured_light","candlelight","mixed",
 "weather": ["visible weather: snow, rain, fog, clear sky, cloudy; empty if indoor/unclear"],
 "quality": ["any of: blurry, dark, overexposed, tilted, noisy, motion_blur; empty if fine"],
 "style": ["any of: black_and_white, sepia, close_up, macro, wide_landscape, aerial, panorama, long_exposure,
           silhouette, reflection, shallow_depth_of_field, from_a_vehicle, night_lights"],
 "event": "the kind of occasion if any (wedding, concert, birthday, sports match, parade, festival, graduation,
           conference, dinner, picnic, hike, sightseeing, museum visit, protest, market) or empty",
 "landmark": "a recognizable named landmark or famous place, or empty"}"""


def data_url(path: str) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    b = io.BytesIO(); im.save(b, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def parse(txt: str):
    s, e = txt.find("{"), txt.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(txt[s:e + 1])
    except json.JSONDecodeError:
        return None


CLAIMS = Path(".cache/cb_claims")


def take(u: str) -> bool:
    """Atomic claim (mkdir) of one library. A claim is taken over only when its heartbeat file is > 40 min old (squeue
    from compute nodes can fail transiently: on 10-12 that made two workers take the same library)."""
    import time as _t
    CLAIMS.mkdir(parents=True, exist_ok=True)
    d = CLAIMS / f"facts_{u}"
    me = os.environ.get("SLURM_JOB_ID", "local")
    try:
        d.mkdir()
        (d / "owner").write_text(me); (d / "beat").touch()
        return True
    except FileExistsError:
        owner = (d / "owner").read_text().strip() if (d / "owner").exists() else ""
        if owner == me:
            return True
        done = (OUT / f"{u}.jsonl").exists() and len(set(l[:40] for l in open(OUT / f"{u}.jsonl"))) >= 0.99 * N_OF.get(u, 10 ** 9)
        last = max([(d / x).stat().st_mtime for x in ("beat", "owner") if (d / x).exists()] + [0])
        if done or _t.time() - last < 40 * 60:
            return False
        import subprocess
        q = subprocess.run(["squeue", "-h", "-j", owner], capture_output=True, text=True) if owner else None
        if q is not None and (q.returncode != 0 and "Invalid job id" not in q.stderr or q.stdout.strip()):
            return False      # owner still queued/running (or squeue unreachable): leave it
        (d / "owner").write_text(me); (d / "beat").touch()
        return True


N_OF: dict = {}


def main():
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    items = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    items["user"] = items.path.str.extract(r"/images/images/([^/]+)/")[0]
    N_OF.update(items.user.value_counts().to_dict())
    users = sorted(items.user.unique())[k::K]
    claim = "--claim" in sys.argv     # work queue: take the next library nobody (alive) has claimed; test/dev first
    if claim:
        sp_ = json.loads(Path("eval/complex_bench/splits.json").read_text())
        users = sp_["test"] + sp_["dev"] + sp_["train"]
    from concurrent.futures import ThreadPoolExecutor
    from vllm import LLM, SamplingParams
    # claim workers go on to phase B (requests + programs + oracle) with the same model load: long context for gen
    llm = LLM(model=MODEL, max_model_len=40960 if claim else 4096, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, enable_prefix_caching=True,
              max_num_seqs=int(os.environ.get("FP_MAX_SEQS", "128")))
    sp = SamplingParams(temperature=0.0, max_tokens=700)
    pool = ThreadPoolExecutor(16)
    OUT.mkdir(parents=True, exist_ok=True)
    for u in users:
        if claim and not take(u):
            continue
        f = OUT / f"{u}.jsonl"
        done = set()
        if f.exists():
            for line in open(f):
                try:
                    r = json.loads(line)
                    if r.get("facts") is not None:
                        done.add(r["item_id"])
                except json.JSONDecodeError:
                    pass
        todo = items[(items.user == u) & ~items.item_id.astype(str).isin(done)]
        if not len(todo):
            print(f"{u}: done already", flush=True); continue
        t0 = time.time()
        for s in range(0, len(todo), 512):
            chunk = todo.iloc[s:s + 512]
            urls = list(pool.map(data_url, chunk.path))
            msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": x}},
                                                   {"type": "text", "text": PROMPT}]}] for x in urls]
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            with open(f, "a") as fh:
                for iid, o in zip(chunk.item_id.astype(str), outs):
                    txt = o.outputs[0].text
                    fh.write(json.dumps({"item_id": iid, "facts": parse(txt), "raw": None if parse(txt) else txt[:2000]}) + "\n")
            print(f"{u}: {s + len(chunk)}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
            if claim:
                (CLAIMS / f"facts_{u}" / "beat").touch()
    if claim:
        phase_b(llm, users, items)


def facts_complete(u) -> bool:
    f = OUT / f"{u}.jsonl"
    if not f.exists():
        return False
    ok = set()
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("facts") is not None:
            ok.add(r["item_id"])
    return len(ok) >= 0.98 * N_OF.get(u, 10 ** 9)


def take_named(key: str) -> bool:
    import time as _t
    CLAIMS.mkdir(parents=True, exist_ok=True)
    d = CLAIMS / key
    me = os.environ.get("SLURM_JOB_ID", "local")
    try:
        d.mkdir(); (d / "owner").write_text(me); (d / "beat").touch()
        return True
    except FileExistsError:
        owner = (d / "owner").read_text().strip() if (d / "owner").exists() else ""
        if owner == me:
            return True
        last = max([(d / x).stat().st_mtime for x in ("beat", "owner") if (d / x).exists()] + [0])
        if _t.time() - last < 10 * 60:
            return False
        import subprocess
        q = subprocess.run(["squeue", "-h", "-j", owner], capture_output=True, text=True) if owner else None
        if q is not None and (q.returncode != 0 and "Invalid job id" not in q.stderr or q.stdout.strip()):
            return False      # owner still queued/running (or squeue unreachable): leave it
        (d / "owner").write_text(me); (d / "beat").touch()
        return True


def phase_b(llm, order, items):
    """After facts: per library (test/dev first) digest -> requests + truth programs (gen.py) -> oracle (oracle.py)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from common import ROOT, load_items
    from findpics import store
    from gen import digest, gen_users
    from oracle import run_oracle
    it = load_items()
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    (ROOT / "digests").mkdir(exist_ok=True)
    while True:
        mine = []
        for u in order:
            if (ROOT / "oracle" / f"{u}.parquet").exists() or not facts_complete(u):
                continue
            if take_named(f"pipe_{u}"):
                mine.append(u)
            if len(mine) >= 4:
                break
        if not mine:
            if all((ROOT / "oracle" / f"{u}.parquet").exists() for u in order):
                print("phase B: nothing left", flush=True); return
            time.sleep(120); continue
        for u in mine:
            if not (ROOT / "digests" / f"{u}.txt").exists():
                d, today = digest(u, it)
                (ROOT / "digests" / f"{u}.txt").write_text(f"TODAY {today}\n{d}")
        t0 = time.time()
        try:
            gen_users(llm, mine)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"phase B gen FAILED {mine}: {e}", flush=True); continue
        print(f"phase B gen {mine} {time.time() - t0:.0f}s", flush=True)
        for u in mine:
            t0 = time.time()
            try:
                run_oracle(llm, [u], idx)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"phase B oracle FAILED {u}: {e}", flush=True); continue
            (CLAIMS / f"pipe_{u}" / "beat").touch()
            print(f"phase B oracle {u} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
