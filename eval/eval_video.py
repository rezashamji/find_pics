"""Video search: does judging only each video's best-matching frame miss videos whose matching content is elsewhere?
Library: data/public/index_pexels (522 Pexels HD videos, 6,036 sampled frames, frames every 2 s capped at 40).
  frames  : decode every indexed frame once (same sampler as the index) -> data/public/pexels_frames/<unit>.jpg
  gen     : 24 videos (seed 0), a RANDOM frame of each (not the middle) -> the VLM describes it -> the LLM writes one search
            query that frame answers -> eval/video/queries.json
  oracle  : per query: product planner (converse) -> judge the planned question on EVERY frame -> eval/video/oracle_<k>.parquet
  analyze : truth = videos with ANY frame judged yes (p >= 0.7). Product = rank frames by the text vector, take each
            video's best frame, judge it (stored answer). Also: best 3 frames per video. Recall of truth videos.
Usage (vLLM env, GPU): python eval/eval_video.py frames|gen|oracle <shard> <n>|analyze
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

IDX = "data/public/index_pexels"
FR = Path("data/public/pexels_frames"); OUT = Path("eval/video"); OUT.mkdir(parents=True, exist_ok=True)


def units():
    from findpics import store
    return store.load(IDX)


def frames():
    from PIL import Image  # noqa: F401
    from findpics.media import sample_video_frames
    idx = units(); FR.mkdir(parents=True, exist_ok=True); U = idx.units.reset_index(drop=True)
    for item_row, g in U.groupby("item_row"):
        if all((FR / f"{u}.jpg").exists() for u in g.index):
            continue
        fr = sample_video_frames(idx.items.path.iloc[int(item_row)])
        for u, t in zip(g.index, g.frame_t):
            im = min(fr, key=lambda x: abs(x[0] - t))[1] if fr else None
            if im is not None:
                im.save(FR / f"{u}.jpg", quality=88)
    print("FRAMES", len(list(FR.glob("*.jpg"))), "of", len(U), flush=True)


def gen():
    from PIL import Image
    from findpics.vlm import VLLMJudge
    idx = units(); U = idx.units.reset_index(drop=True); J = VLLMJudge(gpu_mem=0.8)
    rng = np.random.default_rng(0); qs = []
    for k, item_row in enumerate(rng.choice(idx.n_items, 24, replace=False)):
        us = U.index[U.item_row == item_row].to_numpy(); u = int(rng.choice(us))
        im = Image.open(FR / f"{u}.jpg").convert("RGB")
        cap = J._chat([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": __import__("findpics.vlm", fromlist=["_data_url"])._data_url(im)}},
                       {"type": "text", "text": "Describe what is visible in this video frame in 2 sentences."}]}]],
                      J.SP(temperature=0.0, max_tokens=120))[0].outputs[0].text.strip()
        q = J.text("Write ONE short search query (under 12 words) a person might type into their photo app to find a "
                   f"video showing this scene. Scene: {cap}\nReturn only the query.", max_tokens=40).strip().splitlines()[0]
        qs.append(dict(k=k, item_row=int(item_row), unit=u, caption=cap, query=q.strip('"')))
        print(qs[-1], flush=True)
    json.dump(qs, open(OUT / "queries.json", "w"), indent=1)


def oracle(shard, n):
    from PIL import Image
    from findpics.converse import plan_turn
    from findpics.engine import look_scores
    from findpics.models import ImageTextEncoder
    from findpics.vlm import VLLMJudge
    idx = units(); U = idx.units.reset_index(drop=True)
    enc = ImageTextEncoder(idx.clip_model); J = VLLMJudge(gpu_mem=0.7)
    frs = [Image.open(FR / f"{u}.jpg").convert("RGB") for u in U.index]
    for q in json.load(open(OUT / "queries.json"))[shard::n]:
        f = OUT / f"oracle_{q['k']}.parquet"
        if f.exists():
            continue
        P = plan_turn(q["query"], J.text, today=date(2026, 10, 2)); a = P.albums[0]
        jq = a.judge_question or q["query"]
        look_scores(idx, enc, a.looks or [q["query"]], a.avoid)
        T = enc.texts(a.looks or [q["query"]]).astype(np.float32).mean(0); T /= np.linalg.norm(T)
        unit_look = idx.clip.astype(np.float32) @ T
        p = np.concatenate([J.p_yes(frs[i:i + 96], jq) for i in range(0, len(frs), 96)])
        pd.DataFrame(dict(unit=U.index, item_row=U.item_row, look=unit_look, p=p)).to_parquet(f)
        json.dump(dict(k=q["k"], judge_question=jq, plan=a.model_dump()), open(OUT / f"plan_{q['k']}.json", "w"))
        print(f"q{q['k']} '{q['query']}' -> '{jq}': frames yes {int((p >= .7).sum())}", flush=True)


def analyze():
    Q = {q["k"]: q for q in json.load(open(OUT / "queries.json"))}; res = []
    for f in sorted(OUT.glob("oracle_*.parquet")):
        k = int(f.stem.split("_")[1]); d = pd.read_parquet(f)
        truth = set(d[d.p >= 0.7].item_row)
        best = d.sort_values("look", ascending=False).groupby("item_row").head(1)
        top3 = d.sort_values("look", ascending=False).groupby("item_row").head(3)
        found1 = set(best[best.p >= 0.7].item_row); found3 = set(top3[top3.p >= 0.7].item_row)
        res.append(dict(k=k, query=Q[k]["query"], truth_videos=len(truth),
                        recall_best_frame=len(found1 & truth) / max(len(truth), 1),
                        recall_best3=len(found3 & truth) / max(len(truth), 1),
                        seed_found_best_frame=Q[k]["item_row"] in found1, seed_truth=Q[k]["item_row"] in truth))
    df = pd.DataFrame(res); print(df.round(2).to_string(index=False))
    t = df.truth_videos.sum()
    print("POOLED: truth videos", int(t), "| best frame", round((df.recall_best_frame * df.truth_videos).sum() / t, 3),
          "| best 3 frames", round((df.recall_best3 * df.truth_videos).sum() / t, 3),
          "| seed video judged yes in some frame", int(df.seed_truth.sum()), "of", len(df))
    df.to_json("eval/results_video.json", orient="records", indent=1)


if __name__ == "__main__":
    {"frames": frames, "gen": gen, "analyze": analyze}.get(sys.argv[1], lambda: oracle(int(sys.argv[2]), int(sys.argv[3])))()
