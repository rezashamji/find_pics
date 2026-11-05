"""M82 cluster anchor: judge P(yes) for the 3 #factsCompare questions on every photo of a manifest shard (vLLM env).
Product judge pipeline = findpics.vlm.VLLMJudge prob mode: the photo thumbnailed to <= 896 px (never upscaled), JPEG q88,
"<question> Answer with one word: yes or no.", thinking off, P = P(yes) / (P(yes) + P(no)) over the top-20 first-token
logprobs. The model is loaded once per job.
  python eval/device_anchor82/judge.py --model=<dir or HF id> --tag=<j4|j9> --manifest=<parquet>[,...] --shard=k/K
      --out=<dir>     -> <dir>/<tag>_<k>.parquet (key, q, p)
"""
import sys
import time
from pathlib import Path

import pandas as pd

QUESTIONS = ["Is there a real dog anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?",
             "Is this a photo of food?",
             "Is a beach clearly visible in this photo?"]


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def load(p):
    from PIL import Image, ImageOps
    with Image.open(p) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def main():
    k, K = map(int, arg("shard", "0/1").split("/"))
    tag = arg("tag")
    out = Path(arg("out")); out.mkdir(parents=True, exist_ok=True)
    f = out / f"{tag}_{k}.parquet"
    if f.exists():
        print("done already"); return
    M = pd.concat([pd.read_parquet(m, columns=["key", "src"]) for m in arg("manifest").split(",")], ignore_index=True)
    M = M.iloc[k::K].reset_index(drop=True)
    from concurrent.futures import ThreadPoolExecutor
    from findpics.vlm import VLLMJudge
    t0 = time.time()
    J = VLLMJudge(arg("model"), gpu_mem=float(arg("gpu_mem", "0.85")))
    print(f"{tag}: loaded {arg('model')} in {time.time() - t0:.0f}s; {len(M)} photos", flush=True)
    pool, t0 = ThreadPoolExecutor(16), time.time()
    parts = []
    for c, s in enumerate(range(0, len(M), 1024)):      # preemptible queue: checkpoint every 1,024 photos
        fc = out / f"{tag}_{k}.chunk{c}.parquet"
        if not fc.exists():
            rows = []
            for s2 in range(s, min(s + 1024, len(M)), 256):
                part = M.iloc[s2:min(s2 + 256, s + 1024, len(M))]
                ims = list(pool.map(load, part.src))
                for q in QUESTIONS:
                    ps = J.p_yes(ims, q)
                    rows += [(key, q, float(p)) for key, p in zip(part.key, ps)]
            pd.DataFrame(rows, columns=["key", "q", "p"]).to_parquet(str(fc) + ".tmp"); Path(str(fc) + ".tmp").rename(fc)
        parts.append(pd.read_parquet(fc))
        print(f"{tag} {min(s + 1024, len(M))}/{len(M)} {time.time() - t0:.0f}s", flush=True)
    D = pd.concat(parts, ignore_index=True)
    D.to_parquet(str(f) + ".tmp"); Path(str(f) + ".tmp").rename(f)
    print(f"JUDGE_DONE {tag} {len(M)} photos x {len(QUESTIONS)} questions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
