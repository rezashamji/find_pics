"""RESULTS 58 step 2c: judge inputs packed once, then any judge labels every training pair.

  pack  --shard=k/K   build each pair's judge image exactly as the product does (engine._frame_for / load_image,
                      person rows: engine.person_crop_boxed = person crop + red box; then vlm._data_url's 896-px thumbnail
                      + JPEG q88) and store the JPEG bytes: data/public/judge_teach/img/part{k}.bin + part{k}.parquet
                      (img_key, off, len). Shards are by img_key, so every question about one image is in one shard.
  judge --shard=k/K --name=N --model=M [--tp=1]   P(yes) for every pair of the shard with the product prompt and
                      read-out (VLLMJudge.p_yes 'prob' mode), image-major order so vLLM's prefix cache reuses image
                      tokens -> data/public/judge_teach/labels/N/part{k}.parquet (pair, p)
Usage (vLLM env; pack on CPU or GPU nodes, judge on GPU).
"""
import base64
import io
import math
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/public/judge_teach"


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def shard_pairs(k, K):
    P = pd.read_parquet(D / "pairs.parquet")
    P["pair"] = np.arange(len(P))
    h = P.img_key.map(lambda s: zlib.crc32(s.encode()) % K)
    return P[h == k].sort_values(["img_key", "qid"]).reset_index(drop=True)


def pack(k, K):
    from findpics import store
    from findpics.engine import _frame_for, person_crop_boxed
    from findpics.media import load_image
    P = shard_pairs(k, K)
    keys = P.drop_duplicates("img_key")[["img_key", "pool", "item_row", "face_row", "path"]]
    idx = {"dis": store.load(str(ROOT / "data/public/index_disbench")), "oi": store.load(str(ROOT / "data/public/index_testlib"))}

    def one(r):
        try:
            if r.pool == "cm":
                im = load_image(r.path)
            else:
                I = idx[r.pool]
                im = _frame_for(I, int(r.item_row), int(r.face_row))
                if int(r.face_row) >= 0:
                    im = person_crop_boxed(I, im, int(r.face_row))
            im = im.copy(); im.thumbnail((896, 896))              # == vlm._data_url
            b = io.BytesIO(); im.save(b, format="JPEG", quality=88)
            return b.getvalue()
        except Exception as e:
            print("FAIL", r.img_key, e, flush=True)
            return b""
    (D / "img").mkdir(parents=True, exist_ok=True)
    t0 = time.time(); off = 0; rec = []
    with open(D / "img" / f"part{k}.bin.tmp", "wb") as fh, ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as ex:
        for i, b in enumerate(ex.map(one, keys.itertuples(index=False))):
            fh.write(b); rec.append((keys.img_key.iloc[i], off, len(b))); off += len(b)
            if i % 2000 == 0:
                print(f"pack {k}: {i}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    os.replace(D / "img" / f"part{k}.bin.tmp", D / "img" / f"part{k}.bin")
    pd.DataFrame(rec, columns=["img_key", "off", "len"]).to_parquet(D / "img" / f"part{k}.parquet")
    print("PACK_DONE", k, len(rec), f"{off / 1e9:.2f} GB", f"{time.time() - t0:.0f}s", flush=True)


def images(k):
    ix = pd.read_parquet(D / "img" / f"part{k}.parquet").set_index("img_key")
    buf = (D / "img" / f"part{k}.bin").read_bytes()
    return {key: buf[o:o + n] for key, (o, n) in zip(ix.index, ix[["off", "len"]].to_numpy())}


def judge(k, K, name, model, tp):
    from vllm import LLM, SamplingParams
    P = shard_pairs(k, K)
    img = images(k)
    out_f = D / "labels" / name / f"part{k}.parquet"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    kw = {"max_num_seqs": int(os.environ.get("FP_MAX_SEQS", "256"))}
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=float(arg("gpu-mem", "0.88")),
              tensor_parallel_size=tp, limit_mm_per_prompt={"image": 1}, trust_remote_code=True,
              enable_prefix_caching=True, **kw)
    tok = llm.get_tokenizer()
    yes = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    url = {}
    ps = np.full(len(P), np.nan, np.float32)
    t0 = time.time(); CH = 1024
    for s in range(0, len(P), CH):
        g = P.iloc[s:s + CH]
        msgs = []
        for key, q in zip(g.img_key, g.q):
            b = img.get(key, b"")
            if not b:
                msgs.append(None); continue
            u = url.get(key) or "data:image/jpeg;base64," + base64.b64encode(b).decode()
            url = {key: u}
            msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                                      {"type": "text", "text": q + " Answer with one word: yes or no."}]}])
        live = [i for i, m in enumerate(msgs) if m is not None]
        outs = llm.chat([msgs[i] for i in live], sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        for i, o in zip(live, outs):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = sum(math.exp(v.logprob) for t, v in lp.items() if t in yes)
            pn = sum(math.exp(v.logprob) for t, v in lp.items() if t in no)
            ps[s + i] = py / (py + pn) if (py + pn) > 0 else 0.5
        if (s // CH) % 10 == 0:
            r = (s + len(g)) / (time.time() - t0)
            print(f"judge {name} {k}: {s + len(g)}/{len(P)} {r:.1f} pairs/s", flush=True)
    pd.DataFrame({"pair": P.pair.to_numpy(), "p": ps}).to_parquet(out_f)
    print("JUDGE_DONE", name, k, len(P), int(np.isnan(ps).sum()), "nan", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    k, K = map(int, arg("shard", "0/1").split("/"))
    if sys.argv[1] == "pack":
        pack(k, K)
    else:
        judge(k, K, arg("name"), arg("model"), int(arg("tp", "1")))
