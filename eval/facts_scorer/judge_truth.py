"""RESULTS 59, reference: the PRODUCT judge (Qwen3.5-9B, P(yes) from next-token probabilities) on exactly the
(photo, question) pairs the 27B oracle answered for a library's truth programs (eval/complex_bench/oracle/<user>.parquet),
so the facts scorer, the 9B judge and the oracle can be compared on the SAME questions.

Images as the product judge sees them: media.load_image + vlm._data_url (896 px max, JPEG q88; DISBench is ~500 px
native, so this is the native photo). Person keys "PERSON[name]::q": the whole photo with the red box on that person's
matched face, exactly as oracle.py asked them (oracle.person_matches). Prompt = q + " Answer with one word: yes or no."
(vlm.VLLMJudge.p_yes). Image-major batches so vLLM's prefix cache reuses the image tokens across questions.

Usage (vLLM env, GPU): python eval/facts_scorer/judge_truth.py --users=U1,U2 [--model=Qwen/Qwen3.5-9B]
  -> eval/facts_scorer/judge_truth/<tag>/<user>.parquet (item_id, q, p)   (restartable: finished libraries skipped)
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("eval/complex_bench")
sys.path.insert(0, str(ROOT))


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    users = arg("users").split(",")
    model = arg("model", "Qwen/Qwen3.5-9B")
    tag = arg("tag", model.split("/")[-1])
    out_dir = Path("eval/facts_scorer/judge_truth") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    from findpics import store
    from findpics.engine import _boxed
    from findpics.media import load_image
    from findpics.vlm import _data_url
    from oracle import person_matches
    from vllm import LLM, SamplingParams
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def p_of(o):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in yes_ids)
        pn = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in no_ids)
        return py / (py + pn) if py + pn > 0 else 0.5

    def msg(url, q):
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": q + " Answer with one word: yes or no."}]}]

    def chat(msgs):
        return llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})

    for u in users:
        out_f = out_dir / f"{u}.parquet"
        if out_f.exists():
            print(u, "done already", flush=True)
            continue
        O = pd.read_parquet(ROOT / "oracle" / f"{u}.parquet")
        O["item_id"] = O.item_id.astype(str)
        rows = np.where(user_of == u)[0]
        sub = store.subset(idx, rows)
        ids = sub.items.item_id.astype(str).to_numpy()
        paths = sub.items.path.to_numpy()
        isp = O.q.str.startswith("PERSON[")
        photo_q = O[~isp].groupby("item_id").q.apply(list).to_dict()
        t0 = time.time(); recs = []
        B = 32
        order = [i for i in range(len(ids)) if ids[i] in photo_q]
        for s in range(0, len(order), B):
            rr = order[s:s + B]
            urls = list(pool.map(lambda r: _data_url(load_image(paths[r])), rr))
            msgs, keys = [], []
            for r, url in zip(rr, urls):
                for q in photo_q[ids[r]]:
                    msgs.append(msg(url, q)); keys.append((ids[r], q))
            recs += [(i, q, p_of(o)) for (i, q), o in zip(keys, chat(msgs))]
            if s % (B * 10) == 0:
                print(f"{u}: {s + len(rr)}/{len(order)} photos, {len(recs)} answers, {time.time() - t0:.0f}s", flush=True)
        if isp.any():
            lib_meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
            pm = person_matches(sub, lib_meta)
            want = O[isp]
            for nm, (prow, frow, sc) in pm.items():
                w = want[want.q.str.startswith(f"PERSON[{nm}]::")]
                if not len(w):
                    continue
                qs_of = w.groupby("item_id").q.apply(list).to_dict()
                msgs, keys = [], []
                for r, fr in zip(prow, frow):
                    if ids[r] not in qs_of:
                        continue
                    url = _data_url(_boxed(sub, load_image(paths[r]), int(fr)))
                    for key in qs_of[ids[r]]:
                        msgs.append(msg(url, key.split("::", 1)[1])); keys.append((ids[r], key))
                for s in range(0, len(msgs), 2048):
                    recs += [(i, q, p_of(o)) for (i, q), o in zip(keys[s:s + 2048], chat(msgs[s:s + 2048]))]
        df = pd.DataFrame(recs, columns=["item_id", "q", "p"])
        df["p"] = df.p.astype("float32")
        df.to_parquet(out_f)
        print(f"{u}: {len(df)} answers (oracle {len(O)}) in {time.time() - t0:.0f}s", flush=True)
    print("JUDGE_TRUTH_DONE", flush=True)


if __name__ == "__main__":
    main()
