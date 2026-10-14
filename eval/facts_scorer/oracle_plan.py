"""RESULTS 59 fix B: the 27B ORACLE (same model, prompt and image encoding as eval/complex_bench/oracle.py) on the
judge questions the SHIPPED planner + grounding actually ask for TRAIN requests (harvested from a TRAIN replay log), on
every photo of each TRAIN library -> extra training pairs in the product's own question wording.
  python eval/facts_scorer/oracle_plan.py --users=U1,U2 --questions=<parquet: user, question>
  -> eval/facts_scorer/data/oracle_plan/<user>.parquet (item_id, q, p)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/complex_bench")


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    users = arg("users").split(",")
    sp_ = json.loads(Path("eval/complex_bench/splits.json").read_text())
    assert set(users) <= set(sp_["train"]), "TRAIN libraries only"
    QS = pd.read_parquet(arg("questions"))
    out_dir = Path("eval/facts_scorer/data/oracle_plan"); out_dir.mkdir(parents=True, exist_ok=True)
    from oracle import jpeg_url
    from PIL import Image
    from vllm import LLM, SamplingParams
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    it["item_id"] = it.item_id.astype(str)
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    llm = LLM(model=arg("model", "Qwen/Qwen3.5-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def p_of(o):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in yes_ids)
        pn = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in no_ids)
        return py / (py + pn) if py + pn > 0 else 0.5

    for u in users:
        f = out_dir / f"{u}.parquet"
        if f.exists():
            continue
        qs = sorted(QS[QS.user == u].question.unique())
        lib = it[it.user == u]
        if not qs:
            continue
        t0 = time.time(); recs = []
        B = 32
        for s in range(0, len(lib), B):
            part = lib.iloc[s:s + B]
            urls = list(pool.map(lambda p: jpeg_url(Image.open(p)), part.path))
            msgs, keys = [], []
            for iid, url in zip(part.item_id, urls):
                for q in qs:
                    msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                               {"type": "text", "text": q + " Answer with one word: yes or no."}]}])
                    keys.append((iid, q))
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            recs += [(i, q, p_of(o)) for (i, q), o in zip(keys, outs)]
            if s % (B * 10) == 0:
                print(f"{u}: {s + len(part)}/{len(lib)} photos x {len(qs)} q, {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(recs, columns=["item_id", "q", "p"]); df["p"] = df.p.astype("float32")
        df.to_parquet(f)
        print(f"{u}: {len(df)} answers in {time.time() - t0:.0f}s", flush=True)
    print("ORACLE_PLAN_DONE", flush=True)


if __name__ == "__main__":
    main()
