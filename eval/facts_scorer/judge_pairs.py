"""RESULTS 59: the product judge (default Qwen3.5-9B; same prompt and image encoding as vlm.VLLMJudge.p_yes) on an
arbitrary list of (item_id, question) pairs from a parquet (columns item_id, question) - whole photos, no boxes.
  python eval/facts_scorer/judge_pairs.py --pairs=<in.parquet> --out=<out.parquet> [--model=Qwen/Qwen3.5-9B]
"""
import math
import sys
from pathlib import Path

import pandas as pd


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    model = arg("model", "Qwen/Qwen3.5-9B")
    X = pd.read_parquet(arg("pairs"))[["item_id", "question"]].drop_duplicates()
    X["item_id"] = X.item_id.astype(str)
    from findpics.media import load_image
    from findpics.vlm import _data_url
    from vllm import LLM, SamplingParams
    from concurrent.futures import ThreadPoolExecutor
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    path = dict(zip(it.item_id.astype(str), it.path))
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    pool = ThreadPoolExecutor(8)
    ids = sorted(X.item_id.unique())
    urls = dict(zip(ids, pool.map(lambda i: _data_url(load_image(path[i])), ids)))
    msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": urls[i]}},
                                          {"type": "text", "text": q + " Answer with one word: yes or no."}]}]
            for i, q in zip(X.item_id, X.question)]
    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    P = []
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in yes_ids)
        pn = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in no_ids)
        P.append(py / (py + pn) if py + pn > 0 else 0.5)
    X["p"] = P
    Path(arg("out")).parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(arg("out"))
    print("JUDGE_PAIRS_DONE", len(X), flush=True)


if __name__ == "__main__":
    main()
