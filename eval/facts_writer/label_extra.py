"""RESULTS 61b: label the extra public photos with the SAME two-step 27B pipeline as the TRAIN targets: the 27B VLM
writes the complex_bench JSON facts (eval/complex_bench/facts.py PROMPT, JPEG q92 of the 500 px photo, greedy, 700 tokens),
then rewrites them TEXT-ONLY into the short line (short_format.REWRITE, greedy, 120 tokens).
  python eval/facts_writer/label_extra.py --shard=k/K   -> eval/facts_writer/extra_labels/part<k>.jsonl (restartable)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "eval/complex_bench")
from facts import PROMPT, data_url, parse  # noqa: E402
from short_format import REWRITE, clean, facts_json  # noqa: E402


def main():
    k, K = map(int, next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1").split("/"))
    M = pd.read_parquet("data/public/extra_photos/final.parquet").iloc[k::K]
    out = Path("eval/facts_writer/extra_labels"); out.mkdir(parents=True, exist_ok=True)
    f = out / f"part{k}.jsonl"
    have = set()
    if f.exists():
        for line in open(f):
            try:
                have.add(json.loads(line)["id"])
            except Exception:
                pass
    M = M[~M.id.isin(have)]
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3.5-27B-FP8", max_model_len=4096, gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    sp_f = SamplingParams(temperature=0.0, max_tokens=700)
    sp_s = SamplingParams(temperature=0.0, max_tokens=120)
    pool = ThreadPoolExecutor(16)
    t0 = time.time(); n = 0
    for s in range(0, len(M), 1024):
        part = M.iloc[s:s + 1024]
        urls = list(pool.map(data_url, part.path))
        outs = llm.chat([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                                       {"type": "text", "text": PROMPT}]}] for u in urls], sp_f,
                        use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        fx = [parse(o.outputs[0].text) for o in outs]
        ok = [j for j, x in enumerate(fx) if x]
        outs2 = llm.chat([[{"role": "user", "content": REWRITE.format(facts=facts_json(fx[j]))}] for j in ok], sp_s,
                         use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        short = dict(zip(ok, [(clean(o.outputs[0].text), len(o.outputs[0].token_ids)) for o in outs2]))
        with open(f, "a") as fo:
            for j, r in enumerate(part.itertuples()):
                fo.write(json.dumps(dict(id=r.id, src=r.src, path=r.path, facts=fx[j],
                                         n_out_facts=len(outs[j].outputs[0].token_ids),
                                         short=short.get(j, (None, 0))[0], n_out_short=short.get(j, (None, 0))[1])) + "\n")
        n += len(part)
        print(f"shard {k}: {n}/{len(M)} photos, {time.time() - t0:.0f}s", flush=True)
    print("LABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
