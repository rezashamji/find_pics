"""RESULTS 59 step 2: anti-loop decoding settings for the small facts writers, on 256 DEV photos (one model load).
Greedy 2B looped in 80% of photos and 4B in 20% (10-14). Reports parse rate (strict JSON), mean/median output tokens,
share hitting the token cap. Usage: python eval/facts_scorer/phone_facts_sweep.py --model=<dir>
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/complex_bench")
from facts import PROMPT, data_url, parse  # noqa: E402


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    from vllm import LLM, SamplingParams
    sp_ = json.load(open("eval/complex_bench/splits.json"))
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    X = pd.concat([it[it.user == u].sample(32, random_state=0) for u in sp_["dev"][:8]])
    llm = LLM(model=arg("model"), max_model_len=4096, gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(p)}},
                                          {"type": "text", "text": PROMPT}]}] for p in X.path]
    for name, kw in [("greedy", {}), ("rep1.05", dict(repetition_penalty=1.05)), ("rep1.1", dict(repetition_penalty=1.1)),
                     ("rep1.2", dict(repetition_penalty=1.2)), ("pres1.5", dict(presence_penalty=1.5)),
                     ("freq0.5", dict(frequency_penalty=0.5)), ("pres0.5", dict(presence_penalty=0.5))]:
        outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=700, **kw), use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})
        n = np.array([len(o.outputs[0].token_ids) for o in outs])
        ok = [parse(o.outputs[0].text) is not None for o in outs]
        nobj = [len((parse(o.outputs[0].text) or {}).get("objects") or []) for o in outs]
        print(f"{name}: parsed {sum(ok)}/{len(ok)}, n_out mean {n.mean():.0f} median {np.median(n):.0f}, at cap "
              f"{(n >= 700).sum()}, objects/photo (parsed) {np.mean([x for x, k in zip(nobj, ok) if k]) if any(ok) else 0:.1f}",
              flush=True)
        print("   example:", outs[3].outputs[0].text[:400].replace("\n", " "), flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
