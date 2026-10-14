"""RESULTS 59 step 2: PHONE-SIZE facts. A small VLM at the phone's 4-bit weights (MLX affine 4-bit simulated:
models/qwen3vl_{2b,4b}_mlx4sim, scripts/sim_mlx_quant.py) writes the SAME structured facts as the 27B
(eval/complex_bench/facts.py PROMPT, same native-photo JPEG q92 input, greedy, max 700 tokens) for TRAIN + DEV
libraries. Records prompt (image + text) and output token counts per photo for the phone's index cost.
  python eval/facts_scorer/phone_facts.py --model=<dir|hf id> --tag=<tag> --users=U1,U2
  -> eval/facts_scorer/phone_facts/<tag>/<user>.jsonl  {item_id, facts, raw (if unparsed), n_prompt, n_out}
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "eval/complex_bench")
from facts import PROMPT, data_url, parse  # noqa: E402


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    model, tag = arg("model"), arg("tag")
    users = arg("users").split(",")
    out_dir = Path("eval/facts_scorer/phone_facts") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_ = json.loads(Path("eval/complex_bench/splits.json").read_text())
    assert not set(users) & set(sp_["test"]), "test libraries are closed"
    from concurrent.futures import ThreadPoolExecutor
    from vllm import LLM, SamplingParams
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    it["item_id"] = it.item_id.astype(str)
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=700)
    pool = ThreadPoolExecutor(16)
    for u in users:
        f = out_dir / f"{u}.jsonl"
        have = set()
        if f.exists():
            for line in open(f):
                try:
                    have.add(json.loads(line)["item_id"])
                except Exception:
                    pass
        lib = it[(it.user == u) & ~it.item_id.isin(have)]
        t0 = time.time(); n = 0
        B = 256
        for s in range(0, len(lib), B):
            part = lib.iloc[s:s + B]
            urls = list(pool.map(data_url, part.path))
            msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                  {"type": "text", "text": PROMPT}]}] for url in urls]
            try:
                outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            except TypeError:
                outs = llm.chat(msgs, sp, use_tqdm=False)
            with open(f, "a") as fo:
                for iid, o in zip(part.item_id, outs):
                    txt = o.outputs[0].text
                    fx = parse(txt)
                    fo.write(json.dumps(dict(item_id=iid, facts=fx, raw=None if fx is not None else txt[:2000],
                                             n_prompt=len(o.prompt_token_ids or []),
                                             n_out=len(o.outputs[0].token_ids))) + "\n")
            n += len(part)
            print(f"{u}: {n}/{len(lib)} photos, {time.time() - t0:.0f}s", flush=True)
    print("PHONE_FACTS_DONE", flush=True)


if __name__ == "__main__":
    main()
