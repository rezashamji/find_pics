"""M82 cluster anchor, cross-check engine: the same writer weights in vLLM (the de-quantized phone form that RESULTS 61d
showed equals the MLX 4-bit checkpoint weight for weight), same rendition bytes, prompt and sampling as writer_mlx.py
(vLLM's repetition penalty covers prompt + output). Raw text, no clean(), no repeat-stop (score.py applies it).
  python eval/device_anchor82/writer_vllm.py --model=<dir> --tag=<name> --manifest=<parquet>[,...] --out=<dir>
      -> <dir>/<tag>_vllm.jsonl {key, writer, text, tokens, n_prompt}
"""
import base64
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROMPT = "Search facts for this photo, one line."


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def main():
    M = pd.concat([pd.read_parquet(m, columns=["key", "r448"]) for m in arg("manifest").split(",")], ignore_index=True)
    out = Path(arg("out")); out.mkdir(parents=True, exist_ok=True)
    f = out / f"{arg('tag')}_vllm.jsonl"
    done = {json.loads(x)["key"] for x in open(f)} if f.exists() else set()
    M = M[~M.key.isin(done)]
    from vllm import LLM, SamplingParams
    llm = LLM(model=arg("model"), max_model_len=4096, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=120, repetition_penalty=1.05)
    t0 = time.time()
    with open(f, "a") as fo:
        for s in range(0, len(M), 512):
            part = M.iloc[s:s + 512]
            urls = ["data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode() for p in part.r448]
            msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                                  {"type": "text", "text": PROMPT}]}] for u in urls]
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            for key, o in zip(part.key, outs):
                fo.write(json.dumps(dict(key=key, writer=arg("tag"), text=o.outputs[0].text,
                                         tokens=list(o.outputs[0].token_ids), n_prompt=len(o.prompt_token_ids or []))) + "\n")
            fo.flush()
            print(f"{s + len(part)}/{len(M)} {time.time() - t0:.0f}s", flush=True)
    print("VLLM_WRITER_DONE", flush=True)


if __name__ == "__main__":
    main()
