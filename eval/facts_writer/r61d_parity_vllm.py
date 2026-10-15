"""RESULTS 61d (2) parity, NOISE FLOOR: the same vLLM phone-form checkpoint on the parity photos, but one photo per
batch (max_num_seqs=1) instead of write_facts.py's 256-photo batches - a benign numeric change (different kernels /
batch shapes) that shows how far greedy decoding of this 2B moves without any change of weights or runtime.
Same decode settings as write_facts.py (greedy, repetition penalty 1.05, max 120). Output rows match r61d_parity_mlx.py.
  python eval/facts_writer/r61d_parity_vllm.py <q4 dir> <photos json> <out jsonl>
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from short_format import STUDENT_PROMPT, clean  # noqa: E402


def main():
    q4, photos, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    from vllm import LLM, SamplingParams
    llm = LLM(model=q4, max_model_len=4096, gpu_memory_utilization=0.5, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=False, max_num_seqs=1)
    sp = SamplingParams(temperature=0.0, max_tokens=120, repetition_penalty=1.05)
    with open(out, "w") as fo:
        for p in photos:
            url = "data:image/jpeg;base64," + base64.b64encode(Path(p["image"]).read_bytes()).decode()
            t0 = time.time()
            o = llm.chat([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                         {"type": "text", "text": STUDENT_PROMPT}]}]], sp, use_tqdm=False,
                         chat_template_kwargs={"enable_thinking": False})[0]
            fo.write(json.dumps(dict(item_id=p["item_id"], mlx=clean(o.outputs[0].text), tokens=list(o.outputs[0].token_ids),
                                     n_prompt=len(o.prompt_token_ids or []), ctx="vllm_bs1", s=round(time.time() - t0, 2))) + "\n")
    print("PARITY_VLLM_DONE", flush=True)


if __name__ == "__main__":
    main()
