"""RESULTS 61 step 1: the 27B rewrites its own JSON facts into the compact one-line format (TEXT ONLY, no image) for
every TRAIN + DEV photo (test libraries never opened). Greedy, thinking off, max 120 tokens.
  python eval/facts_writer/short_rewrite.py --users=U1,U2 [--model=Qwen/Qwen3.5-27B-FP8]
  -> eval/facts_writer/short27b/<user>.jsonl {item_id, facts: {"caption": <short line>}, n_out}
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "eval/complex_bench")
from short_format import REWRITE, clean, facts_json  # noqa: E402


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    users = arg("users").split(",")
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
    assert not set(users) & set(sp["test"]), "test closed"
    from common import load_facts
    from vllm import LLM, SamplingParams
    out_dir = Path("eval/facts_writer/short27b"); out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=arg("model", "Qwen/Qwen3.5-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.90,
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    sp_ = SamplingParams(temperature=0.0, max_tokens=120)
    for u in users:
        f = out_dir / f"{u}.jsonl"
        if f.exists():
            continue
        F = load_facts(u)
        ids = [i for i, x in F.items() if x]
        t0 = time.time()
        msgs = [[{"role": "user", "content": REWRITE.format(facts=facts_json(F[i]))}] for i in ids]
        outs = llm.chat(msgs, sp_, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        with open(str(f) + ".tmp", "w") as fo:
            for i, o in zip(ids, outs):
                fo.write(json.dumps(dict(item_id=str(i), facts={"caption": clean(o.outputs[0].text)},
                                         n_out=len(o.outputs[0].token_ids))) + "\n")
        Path(str(f) + ".tmp").rename(f)
        print(f"{u}: {len(ids)} rewritten in {time.time() - t0:.0f}s", flush=True)
    print("SHORT_DONE", flush=True)


if __name__ == "__main__":
    main()
