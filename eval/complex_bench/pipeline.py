"""One GPU job, one model load (Qwen3.5-27B-FP8): for the given libraries, write requests + truth programs (gen.py)
and then answer every program question on every photo (oracle.py).
Usage (vLLM env, GPU with FP8): python eval/complex_bench/pipeline.py USER1,USER2,...   (or --shard=k/K over digests/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402


def main():
    arg = sys.argv[1]
    if arg.startswith("--shard="):
        k, K = map(int, arg.split("=", 1)[1].split("/"))
        users = sorted(p.stem for p in (ROOT / "digests").glob("*.txt"))[k::K]
    else:
        users = arg.split(",")
    from findpics import store
    from gen import gen_users, make_llm
    from oracle import run_oracle
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    llm = make_llm()
    gen_users(llm, users)
    run_oracle(llm, users, idx)


if __name__ == "__main__":
    main()
