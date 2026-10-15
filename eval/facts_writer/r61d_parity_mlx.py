"""RESULTS 61d (2) parity, step 2: run the MLX checkpoint (eval/facts_writer/to_mlx.py output) with REAL MLX (mlx[cpu] +
mlx-vlm, tools/mlxpy/venv inside the Debian container tools/mlxpy/py312.sif) on the parity photos, greedy, max 120 new
tokens, repetition penalty 1.05 over a context of N tokens (prompt + generated; mlx-vlm's processor sees the prompt ids
on the first step): N=100000 = vLLM's semantics (every prompt and generated token), N=20 = MLX's / mlx-swift-lm's default.
  singularity exec --cleanenv -B $PWD:$PWD tools/mlxpy/py312.sif tools/mlxpy/venv/bin/python \
      eval/facts_writer/r61d_parity_mlx.py <mlx dir> <photos json> <out jsonl> [--ctx=100000]
"""
import json
import sys
import time
from pathlib import Path

STUDENT_PROMPT = "Search facts for this photo, one line."   # = short_format.STUDENT_PROMPT (no torch import here)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    ctx = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ctx=")), 100000))
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(pos[0])
    photos = json.loads(Path(pos[1]).read_text())
    out = Path(pos[2]); done = set()
    if out.exists():
        done = {json.loads(x)["item_id"] for x in open(out)}
    with open(out, "a") as fo:
        for p in photos:
            if p["item_id"] in done:
                continue
            prompt = apply_chat_template(processor, model.config, STUDENT_PROMPT, num_images=1)
            t0 = time.time(); toks, text, n_prompt = [], "", None
            for r in stream_generate(model, processor, prompt, image=[p["image"]], max_tokens=120, temperature=0.0,
                                     repetition_penalty=1.05, repetition_context_size=ctx):
                text += r.text
                if getattr(r, "token", None) is not None:
                    toks.append(int(r.token))
                n_prompt = getattr(r, "prompt_tokens", n_prompt)
            fo.write(json.dumps(dict(item_id=p["item_id"], mlx=" ".join(text.split()), tokens=toks, n_prompt=n_prompt,
                                     ctx=ctx, s=round(time.time() - t0, 1))) + "\n")
            fo.flush()
            print(p["item_id"], len(toks), f"{time.time() - t0:.1f}s", flush=True)
    print("PARITY_MLX_DONE", flush=True)


if __name__ == "__main__":
    main()
