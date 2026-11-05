"""M82 cluster anchor: the facts WRITER through REAL MLX (mlx[cuda12] 0.32.3 + mlx-vlm 0.7.4, the RESULTS 61d parity
stack) with the product settings of FactsWriter.swift / Judge.generate: prompt "Search facts for this photo, one line.",
the 448 px rendition, greedy, max 120 new tokens, repetition penalty 1.05 over a 4,096-token context (prompt + output).
Repeat-stop v2 is applied afterwards (score.py) to the full greedy text: it is prefix-monotone, so this equals stopping
while streaming (findpics.facts_line docstring). Each model is loaded once per job.
Run inside the container (writer_job.sh):
  python eval/device_anchor82/writer_mlx.py --models=base=<mlx dir>,trained=<mlx dir> --manifest=<parquet>[,...]
      --shard=k/K --out=<dir>        -> <dir>/<writer>_<k>.jsonl {key, writer, text, segs, tokens, n_prompt, s}
  (--photos=<json list of {key, image}> instead of --manifest for the parity set)
"""
import json
import sys
import time
from pathlib import Path

PROMPT = "Search facts for this photo, one line."


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def main():
    k, K = map(int, arg("shard", "0/1").split("/"))
    if arg("photos"):
        photos = json.loads(Path(arg("photos")).read_text())
    else:
        photos = []       # the container venv has no pyarrow: prep.py writes manifest_<set>.json next to the parquet
        for m in arg("manifest").split(","):
            photos += [dict(key=r["key"], image=r["r448"]) for r in json.loads(Path(m.replace(".parquet", ".json")).read_text())]
    photos = photos[k::K]
    out = Path(arg("out")); out.mkdir(parents=True, exist_ok=True)
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    for spec in arg("models").split(","):
        name, mdir = spec.split("=", 1)
        f = out / f"{name}_{k}.jsonl"
        done = set()
        if f.exists():
            for x in open(f):          # a preempted job can leave a torn last line: skip it, rewrite that photo
                try:
                    done.add(json.loads(x)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
        todo = [p for p in photos if p["key"] not in done]
        print(f"{name}: {len(todo)} to write ({len(done)} done) from {mdir}", flush=True)
        if not todo:
            continue
        t_load = time.time()
        model, processor = load(mdir)
        prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
        print(f"{name}: loaded in {time.time() - t_load:.0f}s; prompt {prompt!r}", flush=True)
        t0 = time.time()
        with open(f, "a") as fo:
            for i, p in enumerate(todo):
                t1 = time.time(); segs, toks, n_prompt = [], [], None
                for r in stream_generate(model, processor, prompt, image=[p["image"]], max_tokens=120, temperature=0.0,
                                         repetition_penalty=1.05, repetition_context_size=4096):
                    segs.append(r.text)
                    if getattr(r, "token", None) is not None:
                        toks.append(int(r.token))
                    n_prompt = getattr(r, "prompt_tokens", n_prompt)
                fo.write(json.dumps(dict(key=p["key"], writer=name, text="".join(segs), segs=segs, tokens=toks,
                                         n_prompt=n_prompt, s=round(time.time() - t1, 3))) + "\n")
                fo.flush()
                if i % 50 == 0:
                    print(f"{name} {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        del model, processor
        print(f"{name}: WRITER_DONE {len(todo)} in {time.time() - t0:.0f}s", flush=True)
    print("ALL_WRITERS_DONE", flush=True)


if __name__ == "__main__":
    main()
