"""RESULTS 61: a (student) writer writes the short facts line for photos of the given libraries, as on the phone:
the ~448 px rendition (Lanczos, JPEG q80) + STUDENT_PROMPT, greedy, repetition penalty 1.05, max 120 tokens.
  python eval/facts_writer/write_facts.py --model=<dir> --tag=<tag> --users=U1,U2 [--prompt=student|short27b]
  -> eval/facts_writer/facts/<tag>/<user>.jsonl {item_id, facts: {"caption": line}, n_out, n_prompt}
"""
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from short_format import STUDENT_PROMPT, clean  # noqa: E402


def data_url(path, max_side=448, q=80):
    """The phone rendition's own JPEG q80 bytes (same pixels as train_writer.phone_rendition)."""
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    users = arg("users").split(",")
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
    assert not set(users) & set(sp["test"]), "test closed"
    from vllm import LLM, SamplingParams
    out_dir = Path("eval/facts_writer/facts") / arg("tag"); out_dir.mkdir(parents=True, exist_ok=True)
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    it["item_id"] = it.item_id.astype(str); it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    llm = LLM(model=arg("model"), max_model_len=4096, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=256)
    sp_ = SamplingParams(temperature=0.0, max_tokens=120, repetition_penalty=1.05)
    pool = ThreadPoolExecutor(16)
    for u in users:
        f = out_dir / f"{u}.jsonl"
        if f.exists():
            continue
        lib = it[it.user == u]
        t0 = time.time(); recs = []
        for s in range(0, len(lib), 256):
            part = lib.iloc[s:s + 256]
            urls = list(pool.map(data_url, part.path))
            msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": x}},
                                                  {"type": "text", "text": STUDENT_PROMPT}]}] for x in urls]
            outs = llm.chat(msgs, sp_, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            recs += [dict(item_id=i, facts={"caption": clean(o.outputs[0].text)}, n_out=len(o.outputs[0].token_ids),
                          n_prompt=len(o.prompt_token_ids or [])) for i, o in zip(part.item_id, outs)]
        with open(str(f) + ".tmp", "w") as fo:
            fo.write("\n".join(json.dumps(r) for r in recs) + "\n")
        Path(str(f) + ".tmp").rename(f)
        print(f"{u}: {len(recs)} photos in {time.time() - t0:.0f}s", flush=True)
    print("WRITE_DONE", flush=True)


if __name__ == "__main__":
    main()
