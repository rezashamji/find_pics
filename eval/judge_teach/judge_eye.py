"""RESULTS 58: any photo judge on every eye-labelled row of eval/judge_teach/eyeset.parquet, with the PRODUCT pipeline
(engine._judge_rows -> VLLMJudge.p_yes: 896-px thumbnail, JPEG q88, "<question> Answer with one word: yes or no.",
P = py / (py + pn) from the first token's top-20 logprobs; person rows = person crop + red box, crop_person="box").
Usage (vLLM env, GPU): python eval/judge_teach/judge_eye.py <name> <model dir or repo> [--tp=N] [--gpu-mem=0.85]
  -> eval/judge_teach/eye_p/<name>.parquet  (one P per eyeset row; same row order)
FP_JUDGE_THINK=1: let the model think first (enable_thinking, up to FP_THINK_TOKENS tokens), then read P(yes) of the
answer word that follows the thinking (teacher variant; never the product path).
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval/judge_teach/eye_p"


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


class ThinkJudge:
    """Teacher variant: think, then answer; P(yes) from the logprobs of the first non-space token after </think>
    (one generation with logprobs on every token). Cut off before </think> -> NaN (counted)."""

    def __init__(self, J):
        self.J = J
        tok = J.llm.get_tokenizer()
        self.end = tok.convert_tokens_to_ids("</think>")
        self.tok = tok

    def p_yes(self, images, question):
        from findpics.engine import _WORKERS
        from findpics.vlm import _data_url
        J = self.J
        urls = list(_WORKERS.map(_data_url, images))
        n = int(os.environ.get("FP_THINK_TOKENS", "1024"))
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                              {"type": "text", "text": question + " Answer with one word: yes or no."}]}]
                for u in urls]
        outs = J.llm.chat(msgs, J.SP(temperature=0.0, max_tokens=n, logprobs=20), use_tqdm=False,
                          chat_template_kwargs={"enable_thinking": True})
        res = []
        for o in outs:
            ids, lps = list(o.outputs[0].token_ids), o.outputs[0].logprobs or []
            p = float("nan")
            if self.end in ids:
                for j in range(ids.index(self.end) + 1, len(ids)):
                    if self.tok.decode([ids[j]]).strip():
                        lp = lps[j]
                        py = sum(math.exp(v.logprob) for k, v in lp.items() if k in J.yes_ids)
                        pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in J.no_ids)
                        p = py / (py + pn) if (py + pn) > 0 else 0.5
                        break
            res.append(p)
        return res


def main():
    name, model = sys.argv[1], sys.argv[2]
    os.environ["FP_VLM_MODEL"] = model
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    E = pd.read_parquet(ROOT / "eval/judge_teach/eyeset.parquet")
    if arg("sets"):
        E = E[E.set.isin(arg("sets").split(","))]
    idx = store.load(str(ROOT / "data/public/index_disbench"))
    J = VLLMJudge(model=model, gpu_mem=float(arg("gpu-mem", "0.85")), tp=int(arg("tp", "1")))
    if os.environ.get("FP_JUDGE_THINK") == "1":
        J = ThinkJudge(J)
    OUT.mkdir(parents=True, exist_ok=True)
    p = np.full(len(E), np.nan, np.float32)
    t0 = time.time()
    for (q, s), g in E.groupby(["question", "set"], sort=False):
        pos = np.array([E.index.get_loc(i) for i in g.index])
        crop = "box" if s == "person" else False
        p[pos] = _judge_rows(idx, J, g.item_row.to_numpy(int), g.face_row.to_numpy(int), q, crop_person=crop)
        print(f"{s} {q[:70]!r}: {len(g)} rows, {time.time() - t0:.0f}s", flush=True)
    out = E[["set", "search", "item_id", "face_row", "question"]].assign(p=p, model=model)
    out.to_parquet(OUT / f"{name}.parquet")
    print("DONE", name, len(out), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
