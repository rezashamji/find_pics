"""Local vision-language model: (1) text-only planner calls, (2) yes/no judging of candidate images.

Judging returns P(yes) from the model's next-token probabilities (yes vs no), not a sampled word, so every
candidate gets a continuous score we can threshold and audit.
Backends: vLLM on Linux GPUs (fast batch). (Mac/MLX backend: see mac.py.)
"""
from __future__ import annotations

import base64
import io
import math
import os

from PIL import Image, ImageDraw

DEFAULT_VLM = os.environ.get("FP_VLM_MODEL", "Qwen/Qwen3.5-9B")


def draw_box(im: Image.Image, box, width_frac: float = 0.006) -> Image.Image:
    """Red box around the person of interest so the judge knows who 'the person in the red box' is."""
    im = im.copy()
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    # widen face box to include head/shoulders context
    w, h = x2 - x1, y2 - y1
    x1, y1, x2, y2 = max(0, x1 - 0.6 * w), max(0, y1 - 0.5 * h), min(im.width, x2 + 0.6 * w), min(im.height, y2 + 1.2 * h)
    d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(3, int(width_frac * max(im.size))))
    return im


def _data_url(im: Image.Image, max_side: int = 896) -> str:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


class VLLMJudge:
    def __init__(self, model: str = DEFAULT_VLM, max_model_len: int = 8192, gpu_mem: float = 0.85, tp: int = 1):
        from vllm import LLM, SamplingParams
        self.model = model
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       tensor_parallel_size=tp, limit_mm_per_prompt={"image": 1}, trust_remote_code=True)
        self.SP = SamplingParams
        tok = self.llm.get_tokenizer()
        self.yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
        self.no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}

    def _chat(self, msgs, sp):
        kw = {}
        try:  # Qwen3.x: disable "thinking" so the first token is the answer
            return self.llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            return self.llm.chat(msgs, sp, use_tqdm=False, **kw)

    def text(self, prompt: str, max_tokens: int = 1024) -> str:
        out = self._chat([[{"role": "user", "content": prompt}]], self.SP(temperature=0.0, max_tokens=max_tokens))
        return out[0].outputs[0].text

    def p_yes(self, images: list[Image.Image], question: str) -> list[float]:
        msgs = [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(im)}},
            {"type": "text", "text": question + " Answer with one word: yes or no."}]}] for im in images]
        outs = self._chat(msgs, self.SP(temperature=0.0, max_tokens=1, logprobs=20))
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = sum(math.exp(v.logprob) for k, v in lp.items() if k in self.yes_ids)
            pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in self.no_ids)
            res.append(py / (py + pn) if (py + pn) > 0 else 0.5)
        return res
