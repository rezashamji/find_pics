"""Judge throughput: old serial loading (batch 48, decode then judge) vs new (draft decode on all cores, next batch
prefetched while the GPU judges, batch 96). Different photos for each arm (no cache effects). GPU, vLLM env."""
import os
import time

import numpy as np


def main():
    from findpics import engine as E, store
    from findpics.media import load_image
    from findpics.vlm import VLLMJudge, _data_url
    idx = store.load("data/public/index_testlib")
    photos = np.where(idx.items.media.to_numpy() == "photo")[0]
    rng = np.random.default_rng(0); rows = rng.choice(photos, 1920, replace=False)
    a, b = rows[:960], rows[960:]
    J = VLLMJudge(gpu_mem=0.8)
    q = "Is there bread in this photo?"
    J.p_yes([load_image(idx.items.path.iloc[int(r)]) for r in rows[:16]], q)   # warm-up

    def old(rs):
        out = []
        for s in range(0, len(rs), 48):
            ims = [load_image(idx.items.path.iloc[int(r)]) for r in rs[s:s + 48]]
            msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(im)}},
                     {"type": "text", "text": q + " Answer with one word: yes or no."}]}] for im in ims]
            J._chat(msgs, J.SP(temperature=0.0, max_tokens=1, logprobs=20)); out += [0] * len(ims)
        return out
    t = time.time(); old(a); t_old = time.time() - t
    t = time.time(); E._judge_rows(idx, J, b, np.full(len(b), -1), q); t_new = time.time() - t
    print(f"cores={len(os.sched_getaffinity(0))}  old: {960 / t_old:.1f} photos/s  new: {960 / t_new:.1f} photos/s  "
          f"(speedup {t_old / t_new:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
