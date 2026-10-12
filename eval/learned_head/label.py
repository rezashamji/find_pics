"""M48-scale step 2c: teacher labels for (question, photo) pairs = the PHONE judge: Qwen3-VL-4B at the phone's real
mlx-community 4-bit weights (models/q3vl4b_real), exactly the prompt / image route / P(yes) of findpics.vlm.VLLMJudge
(prob mode: "<question> Answer with one word: yes or no.", photo at native DISBench pixels through _data_url, P =
py/(py+pn) over the first token's top-20 logprobs) - the pipeline of eval/recall_audit/scores_q3vl4b_real and RESULTS 46/47.

Speed: requests are sent grouped BY PHOTO (image first in the prompt), so vLLM's encoder cache (keyed by image hash)
and prefix cache reuse the image's vision encoding and KV across the many questions asked about it.
Parity check (shard 0 only, before the real work): 96 audit4 pairs re-judged and compared with the stored P.

  python eval/learned_head/label.py --K=32  -> eval/learned_head/scale/labels/part{k}.parquet: every worker loads all
  photos once, then claims unfinished shards one at a time (mkdir lock in .cache/race_locks/lh_label/k; a worker that
  dies leaves its lock: remove it by hand to let another worker redo that shard).
Input: eval/learned_head/scale/jobs.parquet (question id, item_id), questions.json. Public data only.
"""
import io
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/learned_head"); sys.path.insert(0, "eval")
S = Path("eval/learned_head/scale")
OUT = S / "labels"
MODEL = os.path.abspath("models/q3vl4b_real")
CHUNK = 6144


def p_from(o, yes_ids, no_ids):
    lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
    py = sum(math.exp(v.logprob) for k, v in lp.items() if k in yes_ids)
    pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in no_ids)
    return py / (py + pn) if (py + pn) > 0 else 0.5


def main():
    import json
    from pack import load, PACK
    from findpics.media import load_image
    from findpics.vlm import _data_url, VLLMJudge
    K = int(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--K=")))    # shards are CLAIMED dynamically
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK = Path(".cache/race_locks/lh_label"); LOCK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Q = json.load(open(S / "questions.json"))
    jobs = pd.read_parquet(S / "jobs.parquet")
    need = set(jobs.item_id)
    raw = {}
    for p in range(16):
        ids, off, buf = load(p)
        for i, u in enumerate(ids):
            if u in need:
                raw[u] = buf[off[i]:off[i + 1]]
        del buf
    print(f"{len(jobs)} pairs, {len(need)} photos read ({len(need) - len(raw)} missing) {time.time() - t0:.0f}s", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as ex:
        urls = dict(zip(raw, ex.map(lambda b: _data_url(load_image(io.BytesIO(b))), raw.values())))
    del raw
    print(f"data urls {time.time() - t0:.0f}s", flush=True)
    J = VLLMJudge(model=MODEL, gpu_mem=0.85)
    llm, SP = J.llm, J.SP
    sp = SP(temperature=0.0, max_tokens=1, logprobs=20)
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)

    def run(pairs):          # pairs: list of (question text, data url) -> P(yes)
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                              {"type": "text", "text": q + " Answer with one word: yes or no."}]}]
                for q, u in pairs]
        outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        return [p_from(o, J.yes_ids, J.no_ids) for o in outs]

    if not (S / "parity.txt").exists():      # parity with the stored phone-judge run (audit4 photos read from their own paths)
        import glob
        sr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("eval/recall_audit/scores_q3vl4b_real/part*.parquet"))])
        smp = sr.sample(96, random_state=0)
        with ThreadPoolExecutor(16) as ex:
            us = list(ex.map(lambda p: _data_url(load_image(p)), smp.path))
        pn = np.array(run(list(zip(smp.question, us))))
        d = np.abs(pn - smp.p.to_numpy())
        from score_apple_photos import judge_cutoff
        cutv = np.array([judge_cutoff(q) for q in smp.question])
        msg = (f"PARITY 96 audit4 pairs: max |dP| {d.max():.4f}, median {np.median(d):.5f}, same side of 0.5: "
               f"{int(((pn >= .5) == (smp.p.to_numpy() >= .5)).sum())}/96; same side of the phone cut: "
               f"{int(((pn >= cutv) == (smp.p.to_numpy() >= cutv)).sum())}/96")
        print(msg, flush=True); (S / "parity.txt").write_text(msg + "\n")
    for k in range(K):
        if (OUT / f"part{k}.parquet").exists():
            continue
        try:
            (LOCK / str(k)).mkdir()
        except FileExistsError:
            continue
        (LOCK / str(k) / "owner").write_text(f"{os.environ.get('SLURM_JOB_ID')} {os.uname().nodename} {os.getpid()}\n")
        g = jobs[jobs.qid % K == k].sort_values(["item_id", "qid"])
        pairs = [(Q[q]["q"], urls[u]) for q, u in zip(g.qid, g.item_id)]
        P = []
        t1 = time.time()
        for b in range(0, len(pairs), CHUNK):
            P += run(pairs[b:b + CHUNK])
            print(f"  shard {k}: {b + len(pairs[b:b + CHUNK])}/{len(pairs)} {len(P) / (time.time() - t1):.1f} pairs/s",
                  flush=True)
        tmp = OUT / f".part{k}.{os.getpid()}.tmp"
        g.assign(p=np.asarray(P, np.float32)).to_parquet(tmp)
        os.replace(tmp, OUT / f"part{k}.parquet")
        print(f"shard {k} done: {len(g)} pairs {time.time() - t1:.0f}s", flush=True)
    print(f"LABEL_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
