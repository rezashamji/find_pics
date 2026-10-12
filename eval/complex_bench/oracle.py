"""Complex-query benchmark, step 3: the ORACLE answers every yes/no question of a library's truth programs on EVERY
photo of that library (Qwen3.5-27B-FP8 looking at the native 500 px photo; P(yes) from next-token probabilities, the
product judge's prompt format). Person questions ("the person in the red box ...") are asked only on photos where the
named face group's person is found (product face model + query-time expansion), with the red box drawn on that face.

Image-major order: all questions about one photo are consecutive so vLLM's prefix cache reuses the image tokens.
Usage (vLLM env, GPU): python eval/complex_bench/oracle.py --shard=k/K
  -> eval/complex_bench/oracle/<user>.parquet (item_id, q, p) and <user>_persons.json
"""
import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402

MODEL = os.environ.get("FP_ORACLE_MODEL", "Qwen/Qwen3.5-27B-FP8")
NAMES = ["Sam", "Alex"]


def album_programs(req):
    progs = list(req.get("albums") or [])
    for tp in req.get("turn_programs") or []:
        progs += list(tp or [])
    return progs


def questions(req):
    """(photo questions, {person: [questions]})."""
    qs, pq = set(), {}
    for a in album_programs(req):
        if not isinstance(a, dict):
            continue
        for k in ("need", "need_any", "not"):
            qs |= {q.strip() for q in (a.get(k) or []) if isinstance(q, str) and q.strip()}
        anc = a.get("anchor") or {}
        if isinstance(anc, dict):
            qs |= {q.strip() for q in (anc.get("need") or []) if isinstance(q, str) and q.strip()}
        if a.get("person"):
            pq.setdefault(a["person"], set()).update(q.strip() for q in (a.get("person_need") or []) if isinstance(q, str))
    return qs, pq


def jpeg_url(im) -> str:
    b = io.BytesIO(); im.convert("RGB").save(b, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def person_matches(sub, lib_meta):
    """{name: (item_rows, face_rows)} for the face groups named in the digest (Sam = largest group >= 25 photos...)."""
    from findpics.people import expand_refs, item_person_scores
    from findpics.engine import person_cuts, Thresholds
    out = {}
    G = [g for g in lib_meta.get("face_groups", []) if g["n_items"] >= 25][:2]
    for nm, g in zip(NAMES, G):
        refs = expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3)
        s, best = item_person_scores(sub, refs)
        acc = person_cuts(sub, Thresholds())[0]
        rows = np.where(s >= acc)[0]
        out[nm] = (rows, best[rows], s[rows])
    return out


def run_oracle(llm, users, idx):
    from findpics import store
    from findpics.engine import _boxed
    from PIL import Image
    from vllm import SamplingParams
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def p_of(o):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in yes_ids)
        pn = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in no_ids)
        return py / (py + pn) if py + pn > 0 else 0.5

    (ROOT / "oracle").mkdir(exist_ok=True)
    for u in users:
        out_f = ROOT / "oracle" / f"{u}.parquet"
        P = json.loads((ROOT / "programs" / f"{u}.json").read_text())
        qs, pq = set(), {}
        for r in P["requests"]:
            a, b = questions(r)
            qs |= a
            for nm, x in b.items():
                pq.setdefault(nm, set()).update(x)
        done = pd.read_parquet(out_f) if out_f.exists() else pd.DataFrame(columns=["item_id", "q", "p"])
        have = set(done.q.unique())
        qs = sorted(qs - have)
        pkeys = {nm: sorted(f"PERSON[{nm}]::{q}" for q in v) for nm, v in pq.items()}
        pkeys = {nm: [x for x in v if x not in have] for nm, v in pkeys.items()}
        need_persons = bool(pq) and not (ROOT / "oracle" / f"{u}_persons.json").exists()
        if not qs and not any(pkeys.values()) and not need_persons:
            print(u, "done already", flush=True); continue
        rows = np.where(user_of == u)[0]
        sub = store.subset(idx, rows)
        ids = sub.items.item_id.astype(str).to_numpy()
        t0 = time.time(); recs = []
        lib_meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
        if qs:
            B = 64   # photos per batch -> B * len(qs) prompts
            for s in range(0, len(ids), B):
                urls = [jpeg_url(Image.open(p)) for p in sub.items.path.iloc[s:s + B]]
                msgs, keys = [], []
                for iid, url in zip(ids[s:s + B], urls):
                    for q in qs:
                        msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                                   {"type": "text", "text": q + " Answer with one word: yes or no."}]}])
                        keys.append((iid, q))
                outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
                recs += [(i, q, p_of(o)) for (i, q), o in zip(keys, outs)]
                if s % (B * 5) == 0:
                    print(f"{u}: {s + B}/{len(ids)} photos x {len(qs)} q, {time.time() - t0:.0f}s", flush=True)
        persons = {}
        if any(pkeys.values()) or need_persons:
            pm = person_matches(sub, lib_meta)
            for nm, (prow, frow, sc) in pm.items():
                persons[nm] = {str(ids[r]): float(s_) for r, s_ in zip(prow, sc)}
            for nm, keys_nm in pkeys.items():
                if nm not in pm or not keys_nm:
                    continue
                prow, frow, sc = pm[nm]
                msgs, keys = [], []
                for r, fr in zip(prow, frow):
                    url = jpeg_url(_boxed(sub, Image.open(sub.items.path.iloc[r]).convert("RGB"), int(fr)))
                    for key in keys_nm:
                        q = key.split("::", 1)[1]
                        msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                                   {"type": "text", "text": q + " Answer with one word: yes or no."}]}])
                        keys.append((str(ids[r]), key))
                for s in range(0, len(msgs), 2048):
                    outs = llm.chat(msgs[s:s + 2048], sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
                    recs += [(i, q, p_of(o)) for (i, q), o in zip(keys[s:s + 2048], outs)]
            (ROOT / "oracle" / f"{u}_persons.json").write_text(json.dumps(persons))
        df = pd.concat([done, pd.DataFrame(recs, columns=["item_id", "q", "p"])], ignore_index=True)
        df["p"] = df.p.astype("float32")
        df.to_parquet(out_f)
        print(f"{u}: {len(recs)} answers in {time.time() - t0:.0f}s ({len(qs)} photo q, "
              f"{sum(len(v) for v in pkeys.values())} person q)", flush=True)


def main():
    from findpics import store
    from vllm import LLM
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    users = sorted(p.stem for p in (ROOT / "programs").glob("*.json"))[k::K]
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, max_num_seqs=int(os.environ.get("FP_MAX_SEQS", "256")))
    run_oracle(llm, users, idx)


if __name__ == "__main__":
    main()
