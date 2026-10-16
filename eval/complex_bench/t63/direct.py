"""RESULTS 63 (truth pass for categories b, c, g on dev + TRAIN): a DIRECT whole-request judgment per photo.

The bench's truth = a generated PROGRAM (decomposed yes/no questions + metadata constraints) answered by the 27B oracle.
This script asks an open VLM the OWNER'S WORDS directly, once per (request, album, photo), on EVERY photo of the
request's library (no program, no scope):
  "<photo date/time/place strip> Today is <today>. Someone searching their own photo library asked: "<words>".
   Does this photo belong in the answer? ... Answer with one word: yes or no."
g (two albums) asks the same per album ("... in the FIRST / SECOND group the request names?").
A named-person album (Sam / Alex) is asked only on photos where the bench's face match finds that person (the truth's
own identity constraint; identity is not re-derived here), with the red box drawn on that face, as oracle.py does.
P(yes) from next-token probabilities, the oracle's format (native <= 500 px photo, thinking off).

Usage (vLLM env, GPU): python eval/complex_bench/t63/direct.py <tag> <model> --shard=k/K [--tp=N]
       python eval/complex_bench/t63/direct.py <tag> <model> --units=<parquet of (qid, album, item_id)> --shard=k/K
  -> eval/complex_bench/t63/direct/<tag>/<user>.parquet  (qid, album, item_id, p)
Splits read: dev, train ONLY (the test split is closed).
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, load_items  # noqa: E402

CATS = ("b", "c", "g")
SPLITS = ("dev", "train")
ORD = ["FIRST", "SECOND"]


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def requests():
    out = []
    for s in SPLITS:
        for q in json.loads((ROOT / "queries" / f"{s}.json").read_text())["queries"]:
            if q["cat"] in CATS:
                out.append(q)
    return out


def place_words(addr) -> str:
    if not isinstance(addr, str) or not addr:
        return ""
    return ", ".join(p.strip() for p in addr.split(",") if p.strip() and not re.search(r"\d", p))


def prompt(q, k, t, addr, person):
    strip = f"Photo taken {t:%A %Y-%m-%d %H:%M} (camera clock)"
    pw = place_words(addr)
    strip += f", at: {pw}." if pw else ", no location recorded."
    words = " ".join(q["turns"])
    box = f" The person in the red box is {person}." if person else ""
    if q["cat"] == "g":
        ask = (f"The request asks for two groups of photos. Does this photo belong in the {ORD[k]} group the request "
               f"names? Every part of that group's description must hold (subject, look, people, activity, time, place).")
    else:
        ask = ("Does this photo belong in the answer? Every part of the request must hold (subject, look, people, "
               "activity, time, place).")
    return (f"{strip} Today is {q['today']}.{box}\nSomeone searching their own photo library asked: \"{words}\"\n"
            f"{ask} Answer with one word: yes or no.")


def main():
    tag, model = sys.argv[1], sys.argv[2]
    k_, K_ = map(int, arg("shard", "0/1").split("/"))
    from findpics import store
    from findpics.engine import _boxed
    from PIL import Image
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(HERE.parent))
    from oracle import jpeg_url, person_matches  # noqa: E402

    Q = requests()
    units = None
    if arg("units"):
        U = pd.read_parquet(arg("units"))
        units = {(r.qid, int(r.album)): set() for r in U[["qid", "album"]].drop_duplicates().itertuples()}
        for r in U.itertuples():
            units[(r.qid, int(r.album))].add(str(r.item_id))
    users = sorted({q["user"] for q in Q})[k_::K_]
    out_dir = ROOT / "t63" / "direct" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=float(arg("gpu-mem", "0.88")),
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, enable_prefix_caching=True,
              tensor_parallel_size=int(arg("tp", "1")), max_num_seqs=int(os.environ.get("FP_MAX_SEQS", "256")))
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def p_of(o):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in yes_ids)
        pn = sum(math.exp(v.logprob) for kk, v in lp.items() if kk in no_ids)
        return py / (py + pn) if py + pn > 0 else 0.5

    for u in users:
        out_f = out_dir / f"{u}.parquet"
        if out_f.exists():
            print(u, "done already", flush=True); continue
        t0 = time.time()
        rows = np.where(user_of == u)[0]
        sub = store.subset(idx, rows)
        ids = sub.items.item_id.astype(str).to_numpy()
        it = load_items(); it = it[it.user == u].set_index("item_id")
        lib_meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
        QU = [q for q in Q if q["user"] == u]
        need_p = any(isinstance(a, dict) and a.get("person") for q in QU for a in q["program"])
        pm = person_matches(sub, lib_meta) if need_p else {}
        face_of = {nm: {str(ids[r]): int(fr) for r, fr in zip(pr, fr_)} for nm, (pr, fr_, _) in pm.items()}
        # per photo: list of (qid, album, person or None)
        todo = {i: [] for i in ids}
        for q in QU:
            for k, a in enumerate(q["program"][: (2 if q["cat"] == "g" else 1)]):
                person = a.get("person") if isinstance(a, dict) else None
                allowed = set(face_of.get(person, {})) if person else None
                want = units.get((q["qid"], k)) if units is not None else None
                if units is not None and want is None:
                    continue
                for i in ids:
                    if (allowed is None or i in allowed) and (want is None or i in want):
                        todo[i].append((q, k, person))
        n_all = sum(len(v) for v in todo.values())
        recs = []
        B = 32
        order = [i for i in ids if todo[i]]
        for s in range(0, len(order), B):
            msgs, keys = [], []
            for i in order[s:s + B]:
                path = sub.items.path.iloc[int(np.where(ids == i)[0][0])]
                im = Image.open(path).convert("RGB")
                url_plain = None; url_box = {}
                t, addr = it.loc[i, "t"], it.loc[i, "place"]
                for q, k, person in todo[i]:
                    if person:
                        if person not in url_box:
                            url_box[person] = jpeg_url(_boxed(sub, im, face_of[person][i]))
                        url = url_box[person]
                    else:
                        url_plain = url_plain or jpeg_url(im); url = url_plain
                    msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                               {"type": "text", "text": prompt(q, k, t, addr, person)}]}])
                    keys.append((q["qid"], k, i))
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            recs += [(a, b, c, p_of(o)) for (a, b, c), o in zip(keys, outs)]
            if (s // B) % 10 == 0:
                print(f"{u}: {len(recs)}/{n_all} prompts, {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(recs, columns=["qid", "album", "item_id", "p"])
        df["p"] = df.p.astype("float32")
        df.to_parquet(out_f)
        print(f"{u}: {len(df)} answers in {time.time() - t0:.0f}s", flush=True)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
