"""RESULTS 67, step 1 (GPU, vLLM env): the 27B rewrites every dev + TRAIN category c (person + look) truth program with the
owner's words fixed (prompt REGEN_C below; rules from t67/diag.txt), then the same 27B oracle answers the NEW questions
on every photo of those libraries (person questions on the face-matched photos with the red box drawn, as oracle.py).
Writes ONLY under eval/complex_bench/t67/: programs_c2/<user>.json, oracle/<user>.parquet (new questions only; the base
answers stay in eval/complex_bench/oracle/, read-only here). Test split: never read (libraries from splits.json dev/train).

Usage: python eval/complex_bench/t67/regen.py --shard=k/K
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
from common import ROOT, load_items, whole_photo_q  # noqa: E402
from gen import SCHEMA, G_SCHEMA_EXTRA, make_llm, parse, places_list  # noqa: E402

T67 = ROOT / "t67"
SPLITS = ("dev", "train")

REGEN_C = """You are fixing the TRUTH PROGRAMS of a benchmark of photo-search requests. The OWNER of this photo library
asked a phone photo app for photos of people by how they look or what they are doing, in the request below. A draft truth
program (one album) was written. An eye audit found that drafts like these accept people who do not look as described,
put the look on the wrong person, pin the request to one day or place the owner never named, and add conditions the
owner never stated. Rewrite the album so that it selects EXACTLY the photos the owner means. The owner's words are
fixed; never change them. TODAY is {today}.

{digest}

PLACE NAMES IN THE ADDRESSES (name (photos, first day..last day)). A photo's address is a comma-separated list of such
names from street to country; "place" matches a photo whose address contains the name. Every level is listed
(country, region, city, town, landmark):
{places}

{schema}
{extra}

THE REQUEST: {words}

DRAFT ALBUM (may be wrong; reuse a question WORD FOR WORD when it is right):
{a0}
Draft note: {why}

THE ERRORS THE AUDIT FOUND, AND THE RULES:
1. ONE album holding EVERY photo of the library that matches the words.
2. Each person the owner describes = ONE question that ties that person to their look and activity together: "the woman
   in the red dress dancing on stage" -> "Is there a woman in a red dress dancing on a stage?". Never split one person's
   look over separate questions (draft error: "Is there a red dress?" AND "Is there a stage?" also matches a red dress
   on a hanger near a stage, and a guitar on one person with a stage behind someone else).
3. A named person (a name from PEOPLE) = "person": that name, and "person_need" = questions about "the person in the red
   box" (the photo shows a red box on that person's face): "photos of Sam in the beige cardigan" -> "Is the person in the
   red box wearing a beige cardigan?". The look must be on THAT person (draft error: Sam in a white top accepted because
   the woman next to her wore the cardigan; Sam next to a man holding a microphone accepted for "Sam holding a microphone").
4. Use the owner's own look words: the colour, garment, accessory, costume or activity exactly as said, never a broader or
   neighbouring one (draft errors: "the guy in the Elmo suit" -> "a person in a red costume"; "the marshals in the orange
   suits" -> any orange jumpsuit, which took the racing drivers; "a hat" accepted a climbing helmet; "pink" accepted red;
   "a jacket" accepted a football jersey). Where the words say where something is ("sunglasses on her head"), ask exactly that.
5. Never add a condition the owner did not state (draft errors: "Is there a thatched roof nearby?" added to "the man on the
   beach with the towel around his neck"; "Is there a water bottle?" added to "the guy in the blue beanie"; "Is there a
   wig?" added to "the women in costumes at the party"; "Is there a man with a beard?" added to "the guy getting shaved").
6. Never add a date, hour or place the owner did not say or clearly refer to. "the", "that" or "the one where" alone does
   not name an occasion: "the guy in the red jacket" = every photo of a man in a red jacket in the library, on any day;
   "the hiker with the red backpack" = every such photo on any day, not the one day the draft picked.
7. A trip, event, place or time the owner DOES name = its days, its place names from PLACE NAMES, or its date range
   ("in Montreal" = place "Montreal"; "from January" = that January; "last winter" = that winter, resolved against TODAY).
   An occasion the owner names ("at the wedding", "in the parade", "at that gig", "the fashion show") = ONE direct question
   about it ("Is this photo taken at a wedding?") plus its day(s) when the owner points at one occurrence ("the wedding",
   "that gig"), found in DAYS / DISTINCTIVE PHOTOS. Never test the occasion through objects it may contain.
8. "me", "us", "we", "I" = people visible in the photo (the owner cannot be recognised): "photos of us hiking near the
   waterfall" -> "Are there people hiking near a waterfall?". Numbers and groups the owner states are kept ("a large group").
9. Say "real" where a statue, painting, mural, toy or picture could match, unless the owner asks for one ("the person with
   the big feather headdress" is a real person, not a mosaic figure).
10. Questions: literal yes/no about what is VISIBLE in one photo, as precise as the words and no narrower or wider. Never
   ask about dates or places inside a question.
11. "well_defined": false ONLY if no program of these fields can select the photos the owner means from what is visible
   plus the time / place the owner states (say why in "why"); otherwise true. Still write your best program.

Return ONLY JSON: {{"albums": [the album program], "well_defined": true or false, "why": "one line: what changed from the
draft and why"}}"""

BOX = re.compile(r"red box", re.I)


def box_q(q: str) -> str:
    """A named-person question must say which person it is about (oracle.py asks it of the photo with a red box drawn on
    the matched face; "Is the person wearing X?" names nobody when several people are in the photo)."""
    q = q.strip()
    if BOX.search(q):
        return q
    if re.search(r"\bthe person\b", q, re.I):
        return re.sub(r"\b([Tt])he person\b", lambda m: m.group(1) + "he person in the red box", q, count=1)
    return "About the person in the red box: " + q


def c_requests():
    """{user: [(qid, request index, request dict from the corrected truth)]} for dev + TRAIN c requests."""
    out = {}
    for s in SPLITS:
        for q in json.loads((ROOT / "t63" / "queries" / f"{s}.json").read_text())["queries"]:
            if q["cat"] == "c":
                qi = int(re.search(r":c(\d+)$", q["qid"]).group(1))
                out.setdefault(q["user"], []).append((q["qid"], qi, q))
    return out


def regen(llm, users):
    from vllm import SamplingParams
    it = load_items()
    R = c_requests()
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=14000, seed=0)
    (T67 / "programs_c2").mkdir(exist_ok=True)
    users = [u for u in users if not (T67 / "programs_c2" / f"{u}.json").exists()]
    msgs, keys = [], []
    for u in users:
        raw = (ROOT / "digests" / f"{u}.txt").read_text()
        today, dig = raw.split("\n", 1)[0].split()[1], raw.split("\n", 1)[1]
        pl = places_list(u, it)
        P = json.loads((ROOT / "programs" / f"{u}.json").read_text())
        for qid, qi, q in R[u]:
            r = P["requests"][qi]
            assert r.get("cat") == "c" and r.get("turns") == q["turns"], qid
            p = REGEN_C.format(today=today, digest=dig, places=pl, schema=SCHEMA, extra=G_SCHEMA_EXTRA,
                               words=" ".join(map(str, q["turns"])), a0=json.dumps(q["program"][0]), why=q.get("why") or "")
            msgs.append([{"role": "user", "content": p}]); keys.append((u, qid))
    if not msgs:
        return
    print("regen_c prompts", len(msgs), flush=True)
    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": True})

    def ok(j):
        al = j.get("albums") if isinstance(j, dict) else None
        return isinstance(al, list) and len(al) == 1 and isinstance(al[0], dict)
    js = [parse(o.outputs[0].text) for o in outs]
    bad = [k for k, j in enumerate(js) if not ok(j)]
    if bad:   # one batched retry without thinking
        print("retry", len(bad), flush=True)
        o2 = llm.chat([msgs[k] for k in bad], SamplingParams(temperature=0.3, max_tokens=6000, seed=1), use_tqdm=False,
                      chat_template_kwargs={"enable_thinking": False})
        for k, o in zip(bad, o2):
            js[k] = parse(o.outputs[0].text)
    res = {u: {} for u in users}
    for (u, qid), j, o in zip(keys, js, outs):
        good = ok(j)
        al = j["albums"] if good else None
        if good and al[0].get("person"):
            al[0]["person_need"] = [box_q(x) for x in (al[0].get("person_need") or []) if isinstance(x, str) and x.strip()]
        res[u][qid] = dict(albums=al, well_defined=bool(j.get("well_defined", True)) if good else None,
                           why=(j or {}).get("why") if good else None, ok=good,
                           gen_tokens=len(o.outputs[0].token_ids), raw=o.outputs[0].text[-6000:])
        print(u, qid, "ok" if good else "FAILED", len(o.outputs[0].token_ids), flush=True)
    for u in users:
        (T67 / "programs_c2" / f"{u}.json").write_text(json.dumps(res[u], indent=1))


def questions(a):
    qs, pq = set(), set()
    for k in ("need", "need_any", "not"):
        qs |= {x.strip() for x in (a.get(k) or []) if isinstance(x, str) and x.strip()}
    if a.get("person"):
        pq |= {x.strip() for x in (a.get("person_need") or []) if isinstance(x, str) and x.strip()}
    else:
        qs |= {whole_photo_q(x) for x in (a.get("person_need") or []) if isinstance(x, str) and x.strip()}
    return qs, pq


def oracle_new(llm, users, idx):
    """oracle.py's run_oracle for the rewritten c albums, output under t67/oracle/ (questions not in the base parquet)."""
    from findpics import store
    from findpics.engine import _boxed
    from PIL import Image
    from vllm import SamplingParams
    from oracle import jpeg_url, person_matches
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

    (T67 / "oracle").mkdir(exist_ok=True)
    for u in users:
        out_f = T67 / "oracle" / f"{u}.parquet"
        X = json.loads((T67 / "programs_c2" / f"{u}.json").read_text())
        qs, pq = set(), {}
        for x in X.values():
            if not x.get("ok"):
                continue
            a = x["albums"][0]
            s1, s2 = questions(a)
            qs |= s1
            if a.get("person") and s2:
                pq.setdefault(a["person"], set()).update(s2)
        base = pd.read_parquet(ROOT / "oracle" / f"{u}.parquet", columns=["q"])
        have = set(base.q.unique())
        done = pd.read_parquet(out_f) if out_f.exists() else pd.DataFrame(columns=["item_id", "q", "p"])
        have |= set(done.q.unique())
        qs = sorted(qs - have)
        pkeys = {nm: [k for k in sorted(f"PERSON[{nm}]::{q}" for q in v) if k not in have] for nm, v in pq.items()}
        if not qs and not any(pkeys.values()):
            print(u, "nothing new", flush=True); continue
        rows = np.where(user_of == u)[0]
        sub = store.subset(idx, rows)
        ids = sub.items.item_id.astype(str).to_numpy()
        t0 = time.time(); recs = []
        if qs:
            B = 64
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
        if any(pkeys.values()):
            lib_meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
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
            (T67 / "oracle" / f"{u}_persons.json").write_text(json.dumps(persons))
        df = pd.concat([done, pd.DataFrame(recs, columns=["item_id", "q", "p"])], ignore_index=True)
        df["p"] = df.p.astype("float32")
        df.to_parquet(out_f)
        print(f"{u}: {len(recs)} answers in {time.time() - t0:.0f}s ({len(qs)} photo q, "
              f"{sum(len(v) for v in pkeys.values())} person q)", flush=True)


def main():
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    users = sorted(c_requests())[k::K]
    from findpics import store
    llm = make_llm()
    regen(llm, users)
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    oracle_new(llm, users, idx)
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
