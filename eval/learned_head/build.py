"""M48: assemble (image vector, query text vector, teacher label) pairs from stored public judge runs. No model runs.

Sources (all public):
  JD     data/public/judge_distill: Qwen3.5-9B 16-bit P(yes), 2,412 planner-written questions x 48 DISBench photos each
         (24 best PE-Core-L matches + 24 random). The everyday concepts (dog, cat, car, bicycle, boat, beach, food, flowers,
         sunset, church, selfie, night) were removed from these questions when the set was made (judge_distill_data.py).
  OI     eval/oracle: Qwen3.5-9B 16-bit P(yes) on ALL 19,218 Open Images test-library items x 20 concepts (RESULTS 13/22).
         Image vectors: data/public/index_testlib_B (PE-Core-B-16 on the original file; videos = several frame vectors).
  EV16   eval/everyday16_*: 16 DISBench libraries (30,273 photos) x 12 everyday searches; judge-yes sets (cut 0.7) of
         Qwen3-VL-4B 16-bit (= eval/stop_rule/ev16_ranked.parquet, the RESULTS 27/41/42/45 truth), Qwen3.5-9B 16-bit and
         Qwen3.5-4B 16-bit, each run on every in-scope photo (no P stored: binary only).
  A4     eval/recall_audit: 4 DISBench libraries (7,886 photos) x 6 searches; P(yes) of Qwen3-VL-4B REAL phone 4-bit
         weights and of the simulated 4-bit weights on every photo; Qwen3.5-9B yes set from eval/everyday_q9b.
DISBench image vectors = phone simulation (480 px Lanczos + JPEG q80; whole + 2x2 tiles): eval/tiles_phone (A4, EV16) and
eval/learned_head/img_part*.npy (every other DISBench photo; embed.py).
Eye labels attached where they exist: A4 = RESULTS 34 blind labels + earlier labels + RESULTS 45 census; EV16 = RESULTS 45
ev16 sample + RESULTS 27 audit (eval/everyday16_audit/eye_labels.json).

  python eval/learned_head/build.py   -> eval/learned_head/pairs.parquet, vecs.npy (+ vec_keys.txt, nvec.npy)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/learned_head")
EVERYDAY_HELD = re.compile(r"(?i)\b(dogs?|puppy|puppies|cats?|kittens?|cars?|bicycles?|bikes?|boats?|beach(es)?|food|"
                           r"meals?|flowers?|sunsets?|church(es)?|selfies?|night)\b")       # = judge_distill_data.HELD_OUT

# query kinds for the eval searches (by what the question asks, see report)
KIND_FIXED = {
    "photos with a dog": "object", "photos with a car": "object", "photos with a bicycle": "object",
    "photos with a boat": "object", "beach photos": "scene", "sunset photos": "scene", "food photos": "scene",
    "photos of flowers": "scene", "photos of a church": "scene", "photos of a cat": "scene",
    "photos taken at night": "fact", "selfies": "person",
    "oi:bread": "object", "oi:baked_goods": "object", "oi:pizza": "object", "oi:cake": "object", "oi:sandwich": "object",
    "oi:dog": "object", "oi:bicycle": "object", "oi:guitar": "object", "oi:christmas_tree": "object",
    "oi:swimming_pool": "object", "oi:cat": "object", "oi:horse": "object", "oi:sunglasses": "object",
    "oi:wine_glass": "object", "oi:coffee_cup": "object", "oi:dog_on_beach": "fact", "oi:person_sunglasses": "person",
    "oi:birthday_candles": "fact", "oi:eating_pizza": "fact", "oi:bike_street": "fact"}

R_TEXT = re.compile(r"(?i)\b(receipt|screenshot|document|sign|signs|text|written|writing|words?|menu|email|whiteboard|"
                    r"notes?|letter|ticket|label|poster|invoice|chat|message|meme|form|page|book|list|card|certificate|"
                    r"newspaper|map|slide|spreadsheet|recipe|handwrit\w*|says|reads?)\b")
R_PERSON = re.compile(r"(?i)\b(person|people|man|woman|men|women|child|children|kids?|baby|babies|toddler|someone|anyone|"
                      r"he|she|his|her|girl|boy|bride|groom|husband|wife|dad|mom|mother|father|family|friends?|crowd|"
                      r"smiling|laughing|wearing|dressed|hair|face|selfie|couple|grandma|grandpa|guests?|students?|"
                      r"players?|team|audience|dancing|singing|posing|hugging)\b")
R_FACT = re.compile(r"(?i)\b(more than|at least|exactly|two|three|four|five|several|many|all|both|only|without|next to|"
                    r"in front of|behind|on top of|holding|under|above|beside|between|blurry|out of focus|dark|from the|"
                    r"taken (on|in|during)|date|january|february|march|april|may|june|july|august|september|october|"
                    r"november|december|folder|drone|aerial|same|first|last|while|being|during|before|after|rather than)\b")
R_SCENE = re.compile(r"(?i)^\s*(is|was)\s+this\s+(a\s+|an\s+)?(photo|picture|image|video|shot|pic)?\s*(clip\s+)?"
                     r"(taken\s+)?(of|at|in|inside|from|on)\b")


def jd_kind(q):
    """Kind of a planner question, first rule that fires: text in the image > person > fact/relation (counting, spatial
    relation, time, quality, action) > scene ('Is this a photo of/at ...') > object ('Is there / Are there ...')."""
    if R_TEXT.search(q):
        return "text"
    if R_PERSON.search(q):
        return "person"
    if R_FACT.search(q):
        return "fact"
    if R_SCENE.search(q):
        return "scene"
    if re.match(r"(?i)^\s*(is|are)\s+there\b", q):
        return "object"
    return "fact"


def disbench_vectors():
    keys, arrs = [], []
    ids = Path("eval/tiles_phone/ids.txt").read_text().split()
    keys += ids; arrs.append(np.load("eval/tiles_phone/vectors.npy")[:, :5])
    ids = Path("eval/tiles_phone/ev16_ids.txt").read_text().split()
    keys += ids; arrs.append(np.load("eval/tiles_phone/ev16_vectors.npy"))
    for f in sorted(OUT.glob("img_part*_ids.txt")):
        keys += f.read_text().split(); arrs.append(np.load(str(f).replace("_ids.txt", ".npy")))
    return keys, arrs


def oi_vectors():
    """Open Images test library, PE-Core-B-16: per item up to 5 unit vectors (photos 1, videos = sampled frames)."""
    root = Path("data/public/index_testlib_B")
    items = pd.read_parquet(root / "items.parquet")
    U, C = [], []
    for sd in sorted((root / "shards").glob("*/DONE")):
        U.append(pd.read_parquet(sd.parent / "units.parquet")); C.append(np.load(sd.parent / "clip.npy"))
    U = pd.concat(U, ignore_index=True); C = np.concatenate(C)
    row = {i: k for k, i in enumerate(items.item_id)}
    V = np.zeros((len(items), 5, C.shape[1]), np.float16); n = np.zeros(len(items), np.int8)
    for u, it in enumerate(U.item_id):
        r = row[it]
        if n[r] < 5:
            V[r, n[r]] = C[u]; n[r] += 1
    for r in range(len(items)):              # pad with the first vector (a photo's "tiles" = itself)
        V[r, n[r]:] = V[r, 0]
    return [f"oi:{r}" for r in range(len(items))], V, n, items


def main():
    from eval_oracle import QUERIES as OQ
    from eval_recall import QUERIES as A4Q, load_scores, old_labels, _labels_eye
    from eval_vector_enough import new_labels, truth_audit4
    from score_apple_photos import judge_cutoff
    texts = json.load(open(OUT / "texts.json")); tid = {t: k for k, t in enumerate(texts)}
    it = pd.read_parquet("data/public/index_disbench/items.parquet")
    user_of = dict(zip(it.item_id.astype(str), it.path.str.split("/").str[-2]))

    # ---------------------------------------------------------------------------------------------------- vectors
    keys, arrs = disbench_vectors()
    okeys, OV, on, oitems = oi_vectors()
    first = {}
    for k, key in enumerate(keys + okeys):
        first.setdefault(key, k)
    N = len(keys) + len(okeys)
    V = np.concatenate(arrs + [OV]).astype(np.float16)          # in RAM: mmap page faults on Lustre were slow (10-12)
    assert len(V) == N
    np.save(OUT / "vecs.npy", V)
    del arrs
    nvec = np.concatenate([np.full(len(keys), 5, np.int8), on])
    np.save(OUT / "nvec.npy", nvec)
    vrow = lambda key: first.get(key, -1)     # noqa: E731
    W = np.asarray(V[:, 0], np.float32)
    T = np.load(OUT / "texts.npy").astype(np.float32)

    def look_score(rows, looks, avoid=()):
        """The phone's fast score: mean similarity to the look texts minus mean similarity to the avoid texts, max over
        an item's frames for videos (whole-photo vector only; FindPicsCore lookScores)."""
        L = T[[tid[t] for t in looks]]
        s = np.einsum("nvd,ld->nvl", np.asarray(V[rows], np.float32), L).mean(-1)
        s = np.where(np.arange(5)[None] < nvec[rows][:, None], s, -9).max(1) if any(nvec[rows] < 5) else s[:, 0]
        if avoid:
            s = s - (W[rows] @ T[[tid[t] for t in avoid]].T).mean(1)
        return s

    P = []
    # ---------------------------------------------------------------------------------------------------------- JD
    jd = pd.DataFrame([json.loads(l) for f in sorted(glob.glob("data/public/judge_distill/part*.jsonl")) for l in open(f)])
    jd = jd.drop_duplicates(["question", "item_id"])
    jd = pd.DataFrame(dict(src="JD", lib=jd.item_id.map(user_of), query=jd.question, text=jd.question, item=jd.item_id,
                           p=jd.p9, y=jd.p9 >= 0.7, jd_part=jd.kind, kind=jd.question.map(jd_kind)))
    jd["vrow"] = jd.item.map(vrow)
    jd["look"] = np.nan
    print("JD", len(jd), "rows; dropped for a missing image vector:", int((jd.vrow < 0).sum()), flush=True)
    P.append(jd[jd.vrow >= 0])
    # ---------------------------------------------------------------------------------------------------------- OI
    for Q in OQ:
        d = pd.read_parquet(f"eval/oracle/{Q['qid']}.parquet")
        rows = np.array([first[f"oi:{r}"] for r in d.item_row])
        P.append(pd.DataFrame(dict(src="OI", lib="openimages", query="oi:" + Q["qid"], text=Q["q"],
                                   item=[f"oi:{r}" for r in d.item_row], p=d.p, y=d.p >= 0.7, vrow=rows,
                                   look=look_score(rows, Q["looks"]), kind=KIND_FIXED["oi:" + Q["qid"]])))
    # -------------------------------------------------------------------------------------------------------- EV16
    ev = pd.read_parquet("eval/stop_rule/ev16_ranked.parquet"); ev["item_id"] = ev.item_id.astype(str)
    plans = {}
    for f in sorted(glob.glob("eval/everyday16_q3vl/part*.json")):
        for r in json.load(open(f)):
            a = r["plan"]["albums"][0]
            plans[(r["query"], r["user"])] = (a["judge_question"], a.get("looks") or [], a.get("avoid") or [])
    ys = {}
    for tag, d in (("y9", "everyday16_9b"), ("y4", "everyday16_4b")):
        s = set()
        for f in sorted(glob.glob(f"eval/{d}/part*.json")):
            for r in json.load(open(f)):
                s |= {(r["query"], str(i)) for i in (r["exhaustive"] or [])}
        ys[tag] = s
    for (q, u), g in ev.groupby(["query", "user"]):
        jq, lk, av = plans[(q, u)]
        rows = g.item_id.map(vrow).to_numpy()
        P.append(pd.DataFrame(dict(src="EV16", lib=u, query=q, text=jq, item=g.item_id.to_numpy(), p=np.nan,
                                   y=g.yes.to_numpy(), vrow=rows,
                                   look=look_score(rows, lk, av) if lk else np.nan, kind=KIND_FIXED[q],
                                   y9=[(q, i) in ys["y9"] for i in g.item_id], y4=[(q, i) in ys["y4"] for i in g.item_id])))
    # ---------------------------------------------------------------------------------------------------------- A4
    sr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("eval/recall_audit/scores_q3vl4b_real/part*.parquet"))])
    ss = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("eval/recall_audit/scores/part*.parquet"))])
    sr["item_id"] = sr.item_id.astype(str); ss["item_id"] = ss.item_id.astype(str)
    psim = ss.set_index(["query", "item_id"]).p
    tl = json.load(open("eval/tiles_phone/texts.json"))["looks"]
    y9 = set()
    for f in sorted(glob.glob("eval/everyday_q9b/part*.json")):
        for r in json.load(open(f)):
            y9 |= {(r["query"], str(i)) for i in (r["exhaustive"] or [])}
    df = load_scores(); df["item_id"] = df.item_id.astype(str)
    st = df.set_index(["query", "item_id"]).stratum
    old, eye = old_labels(), _labels_eye()
    new, recs = new_labels()
    a4n = {(r["query"], r["item_id"]) for r in recs if r["source"] == "audit4"}
    new4 = {k: v for k, v in new.items() if k in a4n}
    for q in A4Q:
        g = sr[sr["query"] == q]
        cut = judge_cutoff(g.question.iloc[0])
        rows = g.item_id.map(vrow).to_numpy()
        P.append(pd.DataFrame(dict(src="A4", lib=g.user.to_numpy(), query=q, text=g.question.to_numpy(),
                                   item=g.item_id.to_numpy(), p=g.p.to_numpy(), y=g.p.to_numpy() >= cut, vrow=rows,
                                   p_sim=psim.loc[[(q, i) for i in g.item_id]].to_numpy(), cut=cut,
                                   look=look_score(rows, tl[q]), kind=KIND_FIXED[q],
                                   y9=[(q, i) in y9 for i in g.item_id],
                                   eye=[truth_audit4(q, i, st[(q, i)], old, eye, new4) for i in g.item_id])))
    A = pd.concat(P, ignore_index=True)
    # ------------------------------------------------------------------------------------------------- EV16 eye labels
    ev_eye = {(r["query"], r["item_id"]): r["label"] for r in json.load(open("eval/everyday16_audit/eye_labels.json"))}
    ev_eye = {k: {"right": "match", "wrong": "no_match", "unsure": "unsure"}[v] for k, v in ev_eye.items()}
    ev_eye.update({(r["query"], r["item_id"]): new[(r["query"], r["item_id"])] for r in recs
                   if r["source"] == "ev16" and (r["query"], r["item_id"]) in new})
    m = A.src == "EV16"
    A.loc[m, "eye"] = [ev_eye.get((q, i)) for q, i in zip(A["query"][m], A.item[m])]
    A["tid"] = A.text.map(tid)
    A["y"] = A.y.astype(bool)
    assert (A.vrow >= 0).all(), A[A.vrow < 0].src.value_counts()
    A.to_parquet(OUT / "pairs.parquet")
    print(A.groupby("src").agg(pairs=("y", "size"), queries=("query", "nunique"), photos=("item", "nunique"),
                               libs=("lib", "nunique"), teacher_yes=("y", "sum"),
                               eye=("eye", lambda s: int(s.notna().sum()))).to_string())
    print("JD kinds (questions):", A[A.src == "JD"].drop_duplicates("query").kind.value_counts().to_dict())
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
