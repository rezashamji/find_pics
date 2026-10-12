"""M48-scale step 2b: filter + de-duplicate the generated questions, then choose the photos each one is judged on.

  filter  (GPU or CPU; PE-Core-B-16 text tower): eval/learned_head/scale/qgen/part*.jsonl ->
          scale/questions.json   [{q, kind (writer's), jkind (build.jd_kind regex), seed_item, near_eval, near_jd}]
          scale/qtext.npy        PE-Core-B-16 text vectors (fp16, unit), same order
          scale/filter_report.txt
     Held out (dropped):  (1) the everyday words (judge_distill HELD_OUT) and a hand-written NEIGHBOURHOOD regex
          (vehicles, pets, boats/harbours, shore/sea, food/drink/eating, flowers/gardens, dusk/dawn, places of worship,
          selfies, night/dark/lights); (2) text cosine >= NEAR_EVAL to any everyday eval text (the 12 EV16 / A4 judge
          questions, their look phrases, the 5 OI everyday questions, and short concept prompts) - catches neighbours
          the regex misses. De-duplication: greedy, a question is dropped if its text cosine to an already kept one
          is >= DUP (paraphrases).
  jobs    (CPU): scale/questions.json + the pool's image vectors (eval/learned_head/img_part*.npy, whole-photo vector,
          phone simulation) -> scale/jobs.parquet (qid, item_id, how): per question the TOP_N pool photos by
          cosine(photo, question) + RAND_N random ones from the label shard's own random pool (shared photos -> the
          judge's image cache is reused).
Pool = DISBench photos of the 37 training libraries only (= eval/learned_head/pack.py's photos; the 20 EV16 / audit4
test libraries are never judged here). Public data only.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "eval/learned_head")
S = Path("eval/learned_head/scale")
NEAR_EVAL, DUP = 0.80, 0.95
TOP_N, RAND_N, K_LABEL, RAND_POOL = 250, 150, 32, 3000

HELD = re.compile(r"(?i)\b(dogs?|puppy|puppies|cats?|kittens?|cars?|bicycles?|bikes?|boats?|beach(es)?|food|"
                  r"meals?|flowers?|sunsets?|church(es)?|selfies?|night)\b")
NEIGH = re.compile(r"(?i)\b("
    # dog / cat / pets
    r"pets?|canine|hounds?|labrador|retriever|poodle|terrier|bulldog|beagle|husky|collie|corgi|spaniel|leash|kitty|"
    r"feline|paws?|"
    # car and road vehicles, traffic
    r"vehicles?|automobiles?|trucks?|vans?|suvs?|jeeps?|taxis?|cabs?|bus|buses|sedans?|minivans?|pickups?|"
    r"convertibles?|motorcycles?|motorbikes?|scooters?|mopeds?|traffic|parking|parked|garage|highway|road ?trip|"
    r"license plates?|tires?|tyres?|steering|dashboard|windshield|headlights?|"
    # bicycle
    r"cycl\w*|tricycles?|unicycles?|pedal\w*|"
    # boat
    r"ships?|yachts?|sail\w*|ferry|ferries|kayak\w*|canoe\w*|row ?boats?|rowing|dinghy|dinghies|harbou?rs?|marinas?|"
    r"docks?|piers?|vessels?|cruise|port|"
    # beach / sea
    r"shores?|shoreline|coast\w*|seaside|sand\w*|dunes?|surf(ing|er|ers|board|boards)?|waves?|ocean|sea|seas|bay|lagoon|tide|"
    # food / drink
    r"dish(es)?|dinner|lunch|breakfast|brunch|snacks?|desserts?|cakes?|cupcakes?|pizza|sandwich\w*|burgers?|fries|"
    r"pasta|noodles?|ramen|sushi|salads?|soups?|bread|pastr\w*|cookies?|fruits?|vegetables?|apples?|bananas?|"
    r"oranges?|lemons?|berries|strawberr\w*|grapes?|eat\w*|cook\w*|bak(e|ed|ing|ery)|restaurants?|cafes?|cafés?|"
    r"diner|plates?|bowls?|cups?|mugs?|drinks?|drinking|beverages?|coffee|tea|wine|beer|cocktails?|juice|bottles?|"
    r"glass(es)? of|ice cream|candy|chocolate|cheese|meat|steak|chicken|fish|seafood|grill\w*|barbecue|bbq|picnic|"
    r"kitchen|menu|tacos?|burritos?|donuts?|doughnuts?|pies?|muffins?|croissants?|bagels?|eggs?|rice|"
    r"chopsticks|forks?|spoons?|cutlery|buffet|feast|"
    # flowers / gardens
    r"floral|bouquets?|blossoms?|bloom\w*|roses?|tulips?|dais(y|ies)|sunflowers?|lil(y|ies)|orchids?|petals?|"
    r"gardens?|flowerbeds?|wreaths?|"
    # sunset
    r"sunrises?|dusk|dawn|twilight|golden hour|sundown|setting sun|"
    # church / worship
    r"cathedrals?|chapels?|basilicas?|abbey|monaster\w*|steeples?|spires?|altars?|pews?|stained glass|temples?|"
    r"mosques?|synagogues?|shrines?|worship|religious|crucifix|cross(es)?|nuns?|priests?|"
    # selfie
    r"self-portraits?|self portraits?|front camera|arm'?s length|"
    # night
    r"nighttime|night-time|darkness|in the dark|dark sky|dark outside|after dark|evening|nightlife|city lights|lit up|illuminated|neon|"
    r"fireworks?|stars|starry|moon\w*|long exposure|street ?lights?|lanterns?"
    r")\b")
CONCEPTS = ["a dog", "a puppy", "a cat", "a kitten", "a car", "a truck", "a bicycle", "a boat", "a ship", "a beach",
            "the sea", "food", "a meal", "a drink", "flowers", "a sunset", "a church", "a selfie", "a photo taken at night",
            "a dog on a beach", "a bicycle on a city street"]


def eval_texts():
    T = []
    import pandas as pd
    A = pd.read_parquet("eval/learned_head/pairs.parquet", columns=["src", "query", "text"])
    T += sorted(set(A[A.src.isin(["EV16", "A4"])].text))
    oi5 = {"oi:dog", "oi:cat", "oi:bicycle", "oi:dog_on_beach", "oi:bike_street"}
    T += sorted(set(A[A["query"].isin(oi5)].text))
    T += sorted({l for v in json.load(open("eval/tiles_phone/texts.json"))["looks"].values() for l in v})
    for f in sorted(glob.glob("eval/everyday16_q3vl/part*.json")):
        for r in json.load(open(f)):
            a = r["plan"]["albums"][0]
            T += list(a.get("looks") or [])
    T += [f"Is this a photo of {c}?" for c in CONCEPTS] + CONCEPTS
    return sorted(set(T))


def norm(q):
    q = re.sub(r"\s+", " ", q).strip()
    return q if q.endswith("?") else q.rstrip(".") + "?"


def filt():
    from build import jd_kind
    from findpics.models import ImageTextEncoder
    rows = [json.loads(l) for f in sorted(glob.glob(str(S / "qgen/part*.jsonl"))) for l in open(f)]
    raw = [(norm(q), kind, r["item_id"]) for r in rows for kind, q in r["questions"]]
    log = [f"seed photos {len(rows)}, raw questions {len(raw)}"]
    seen, uniq = set(), []
    for q, kind, it in raw:
        key = q.lower()
        if key not in seen and 12 <= len(q) <= 200:
            seen.add(key); uniq.append((q, kind, it))
    log.append(f"exact-unique (12-200 chars) {len(uniq)}")
    held = [(q, k, i) for q, k, i in uniq if HELD.search(q) or NEIGH.search(q)]
    keep = [(q, k, i) for q, k, i in uniq if not (HELD.search(q) or NEIGH.search(q))]
    log.append(f"dropped by everyday/neighbourhood regex {len(held)}; left {len(keep)}")
    rng = np.random.default_rng(0)
    log.append("  sample of regex-dropped: " + " | ".join(held[i][0] for i in rng.choice(len(held), 15, replace=False)))
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")

    def tx(L):
        V = np.concatenate([enc.texts(L[b:b + 512]) for b in range(0, len(L), 512)]).astype(np.float32)
        return V / np.linalg.norm(V, axis=1, keepdims=True)
    E = eval_texts()
    VE = tx(E)
    VQ = tx([q for q, _, _ in keep])
    near = (VQ @ VE.T)
    ne = near.max(1); arg = near.argmax(1)
    for lo, hi in ((0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.01)):
        m = np.where((ne >= lo) & (ne < hi))[0]
        log.append(f"  near-eval cos [{lo:.2f},{hi:.2f}): {len(m)}; e.g. " + " | ".join(
            f"{keep[i][0]} ~ {E[arg[i]]} ({ne[i]:.2f})" for i in m[rng.permutation(len(m))[:6]]))
    ok = ne < NEAR_EVAL
    log.append(f"dropped by text cosine >= {NEAR_EVAL} to an everyday eval text: {int((~ok).sum())}")
    keep = [x for x, o in zip(keep, ok) if o]; VQ = VQ[ok]; ne = ne[ok]
    # greedy de-dup in random order
    order = rng.permutation(len(keep))
    kept_idx = []
    Vk = np.zeros((0, VQ.shape[1]), np.float32)
    for b in range(0, len(order), 2048):
        blk = order[b:b + 2048]
        for i in blk:
            v = VQ[i]
            if len(Vk) and (Vk @ v).max() >= DUP:
                continue
            kept_idx.append(i); Vk = np.vstack([Vk, v[None]])
    log.append(f"de-duplicated at text cosine >= {DUP}: kept {len(kept_idx)} of {len(keep)}")
    # nearness to the 2,412 real planner questions (judge_distill): for the held-out-JD analysis
    jd = sorted({json.loads(l)["question"] for f in sorted(glob.glob("data/public/judge_distill/part*.jsonl")) for l in open(f)})
    VJ = tx(jd)
    out, vecs = [], []
    for i in kept_idx:
        q, kind, it = keep[i]
        out.append(dict(q=q, kind=kind, jkind=jd_kind(q), seed_item=it, near_eval=round(float(ne[i]), 4),
                        near_jd=round(float((VJ @ VQ[i]).max()), 4)))
        vecs.append(VQ[i])
    json.dump(out, open(S / "questions.json", "w"), indent=0)
    np.save(S / "qtext.npy", np.array(vecs, np.float16))
    np.save(S / "jdtext.npy", VJ.astype(np.float16)); json.dump(jd, open(S / "jd_questions.json", "w"))
    import collections
    log.append("kinds (writer): " + str(dict(collections.Counter(o["kind"] for o in out))))
    log.append("kinds (jd_kind regex): " + str(dict(collections.Counter(o["jkind"] for o in out))))
    log.append(f"near_jd >= 0.93 (a paraphrase of a real planner question): {sum(o['near_jd'] >= 0.93 for o in out)}")
    log.append("random sample of kept: " + " | ".join(out[i]["q"] for i in rng.choice(len(out), 40, replace=False)))
    (S / "filter_report.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log)); print("FILTER_DONE", flush=True)


def pool_vectors():
    ids, V = [], []
    for f in sorted(Path("eval/learned_head").glob("img_part*_ids.txt")):
        ids += f.read_text().split()
        V.append(np.load(str(f).replace("_ids.txt", ".npy"))[:, 0].astype(np.float32))
    V = np.concatenate(V)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    pool = set()
    for k in range(16):
        pool |= set(Path(f".cache/dpack/pack{k}_ids.txt").read_text().split())
    m = np.array([i in pool for i in ids])
    return [i for i, o in zip(ids, m) if o], V[m]


def jobs():
    import pandas as pd
    Q = json.load(open(S / "questions.json"))
    T = np.load(S / "qtext.npy").astype(np.float32)
    ids, V = pool_vectors()
    ids = np.array(ids)
    print("pool photos with vectors", len(ids), flush=True)
    rng = np.random.default_rng(0)
    rpools = [rng.choice(len(ids), RAND_POOL, replace=False) for _ in range(K_LABEL)]
    out = []
    for b in range(0, len(Q), 256):
        Sb = T[b:b + 256] @ V.T
        for j in range(Sb.shape[0]):
            qid = b + j
            top = np.argpartition(-Sb[j], TOP_N)[:TOP_N]
            rp = rpools[qid % K_LABEL]
            rp = rp[~np.isin(rp, top)]
            rnd = rng.choice(rp, RAND_N, replace=False)
            out.append(pd.DataFrame(dict(qid=qid, item_id=ids[np.concatenate([top, rnd])],
                                         how=["top"] * TOP_N + ["rand"] * RAND_N,
                                         cos=Sb[j, np.concatenate([top, rnd])].astype(np.float32))))
    J = pd.concat(out, ignore_index=True)
    J.to_parquet(S / "jobs.parquet")
    print(f"JOBS_DONE {len(J)} pairs, {J.qid.nunique()} questions, {J.item_id.nunique()} photos", flush=True)


if __name__ == "__main__":
    {"filter": filt, "jobs": jobs}[sys.argv[1]]()
