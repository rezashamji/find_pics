"""RESULTS 70 pre-scoring field-usefulness look (TRAIN only; no cascade or scorer numbers): for the towervec training
pairs of TRAIN libraries (eval/facts_scorer/data/pairs_train.parquet: 27B oracle P per photo-question; positives P >= .8,
negatives P < .2, person questions excluded), does a content word of the question appear in a FIELD of the photo's writer
line (t2b_final_rs2_xfit, the TRAIN lines the cascade scores)? Per field: hit rate on positives / negatives, lift = pos - neg,
writer tokens of the field, lift per 10 tokens, and positives where this field is the ONLY field with a hit.
  python eval/trunc70/lift.py > eval/trunc70/lift.txt"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

S = json.load(open("eval/complex_bench/splits.json"))
tok = AutoTokenizer.from_pretrained("models/qwen3vl_2b_mlx4sim")
STOP = set("is are there a an the in on of this photo photos picture image visible any some with at to and or for from by "
           "it its that which who what does do be being someone something person people one two three four five more than "
           "show shown shows showing taken scene can see seen inside outside near".split())
W = re.compile(r"[a-z]+")


def stem(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def fields(cap):
    out = defaultdict(str); cur = "(unnamed)"
    for sg in cap.split("; "):
        m = re.match(r"([a-z]+):\s*(.*)$", sg, re.S)
        if m:
            cur = m.group(1); out[cur] += " " + m.group(2)
        else:
            out[cur] += " " + sg
    return out


F, T = {}, defaultdict(list)
for u in S["train"]:
    for l in open(f"eval/facts_writer/facts/t2b_final_rs2_xfit/{u}.jsonl"):
        r = json.loads(l); cap = (r.get("facts") or {}).get("caption")
        if not cap:
            continue
        fd = fields(cap)
        F[str(r["item_id"])] = {k: {stem(w) for w in W.findall(v.lower())} for k, v in fd.items()}
        if len(T["scene"]) < 3000:
            for k, v in fd.items():
                T[k].append(len(tok(v.strip(), add_special_tokens=False).input_ids) + 2)   # + "name:" / "; " approx
P = pd.read_parquet("eval/facts_scorer/data/pairs_train.parquet", columns=["item_id", "qs", "p", "person"])
P = P[~P.person]
P["item_id"] = P.item_id.astype(str)
P = P[P.item_id.isin(F)]
pos = P[P.p >= .8]; neg = P[P.p < .2].sample(min(300000, (P.p < .2).sum()), random_state=70)
qw = {q: {stem(w) for w in W.findall(q.lower()) if w not in STOP and len(w) > 2} for q in P.qs.unique()}
names = ["scene", "objects", "colours", "people", "activity", "text", "look", "event"]


def rates(D):
    hit = defaultdict(int); only = defaultdict(int); n = 0; anyh = 0
    for iid, q in zip(D.item_id, D.qs):
        c = qw[q]
        if not c:
            continue
        n += 1; f = F[iid]
        h = [k for k in names if f.get(k) and c & f[k]]
        anyh += bool(h)
        for k in h:
            hit[k] += 1
        if len(h) == 1:
            only[h[0]] += 1
    return n, anyh, hit, only


np_, ap, hp, op = rates(pos); nn, an, hn, on_ = rates(neg)
print(f"TRAIN pairs with >= 1 content word: positives {np_} (any field hit {ap/np_:.3f}), negatives {nn} (sampled; any hit {an/nn:.3f})")
print("field | writer tokens mean (n lines) | pos hit | neg hit | lift | lift per 10 tokens | positives hit ONLY here")
for k in names:
    t = np.mean(T[k]) if T[k] else float("nan")
    lift = hp[k] / np_ - hn[k] / nn
    print(f"{k} | {t:.1f} ({len(T[k])}) | {hp[k]/np_:.3f} | {hn[k]/nn:.3f} | {lift:+.3f} | {10*lift/t:+.4f} | {op[k]/np_:.3f}")
