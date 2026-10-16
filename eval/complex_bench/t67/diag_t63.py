"""RESULTS 67 diagnosis input: every t63 c eye unit with its program and the oracle P of each program question on that photo
(t63 labels are already unblinded; used for diagnosis only, never for verification)."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, 'eval/complex_bench'); sys.path.insert(0, 'eval/complex_bench/t63')
from common import ROOT, whole_photo_q
Q = {}
for s in ('dev', 'train'):
    for q in json.load(open(f'eval/complex_bench/t63/queries/{s}.json'))['queries']:
        Q[q['qid']] = q
key = json.load(open('eval/complex_bench/t63/key.json'))
lab = {}
for l in open('eval/complex_bench/t63/labels_main.txt'):
    if l.startswith('#') or not l.strip(): continue
    p = l.rstrip('\n').split(None, 2); lab[int(p[0])] = (p[1], p[2] if len(p) > 2 else '')
O = {}
def P(u, qtext, iid):
    if u not in O:
        d = pd.read_parquet(ROOT / 'oracle' / f'{u}.parquet'); d['item_id'] = d.item_id.astype(str)
        O[u] = {(r.q, r.item_id): r.p for r in d.itertuples()}
    return O[u].get((qtext, iid))
for k in sorted([k for k in key if k['cat'] == 'c'], key=lambda k: (k['stratum'], k['qid'])):
    q = Q.get(k['qid'])
    if q is None:
        print(k['n'], 'qid gone', k['qid']); continue
    a = q['program'][0]; u = q['user']
    qs = [(x, x) for x in (a.get('need') or [])] + [('ANY:' + x, x) for x in (a.get('need_any') or [])] + [('NOT:' + x, x) for x in (a.get('not') or [])]
    if a.get('person'):
        qs += [('P:' + x, f"PERSON[{a['person']}]::{x.strip()}") for x in a.get('person_need') or []]
    else:
        qs += [('W:' + x, whole_photo_q(x)) for x in a.get('person_need') or []]
    ps = ' ; '.join(f"{n[:45]}={P(u, t.strip(), k['item_id']) if P(u, t.strip(), k['item_id']) is None else round(P(u, t.strip(), k['item_id']), 2)}" for n, t in qs)
    meta = {x: a.get(x) for x in ('days', 'date_from', 'date_to', 'place', 'person') if a.get(x)}
    print(f"{k['n']} {k['stratum']} T={k['T']} {k['qid']} | {' || '.join(q['turns'])} | eye={lab[k['n']][0]} ({lab[k['n']][1][:70]}) | {ps} | {json.dumps(meta)}")
