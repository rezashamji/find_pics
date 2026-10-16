"""RESULTS 67 (reported, not part of the decision rule): paired stratified bootstrap of F1 / P / R differences between
versions on the same eye labels (t63.estimate), over the requests the candidate keeps."""
import json, sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, 'eval/complex_bench'); sys.path.insert(0, 'eval/complex_bench/t63'); sys.path.insert(0, 'eval/complex_bench/t67')
from t63 import estimate
from t67 import T67, VERS, read_labels
U = pd.read_parquet(T67 / 'units.parquet'); key = json.loads((T67 / 'key.json').read_text()); lab = read_labels()
drops = {d['qid'] for d in json.loads((T67 / 'drops.json').read_text()) if d['version'] == 'R5'}
keep = set(U.qid) - drops
N = U[U.qid.isin(keep)].groupby('s').size().to_dict()
by = defaultdict(list)
for k in key:
    if k['qid'] in keep and lab[k['n']] != 'unsure':
        by[k['stratum']].append(int(lab[k['n']] == 'match'))
ins = {v: {s for s in N if s.startswith('P') and s[1 + VERS.index(v)] == '1'} for v in VERS}
def f(M, v):
    P, R, *_ = estimate(N, M, ins[v]); return P, R, (2 * P * R / (P + R) if P + R else 0)
rng = np.random.default_rng(0); out = defaultdict(list)
for _ in range(2000):
    M = {s: (int(rng.choice(x, len(x)).sum()), len(x)) for s, x in by.items()}
    a = {v: f(M, v) for v in VERS}
    for v in ('R5', 'R9', 'T9'):
        out[v].append([a[v][i] - a['T'][i] for i in range(3)])
M0 = {s: (sum(x), len(x)) for s, x in by.items()}
lines = [f"paired stratified bootstrap (2000), {len(keep)} requests R5 keeps; version - T: P, R, F1 [95%]"]
for v, d in out.items():
    d = np.array(d); pt = [f(M0, v)[i] - f(M0, 'T')[i] for i in range(3)]
    lines.append(f"  {v} - T: " + ", ".join(f"{n} {p:+.3f} [{np.percentile(d[:, i], 2.5):+.3f}, {np.percentile(d[:, i], 97.5):+.3f}]" for i, (n, p) in enumerate(zip('PR' + 'F', pt))))
(T67 / 'paired.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
