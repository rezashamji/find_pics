"""RESULTS 67 diagnosis: WHY the category c (person + look) truth programs are wrong.
(1) census of every kept dev + TRAIN c program (t63/queries = the corrected truth, adopted into queries/ 10-16);
(2) cause of every RESULTS 63 c eye error (t63 labels, already unblinded; used for diagnosis only, never for the RESULTS 67
    verification), from the oracle P of each program question on that photo (diag_t63.txt) and the label note.
Usage: python eval/complex_bench/t67/diag.py -> eval/complex_bench/t67/diag.txt"""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'eval/complex_bench')
import build  # noqa: E402
from common import norm  # noqa: E402

HERE = Path('eval/complex_bench/t67')
out = []
REQ = []
for s in ('dev', 'train'):
    REQ += [q for q in json.load(open(f'eval/complex_bench/t63/queries/{s}.json'))['queries'] if q['cat'] == 'c']
C = Counter()
ex = {}
PERSONW = re.compile(r"\b(person|man|men|woman|women|guy|guys|lady|girl|boy|kid|child|people|player|bride|groom|"
                     r"someone|hiker|climber|performer|model|goalie|musician|cyclist|tourist|marshal)s?\b", re.I)
for q in REQ:
    a = q['program'][0]; w = ' '.join(q['turns'])
    W = {norm(x) for x in re.findall(r"[a-zA-Z]{3,}", w.lower())}; W4 = {x[:4] for x in W}
    cue = bool(build.TIME_CUE.search(w))
    dated = any(a.get(k) for k in ('date_from', 'date_to', 'days', 'hours'))
    pl = a.get('place'); pls = ' '.join(pl) if isinstance(pl, list) else str(pl or '')
    pl_stated = bool(pl) and any(x.lower() in w.lower() for x in re.findall(r"[A-Za-z]{3,}", pls))
    qs = [x for k in ('need', 'need_any', 'person_need') for x in (a.get(k) or []) if isinstance(x, str)]
    unstated_q = [x for x in qs if not {c for c in build._content(x) if c in W or c[:4] in W4}]
    n_person_q = sum(bool(PERSONW.search(x)) for x in qs)
    C['requests'] += 1
    C[f"split {q['split']}"] += 1
    if a.get('person'):
        C['named person (Sam/Alex) album'] += 1
    if dated and not cue:
        C['date/day/hour constraint, words have NO time cue'] += 1; ex.setdefault('hidden date', w)
    if dated and cue:
        C['date constraint, words have a time cue'] += 1
    if pl and not pl_stated:
        C['place constraint the words never name'] += 1; ex.setdefault('hidden place', f"{w} -> {pls}")
    if (dated and not cue) or (pl and not pl_stated):
        C['ANY hidden date or place pin'] += 1
    if unstated_q:
        C['a question with no content word of the request (proxy / added condition)'] += 1; ex.setdefault('proxy q', f"{w} -> {unstated_q[0]}")
    if len(qs) >= 2 and not a.get('person') and n_person_q < len(qs) and PERSONW.search(w):
        C['person described in words, look split over separate whole-photo questions (binding lost)'] += 1
        ex.setdefault('split binding', f"{w} -> {qs}")
    if len(qs) >= 2:
        C['>= 2 AND-ed questions'] += 1
    if a.get('person_need') and not a.get('person'):
        C['person_need with no named person (asked of the whole photo)'] += 1
    n = len(q['answers'][-1][0])
    C['answer size 1-5'] += n <= 5; C['answer size > 50'] += n > 50
out.append(f"(1) CENSUS of the kept c programs (dev + TRAIN, t63 corrected truth): {C['requests']} requests "
           f"(dev {C['split dev']}, TRAIN {C['split train']})")
for k, v in C.items():
    if k not in ('requests',) and not k.startswith('split'):
        out.append(f"  {k}: {v}/{C['requests']}")
out.append("  examples: " + " | ".join(f"{k}: {v}" for k, v in ex.items()))

# (2) causes of the RESULTS 63 eye errors (manual, from diag_t63.txt: oracle P per question + label note)
CAUSE = {
 'ORACLE_YES_LT_.9: the oracle says yes (P .50-.89) to a literal question whose thing is NOT in the photo':
   [5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 34, 36, 75, 76, 78, 101, 149, 150, 176, 183, 184,
    243, 245, 246, 247, 248, 279, 284, 286, 374],
 'ORACLE_YES_GE_.9: near-miss look accepted at P >= .9 (red as pink, helmet as hat, jersey as jacket)': [35, 277, 285, 314],
 'BINDING: named person boxed, the look belongs to someone else in the photo': [77, 328],
 'PROXY: the program question is not what the words say (Elmo suit -> "red costume"; marshals -> "orange jumpsuit")': [194, 389],
}
MISS = {
 'HIDDEN_PIN: right photo outside an unstated day / date window / place': [114, 158, 168, 178, 360],
 'ORACLE_NO: the oracle says no (P < .5) to a stated thing that is there (red guitar, teal tie, pillow, log)': [186, 215, 229, 249],
 'RELATION: an over-decomposed relational question fails ("Is the parrot on the woman\'s chest?")': [237],
}
key = {k['n']: k for k in json.load(open('eval/complex_bench/t63/key.json'))}
named = {q['qid'] for q in REQ if q['program'][0].get('person')}
allc = sum(len(v) for v in CAUSE.values())
out.append(f"\n(2) CAUSES of the RESULTS 63 c eye errors (t63 labels; truth = the pre-63 truth T):")
out.append(f"  truth photos labelled no_match: {allc} (A11 6, FREM 27, KREM 11)")
for k, v in CAUSE.items():
    out.append(f"    {len(v):2d}  {k}   [named-person requests: {sum(key[n]['qid'] in named for n in v)}]")
alln = sum(len(v) for v in MISS.values())
out.append(f"  photos outside the truth labelled match: {alln} (all FADD)")
for k, v in MISS.items():
    out.append(f"    {len(v):2d}  {k}")
# what a stricter oracle cut does to the labelled truth photos (old programs)
lines = open(HERE / 'diag_t63.txt').read().splitlines()
rows = []
for l in lines:
    n = int(l.split()[0]); k = key[n]
    if k['cat'] != 'c' or k['T'] != 1:
        continue
    ps = [float(x) for x in re.findall(r"=([01](?:\.\d+)?)\b", l.split(' | ')[3])]
    eye = re.search(r"eye=(\w+)", l).group(1)
    if eye == 'unsure' or not ps:
        continue
    rows.append((k['stratum'], min(ps), eye == 'match'))
out.append("\n(3) The labelled TRUTH photos (old programs, T=1) by the lowest oracle P of their program's questions:")
for lo, hi in ((.5, .7), (.7, .9), (.9, 1.01)):
    g = [r for r in rows if lo <= r[1] < hi]
    out.append(f"  P [{lo:.1f}, {min(hi, 1):.1f}{')' if hi < 1 else ']'}: {sum(r[2] for r in g)}/{len(g)} right by eye")
a11 = [r for r in rows if r[0] == 'A11']
out.append(f"  A11 only (the corrected truth's main stratum): P < .9 {sum(r[2] for r in a11 if r[1] < .9)}/{sum(r[1] < .9 for r in a11)} right, "
           f"P >= .9 {sum(r[2] for r in a11 if r[1] >= .9)}/{sum(r[1] >= .9 for r in a11)} right")
out.append(f"  named-person A11 units: {sum(key[n]['qid'] in named for n in key if key[n]['cat']=='c' and key[n]['stratum']=='A11')}/15, of which wrong "
           f"{sum(key[n]['qid'] in named for n in [75,76,78,149,314])} (cardigan x3, sunglasses-on-head, hat=helmet); other A11 wrong: 277 (red as pink)")
(HERE / 'diag.txt').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
