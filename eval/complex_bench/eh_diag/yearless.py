# RESULTS 51: albums whose dates come from a year-less phrase: is the window empty (at the album's place) in THIS
# library, and would the latest earlier year with photos hit the truth's days?
import json, glob, re, sys
from types import SimpleNamespace
import pandas as pd
from findpics.converse import Album
from findpics.engine import scope_mask
from findpics.converse import _years_back
from datetime import date
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {r["qid"]: r for f in glob.glob("eval/complex_bench/runs/cmp/A4oldq4_search/part*.json") for r in json.load(open(f))}
S = {s["qid"]: s for s in json.load(open("eval/complex_bench/runs/cmp/r49c_A4oldq4_new_truthnew/scored.json"))}
IT = pd.read_parquet("data/public/index_disbench/items.parquet")
IT["user"] = IT.path.str.extract(r"/images/images/([^/]+)/")[0]
REL = re.compile(r"(?i)\b(last|this|next|ago|yesterday|today|tonight|recent|recently|past|earlier|later|previous|coming|beginning|end of|weekend)\b")
for qid in sorted(R):
    if qid not in Q: continue
    r, q = R[qid], Q[qid]
    if r.get("error") or not r["turns"]: continue
    sub = SimpleNamespace(items=IT[IT.user == q["user"]].reset_index(drop=True))
    t = pd.to_datetime(sub.items.taken, utc=True, errors="coerce", format="ISO8601")
    for ti, tr in enumerate(r["turns"]):
        for a in tr["plan"]["albums"]:
            tp = a.get("time_phrase")
            if not tp or re.search(r"\b(19|20)\d\d\b", tp) or not a.get("date_from"): continue
            A = Album.model_validate(a)
            A2 = A.model_copy(update=dict(anchor=None, judge_question=None))
            n0 = int(scope_mask(sub, A2).sum())
            nd = int(scope_mask(sub, A2.model_copy(update=dict(place=None))).sum())
            gt = set().union(*map(set, q["answers"][ti]))
            gdays = sorted({str(x.date()) for x, i in zip(t, sub.items.item_id.astype(str)) if i in gt and pd.notna(x)})
            shift = None
            if n0 == 0:
                for k in range(1, 15):
                    df = _years_back(date.fromisoformat(A.date_from), k).isoformat()
                    dt = _years_back(date.fromisoformat(A.date_to), k).isoformat() if A.date_to else None
                    n = int(scope_mask(sub, A2.model_copy(update=dict(date_from=df, date_to=dt))).sum())
                    if n:
                        shift = (k, df, dt, n); break
            print(f"{qid} t{ti} {q['cat']} F1 {S[qid]['final']['f1']:.2f} tp={tp!r} rel={bool(REL.search(tp))} {A.date_from}..{A.date_to} "
                  f"place={A.place!r} media={A.media} in_window_at_place={n0} in_window={nd} shift={shift} truth_days={gdays[:4]}{'...' if len(gdays) > 4 else ''} n_truth_days={len(gdays)}")
