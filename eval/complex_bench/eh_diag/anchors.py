# RESULTS 51 step 1: anchor-stage diagnostics on dev requests with an anchor (cached 9B judge only, CPU).
# For each request: every anchor hit (P, day, event), whether it is (on the day of) the truth program's anchor photo.
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "eval/complex_bench")
from findpics import store
from findpics.converse import CachedJudge, Plan, Album, place_or_look, filter_to_place
from findpics.engine import Thresholds, run_album
from findpics.models import ImageTextEncoder
from findpics.agent import events

class CacheOnly:
    def p_yes(self, images, question):
        raise RuntimeError(f"miss {question[:60]}")
Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
R = {r["qid"]: r for f in glob.glob("eval/complex_bench/runs/cmp/A4oldq4_search/part*.json") for r in json.load(open(f))}
S = {s["qid"]: s for s in json.load(open("eval/complex_bench/runs/cmp/r49c_A4oldq4_new_truthnew/scored.json"))}
want = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
idx = store.load("data/public/index_disbench")
user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
enc = ImageTextEncoder(idx.clip_model)
out = {}
SUBS, JS = {}, {}
for qid in sorted(R, key=lambda q: (q.split(":")[0], q)):
    r = R[qid]
    if want and qid not in want and not any(qid.endswith(":" + w) for w in want): continue
    if qid not in Q or qid not in S: continue
    if r.get("error") or not r["turns"]: continue
    q = Q[qid]
    for ti, t in enumerate(r["turns"]):
        P = Plan.model_validate(t["plan"])
        for ai, a in enumerate(P.albums):
            if not a.anchor: continue
            u = q["user"]
            if u not in SUBS:
                SUBS.clear(); JS.clear()
                SUBS[u] = store.subset(idx, np.where(user_of == u)[0])
                JS[u] = CachedJudge(CacheOnly(), None)
                for f in sorted(glob.glob(f"eval/complex_bench/runs/cmp/jcache/*/{u}.json")):
                    JS[u].cache.update(json.load(open(f)))
            sub, J = SUBS[u], JS[u]
            a2 = place_or_look(sub, filter_to_place(sub, a))
            anc = Album(name="anchor", looks=a2.anchor.looks, judge_question=a2.anchor.judge_question,
                        date_from=a2.date_from, date_to=a2.date_to, place=a2.place, media=a2.media)
            try:
                ar = run_album(sub, anc, enc, J, None, th=Thresholds(tail_budget=300))
            except RuntimeError as e:
                print(qid, ti, "MISS", e); continue
            tt = pd.to_datetime(sub.items.taken, utc=True, errors="coerce", format="ISO8601")
            ev = events(sub.items.taken)
            prog = q["program"][ti] if q["cat"] == "h" else q["program"]
            tids = set()
            for pa in prog:
                if pa and pa.get("anchor"): tids |= set(map(str, pa["anchor"].get("ids") or []))
            gt = set().union(*map(set, q["answers"][ti]))
            gt_days = sorted({str(tt.iloc[i].date()) for i, iid in enumerate(sub.items.item_id.astype(str)) if iid in gt and pd.notna(tt.iloc[i])})
            ret = ar.returned.sort_values("p_attr", ascending=False)
            hits = []
            for _, x in ret.iterrows():
                i = int(x.item_row)
                hits.append(dict(id=str(x.item_id), p=round(float(x.p_attr), 3), day=str(tt.iloc[i].date()) if pd.notna(tt.iloc[i]) else None,
                                 ev=int(ev[i]), truth_anchor=str(x.item_id) in tids, in_gt=str(x.item_id) in gt, look=round(float(x.look), 3)))
            # P of the truth anchor photo(s), if judged
            jd = ar.judged.set_index(ar.judged.item_id.astype(str))
            tp = {i: (round(float(jd.loc[i].p_attr if not isinstance(jd.loc[i], pd.DataFrame) else jd.loc[i].p_attr.max()), 3) if i in jd.index else None) for i in tids}
            jj = ar.judged.sort_values("p_attr", ascending=False).drop_duplicates("item_id").head(8)
            top_judged = [dict(id=str(x.item_id), p=round(float(x.p_attr), 3), day=str(tt.iloc[int(x.item_row)].date()) if pd.notna(tt.iloc[int(x.item_row)]) else None,
                               truth_anchor=str(x.item_id) in tids, in_gt=str(x.item_id) in gt) for _, x in jj.iterrows()]
            rec = dict(top_judged=top_judged, n_judged=int(ar.judged.item_id.nunique()), qid=qid, cat=q["cat"], turn=ti, f1=S[qid]["final"]["f1"], words=q["turns"][ti], window=a.window,
                       aq=a2.anchor.judge_question, place=a2.place, dates=[a2.date_from and str(a2.date_from), a2.date_to and str(a2.date_to)],
                       truth_anchor_p=tp, gt_days=gt_days[:12], n_gt_days=len(gt_days), hits=hits)
            out[f"{qid}#{ti}#{ai}"] = rec
            days = {}
            for h in hits: days.setdefault(h["day"], []).append(h["p"])
            print(f"{qid} t{ti} {q['cat']} F1 {rec['f1']:.2f} win={a.window} aq={rec['aq']!r} place={a2.place!r}")
            print(f"    truth anchor P {tp} gt_days({len(gt_days)}) {gt_days[:6]}")
            print("    top judged: " + "; ".join(f"{t['p']:.2f} {t['day']}{'*' if t['in_gt'] else ''}{'A' if t['truth_anchor'] else ''}" for t in top_judged[:5]) + f" (judged {rec['n_judged']})")
            print(f"    hits {len(hits)} on {len(days)} days: " + "; ".join(f"{d}:{len(ps)} max{max(ps):.2f}{'*' if d in gt_days else ''}" for d, ps in sorted(days.items(), key=lambda kv: -max(kv[1]))[:10]), flush=True)
json.dump(out, open("eval/complex_bench/eh_diag/anchors.json", "w"), indent=1)
