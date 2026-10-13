# RESULTS 51, falsification probe for the "closest moment" fallback: a same_day anchor that matches nothing falls back to
# the best-scoring photo if its P >= floor. How often does a moment that is NOT in the library reach the floor?
# Each dev e request's same_day anchor question (adapter q4 plans) is asked in 4 OTHER dev libraries (seeded), anchor
# stage as the product runs it but head only (300 by vector rank; the truth anchors of the dev e requests were all in
# the head). 9B judge, answers cached in the shared jcache under run name "probe51".
import json, glob, sys
from pathlib import Path
import numpy as np
from findpics import store
from findpics.converse import Album, CachedJudge, Plan
from findpics.engine import Thresholds, run_album
from findpics.models import ImageTextEncoder
from findpics.vlm import VLLMJudge
def main():
    Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
    R = {r["qid"]: r for f in glob.glob("eval/complex_bench/runs/cmp/A4oldq4_search/part*.json") for r in json.load(open(f))}
    anchors = []
    for qid, r in sorted(R.items()):
        if qid not in Q or Q[qid]["cat"] != "e" or r.get("error") or not r["turns"]:
            continue
        for a in r["turns"][-1]["plan"]["albums"]:
            if a.get("anchor") and a.get("window") == "same_day" and "Everest" not in (a["anchor"]["judge_question"] or ""):
                anchors.append((qid, a["anchor"]["looks"], a["anchor"]["judge_question"]))
    users = sorted({q["user"] for q in Q.values()})
    rng = np.random.default_rng(51)
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    J0 = VLLMJudge(gpu_mem=0.7)
    cache_dir = Path("eval/complex_bench/runs/cmp/jcache")
    plan = {}
    for qid, looks, aq in anchors:
        own = qid.split(":")[0]
        for u in rng.choice([x for x in users if x != own], size=4, replace=False):
            plan.setdefault(u, []).append((qid, looks, aq))
    out = []
    for u in sorted(plan):
        sub = store.subset(idx, np.where(user_of == u)[0])
        mine = cache_dir / "probe51" / f"{u}.json"; mine.parent.mkdir(parents=True, exist_ok=True)
        J = CachedJudge(J0, mine)
        for f in sorted(cache_dir.glob(f"*/{u}.json")):
            if f != mine:
                try: J.cache = {**json.loads(f.read_text()), **J.cache}
                except json.JSONDecodeError: pass
        for qid, looks, aq in plan[u]:
            ar = run_album(sub, Album(name="probe", looks=looks, judge_question=aq), enc, J, None,
                           th=Thresholds(head_size=300, head_max=300, tail_budget=0))
            jd = ar.judged.sort_values("p_attr", ascending=False).drop_duplicates("item_id").head(5)
            top = [dict(id=str(x.item_id), p=round(float(x.p_attr), 3)) for _, x in jd.iterrows()]
            out.append(dict(anchor_of=qid, library=u, question=aq, n_judged=int(ar.judged.item_id.nunique()), top=top))
            print(u, qid, f"max {top[0]['p'] if top else None}", aq[:70], flush=True)
        J.save()
    Path("eval/complex_bench/eh_diag/probe_absent.json").write_text(json.dumps(out, indent=1))
    ps = [o["top"][0]["p"] for o in out if o["top"]]
    for fl in (0.2, 0.3, 0.4, 0.5):
        print(f"absent moments with best P >= {fl}: {sum(p >= fl for p in ps)}/{len(ps)}")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
