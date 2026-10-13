"""Complex-query benchmark, step 5: the CURRENT product (chat planner + streaming engine, 9B judge) on a split.

Same path as eval/eval_disbench.py unified / eval/eval_everyday.py: per query, library = that user's photos only;
plan_turn on the owner's words (follow-up turns get the history + current plan, as the chat does); known people =
the face groups the benchmark names (Sam, Alex), their references = the group's faces + query-time expansion (the
product's "named face group" path); "me"/unknown people -> no identity (Flickr libraries have no owner tags).
Records the FIRST answer (fast) and the FINAL round (judge has looked at every in-scope photo).
Usage (vLLM env, GPU): python eval/complex_bench/run_system.py --split=dev --shard=k/K
  -> eval/complex_bench/runs/<split>/part<k>.json
Options (10-12, planner comparisons): --plans=<plans.json from plan_system.py> uses those plans instead of planning here
(same judge stage for every planner); --by-user shards by library instead of by query; --out=<dir> output folder;
--cache=<dir> caches the judge's answers per library (key = image pixels + question, as converse.CachedJudge) in
<dir>/<run name>/<user>.json and reads every other run's file for that library first, so runs whose plans ask the same
questions get the same answers (differences between runs then come from the plans alone).
--judge=cache (10-12, RESULTS 49c): no model at all; every judge call must be answered by the --cache files (a miss is an
error for that request), so a CODE change can be replayed on a run's own judge answers (CPU only). --qids=Q1,Q2 or
--qids=@file (one per line): only these requests.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402

NAMES = ["Sam", "Alex"]


def main():
    from findpics import store
    from findpics.converse import plan_turn, stream_plan
    from findpics.engine import Thresholds
    from findpics.models import ImageTextEncoder
    from findpics.people import expand_refs
    from findpics.vlm import VLLMJudge
    from findpics.converse import CachedJudge, Plan
    arg = lambda key, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), d)
    split = arg("split")
    shard = arg("shard", "0/1")
    k, K = map(int, shard.split("/"))
    bench = json.loads((ROOT / "queries" / f"{split}.json").read_text())
    if "--by-user" in sys.argv:
        users = sorted({q["user"] for q in bench["queries"]})[k::K]
        Q = [q for q in bench["queries"] if q["user"] in users]
    else:
        Q = bench["queries"][k::K]
    if arg("qids"):
        want = arg("qids")
        want = set(Path(want[1:]).read_text().split()) if want.startswith("@") else set(want.split(","))
        Q = [q for q in Q if q["qid"] in want]
    stored = json.loads(Path(arg("plans")).read_text())["plans"] if arg("plans") else None
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    cache_only = arg("judge") == "cache"

    class CacheOnly:
        """--judge=cache: answers come only from the cache files; a miss raises."""
        def p_yes(self, images, question):
            raise RuntimeError(f"judge cache miss ({len(images)} images): {question[:80]}")

        def text(self, prompt, **kw):
            raise RuntimeError("--judge=cache cannot plan: pass --plans")
    J0 = CacheOnly() if cache_only else VLLMJudge(gpu_mem=0.7)
    out_dir = Path(arg("out")) if arg("out") else ROOT / "runs" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(arg("cache")) if arg("cache") else None
    J, cur_user = J0, None
    out_f = out_dir / f"part{k}.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else []
    done = {r["qid"] for r in out}
    subs = {}
    for q in Q:
        if q["qid"] in done:
            continue
        u = q["user"]
        if cache_dir is not None and u != cur_user:
            if isinstance(J, CachedJudge):
                J.save()
            mine = cache_dir / out_dir.name / f"{u}.json"
            if not cache_only:
                mine.parent.mkdir(parents=True, exist_ok=True)
            J = CachedJudge(J0, None if cache_only else mine)
            for f in sorted(cache_dir.glob(f"*/{u}.json")):     # other runs' answers for this library (read-only)
                if f != mine:
                    try:
                        J.cache = {**json.loads(f.read_text()), **J.cache}
                    except json.JSONDecodeError:
                        pass
            cur_user = u
        if u not in subs:
            subs.clear()
            subs[u] = store.subset(idx, np.where(user_of == u)[0])
        sub = subs[u]
        lib = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        people = NAMES[:len(G)]

        def refs_for(person, G=G, sub=sub):
            for nm, g in zip(NAMES, G):
                if person and person.lower() == nm.lower():
                    refs = expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3)
                    return nm, refs, int(g["rep"]), len(g["item_ids"])
            raise ValueError(f"unknown person {person}")
        today = date.fromisoformat(q["today"])
        history, P, turns = [], None, []
        t0 = time.time(); err = None
        try:
            sp = stored.get(q["qid"]) if stored is not None else None
            if stored is not None and (sp is None or sp["error"] or len(sp["turns"]) < len(q["turns"])):
                raise ValueError(f"no stored plan: {(sp or {}).get('error')}")
            for ti, msg in enumerate(q["turns"]):
                if stored is not None:
                    P = Plan.model_validate(sp["turns"][ti]["plan"])
                else:
                    P = plan_turn(msg, J.text, history=history, current=P, people=people, today=today)
                history.append(msg)
                for a in P.albums:
                    if a.person and a.person not in people:
                        a.person = None          # "me"/unknown: no identity tags on Flickr
                    a.with_people = [w for w in a.with_people if w in people]
                rounds = []
                for rs in stream_plan(sub, P, enc, J, refs_for, th=Thresholds(tail_budget=300, stream=True)):
                    rounds.append(dict(t=round(time.time() - t0, 1),
                                       albums=[sorted(r.returned.item_id.astype(str)) for r in rs],
                                       traces=[getattr(r, "trace", {}) for r in rs]))
                turns.append(dict(message=msg, plan=json.loads(P.model_dump_json()), first=rounds[0] if rounds else None,
                                  final=rounds[-1] if rounds else None, n_rounds=len(rounds)))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:400]
        out.append(dict(qid=q["qid"], user=u, cat=q["cat"], turns=turns, error=err, seconds=round(time.time() - t0, 1)))
        out_f.write_text(json.dumps(out))
        if isinstance(J, CachedJudge):
            J.save()
        print(q["qid"], q["cat"], "err" if err else "ok", f"{time.time() - t0:.0f}s", q["turns"][-1][:80], flush=True)


if __name__ == "__main__":
    main()
