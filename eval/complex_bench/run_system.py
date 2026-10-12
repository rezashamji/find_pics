"""Complex-query benchmark, step 5: the CURRENT product (chat planner + streaming engine, 9B judge) on a split.

Same path as eval/eval_disbench.py unified / eval/eval_everyday.py: per query, library = that user's photos only;
plan_turn on the owner's words (follow-up turns get the history + current plan, as the chat does); known people =
the face groups the benchmark names (Sam, Alex), their references = the group's faces + query-time expansion (the
product's "named face group" path); "me"/unknown people -> no identity (Flickr libraries have no owner tags).
Records the FIRST answer (fast) and the FINAL round (judge has looked at every in-scope photo).
Usage (vLLM env, GPU): python eval/complex_bench/run_system.py --split=dev --shard=k/K
  -> eval/complex_bench/runs/<split>/part<k>.json
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
    split = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--split="))
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, K = map(int, shard.split("/"))
    bench = json.loads((ROOT / "queries" / f"{split}.json").read_text())
    Q = bench["queries"][k::K]
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    J = VLLMJudge(gpu_mem=0.7)
    out_dir = ROOT / "runs" / split; out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"part{k}.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else []
    done = {r["qid"] for r in out}
    subs = {}
    for q in Q:
        if q["qid"] in done:
            continue
        u = q["user"]
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
            for msg in q["turns"]:
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
        print(q["qid"], q["cat"], "err" if err else "ok", f"{time.time() - t0:.0f}s", q["turns"][-1][:80], flush=True)


if __name__ == "__main__":
    main()
