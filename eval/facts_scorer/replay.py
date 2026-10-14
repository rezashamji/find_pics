"""RESULTS 59, end-to-end: the complex bench search stage (eval/complex_bench/run_system.py, same code path, stored
plans) with EVERY judge call (engine._judge_rows, the one function all judge questions go through) answered by a
query-time scorer from stored per-photo data - no pixels, no VLM. Plan stage unchanged (--plans).

  --scorer=<model dir>        a trained/zero-shot scorer (eval/facts_scorer/models/<name>; kinds in scorers.py)
  --scorer=passthrough        the real judge path (needs --cache: the 9B answer cache) + a log of every judged pair:
                              reproduces the reference run exactly and records (qid, item, question, P_9B)
  --facts=<facts parquet>     facts text the scorer reads (default 27B facts)
  --split=dev --by-user --shard=k/K --plans=<plans.json> --out=<dir> [--cache=<dir>]
  -> <out>/part<k>.json (run_system format, scored by eval/complex_bench/score.py), <out>/pairs<k>.parquet
Run with PYTHONPATH=.cache/wt_m54_s1/src (the shipped code), vLLM env.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/complex_bench")
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402
from data import scorer_question  # noqa: E402

NAMES = ["Sam", "Alex"]


def arg(key, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), d)


def main():
    from findpics import agent, converse, engine, store
    from findpics.converse import CachedJudge, Plan, stream_plan
    from findpics.engine import Thresholds
    from findpics.models import ImageTextEncoder
    from findpics.people import expand_refs
    split, (k, K) = arg("split", "dev"), map(int, arg("shard", "0/1").split("/"))
    mode = arg("scorer")
    bench = json.loads((ROOT / "queries" / f"{split}.json").read_text())
    users = sorted({q["user"] for q in bench["queries"]})[k::K]
    Q = [q for q in bench["queries"] if q["user"] in users]
    if arg("qids"):
        want = set(arg("qids").split(","))
        Q = [q for q in Q if q["qid"] in want]
    stored = json.loads(Path(arg("plans")).read_text())["plans"]
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    out_dir = Path(arg("out")); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(arg("cache")) if arg("cache") else None

    class CacheOnly:
        def p_yes(self, images, question):
            raise RuntimeError(f"judge cache miss ({len(images)} images): {question[:80]}")

        def text(self, prompt, **kw):
            raise RuntimeError("no planning here: pass --plans")

    S = None
    if mode != "passthrough":
        from scorers import Scorer
        FT = pd.read_parquet(arg("facts", "eval/facts_scorer/data/facts_27b.parquet"))
        facts = dict(zip(FT.item_id.astype(str), FT.text))
        cfg = json.loads((Path(mode) / "config.json").read_text())
        img = qfn = None
        if cfg["kind"] in ("vec", "towervec", "ens"):
            from train import b16_image_vectors, b16_text
            img = b16_image_vectors(set(facts)); qfn = b16_text
        if cfg["kind"].startswith("zeroshot"):
            raise SystemExit("zero-shot baselines are scored per photo only (eval_photo.py)")
        S = Scorer(mode, facts=facts, img=img, qvec_fn=qfn)
    orig = engine._judge_rows
    LOG, CUR = [], {"qid": None}

    def judge_rows(idx_, judge, rows, face_rows, question, ref_img=None, batch=96, crop_person=False):
        rows = np.asarray(rows, int)
        if not len(rows):
            return np.zeros(0, np.float32)
        ids = idx_.items.item_id.iloc[rows].astype(str).tolist()
        if S is None:
            p = orig(idx_, judge, rows, face_rows, question, ref_img=ref_img, batch=batch, crop_person=crop_person)
        else:
            p = S.p(ids, [scorer_question(question)] * len(ids))
        p = np.asarray(p, np.float32)
        LOG.extend((CUR["qid"], i, question, str(crop_person), ref_img is not None, float(x)) for i, x in zip(ids, p))
        return p
    engine._judge_rows = converse._judge_rows = agent._judge_rows = judge_rows
    J0 = CacheOnly()
    J, cur_user = J0, None
    out_f = out_dir / f"part{k}.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else []
    done = {r["qid"] for r in out}
    subs = {}
    for q in Q:
        if q["qid"] in done:
            continue
        u = q["user"]
        CUR["qid"] = q["qid"]
        if cache_dir is not None and u != cur_user:
            J = CachedJudge(J0, None)
            for f in sorted(cache_dir.glob(f"*/{u}.json")):
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
        t0 = time.time(); err = None; turns = []
        try:
            sp = stored.get(q["qid"])
            if sp is None or sp["error"] or len(sp["turns"]) < len(q["turns"]):
                raise ValueError(f"no stored plan: {(sp or {}).get('error')}")
            for ti, msg in enumerate(q["turns"]):
                P = Plan.model_validate(sp["turns"][ti]["plan"])
                for a in P.albums:
                    if a.person and a.person not in people:
                        a.person = None
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
        print(q["qid"], q["cat"], "err " + err if err else "ok", f"{time.time() - t0:.1f}s", q["turns"][-1][:70], flush=True)
    pd.DataFrame(LOG, columns=["qid", "item_id", "question", "crop", "ref_img", "p"]).to_parquet(out_dir / f"pairs{k}.parquet")
    print("REPLAY_DONE", len(LOG), "judged pairs", flush=True)


if __name__ == "__main__":
    main()
