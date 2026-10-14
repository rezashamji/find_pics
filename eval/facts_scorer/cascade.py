"""RESULTS 59b: the CASCADE end to end on the complex bench (same search code path and stored plans as replay.py).
Every engine judge call (engine._judge_rows) is routed by route.py from its question text:
  pixel kinds -> the 9B judge (answers from the shared cache; a 9B is loaded lazily only for a cache miss)
  the rest    -> towervec (ms scorer) P on every in-scope photo; the facts reader (cross-encoder) P replaces it on the
                 photos whose towervec P >= the K-th largest among that question's in-scope photos.
Per request: pass A = towervec only (+ judge kinds) = variant k0, which also records each question's in-scope set;
then one pass B per variant with that variant's K. Variants: k0, k500, kall, f500, f2000, f5000 (f = K scaled to the
library: ceil(K x N_lib / 187,084)).
  python eval/facts_scorer/cascade.py --split=dev --shard=k/K --plans=<plans.json> --cache=<jcache dir>
      --tv=<towervec model dir> --ce=<cross-encoder model dir> --facts=<facts parquet> --out=<dir> [--variants=...]
  -> <out>/<variant>/part<k>.json (run_system format; score.py) and <out>/<variant>/stats<k>.json (per request:
     judge pairs, facts-reader pairs, towervec pairs, judge misses, library size, routing counts)
Run with PYTHONPATH=<frozen src>, vLLM env, GPU node (scorers on GPU; the 9B only if the cache misses).
"""
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval/complex_bench")
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402
from data import scorer_question  # noqa: E402
from route import route  # noqa: E402

NAMES = ["Sam", "Alex"]
N_PHONE = 187084


def arg(key, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), d)


def main():
    from findpics import agent, converse, engine, store
    from findpics.converse import CachedJudge, Plan, stream_plan
    from findpics.engine import Thresholds
    from findpics.models import ImageTextEncoder
    from findpics.people import expand_refs
    from scorers import Scorer
    split, (k, K) = arg("split", "dev"), map(int, arg("shard", "0/1").split("/"))
    variants = arg("variants", "k0,k500,kall,f500,f2000,f5000").split(",")
    bench = json.loads((ROOT / "queries" / f"{split}.json").read_text())
    users = sorted({q["user"] for q in bench["queries"]})[k::K]
    Q = [q for q in bench["queries"] if q["user"] in users]
    if arg("qids"):
        want = set(arg("qids").split(",")); Q = [q for q in Q if q["qid"] in want]
    stored = json.loads(Path(arg("plans")).read_text())["plans"]
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    enc = ImageTextEncoder(idx.clip_model)
    out_dir = Path(arg("out")); cache_dir = Path(arg("cache"))
    FT = pd.read_parquet(arg("facts"))
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    from train import b16_image_vectors, b16_text
    TV = Scorer(arg("tv"), facts=facts, img=b16_image_vectors(set(facts)), qvec_fn=b16_text)
    CE = Scorer(arg("ce"), facts=facts)
    PTV, PCE, JP = {}, {}, {}

    def cached(S, P, ids, qs):
        miss = [i for i in dict.fromkeys(ids) if (i, qs) not in P]
        if miss:
            for i, p in zip(miss, S.p(miss, [qs] * len(miss))):
                P[(i, qs)] = float(p)
        return np.array([P[(i, qs)] for i in ids], np.float32)

    class LazyJudge:
        """The 9B, loaded only when the answer cache misses (counted)."""
        def __init__(self):
            self.j = None; self.misses = 0

        def p_yes(self, images, question):
            self.misses += len(images)
            if self.j is None:
                from findpics.vlm import VLLMJudge
                self.j = VLLMJudge(gpu_mem=0.6)
            return self.j.p_yes(images, question)

        def text(self, prompt, **kw):
            raise RuntimeError("no planning here")

    LJ = LazyJudge()
    orig = engine._judge_rows
    ST = {"mode": "A", "tK": {}, "scope": defaultdict(dict), "judge": set(), "ce": set(), "tv": set(),
          "route": Counter(), "unseen": 0}

    def judge_rows(idx_, judge, rows, face_rows, question, ref_img=None, batch=96, crop_person=False):
        rows = np.asarray(rows, int)
        if not len(rows):
            return np.zeros(0, np.float32)
        fr = np.asarray(face_rows, int)
        ids = idx_.items.item_id.iloc[rows].astype(str).tolist()
        cr = str(crop_person)
        r = route(question, crop_person)
        ST["route"][r or "scorer"] += len(ids)
        if r and ST.get("jM") is not None or (r and ST["mode"] == "A" and ST.get("record_j")):
            # post-hoc variants jM (10-14, after the pre-registered ones were scored): the judge sees only the top M of
            # a pixel-kind question's in-scope photos by towervec P; the others keep towervec P
            qs_ = scorer_question(question)
            ptj = cached(TV, PTV, ids, qs_)
            if ST["mode"] == "A":
                ST["scope_j"][(question, cr)].update(zip(ids, ptj))
            elif ST.get("jM") is not None:
                tj = ST["tJ"].get((question, cr), -np.inf)
                keep = np.where(ptj >= tj)[0]
                outj = ptj.copy()
                if len(keep):
                    sub_rows, sub_fr, sub_ids = rows[keep], fr[keep], [ids[j] for j in keep]
                    need = [j for j, (i, f) in enumerate(zip(sub_ids, sub_fr)) if (i, question, cr, int(f), ref_img is not None) not in JP]
                    if need:
                        p = orig(idx_, judge, sub_rows[need], sub_fr[need], question, ref_img=ref_img, batch=batch, crop_person=crop_person)
                        for j, x in zip(need, p):
                            JP[(sub_ids[j], question, cr, int(sub_fr[j]), ref_img is not None)] = float(x)
                    ST["judge"].update((i, question, cr) for i in sub_ids)
                    outj[keep] = [JP[(i, question, cr, int(f), ref_img is not None)] for i, f in zip(sub_ids, sub_fr)]
                return outj.astype(np.float32)
        if r:
            need = [j for j, (i, f) in enumerate(zip(ids, fr)) if (i, question, cr, int(f), ref_img is not None) not in JP]
            if need:
                p = orig(idx_, judge, rows[need], fr[need], question, ref_img=ref_img, batch=batch, crop_person=crop_person)
                for j, x in zip(need, p):
                    JP[(ids[j], question, cr, int(fr[j]), ref_img is not None)] = float(x)
            ST["judge"].update((i, question, cr) for i in ids)
            return np.array([JP[(i, question, cr, int(f), ref_img is not None)] for i, f in zip(ids, fr)], np.float32)
        qs = scorer_question(question)
        pt = cached(TV, PTV, ids, qs)
        ST["tv"].update((i, qs) for i in ids)
        key = (question, cr)
        if ST["mode"] == "A":
            ST["scope"][key].update(zip(ids, pt))
            return pt
        if key not in ST["tK"]:
            ST["unseen"] += 1
        t = ST["tK"].get(key, -np.inf)
        m = pt >= t
        out = pt.copy()
        if m.any():
            sel = [ids[j] for j in np.where(m)[0]]
            out[m] = cached(CE, PCE, sel, qs)
            ST["ce"].update((i, qs) for i in sel)
        return out
    engine._judge_rows = converse._judge_rows = agent._judge_rows = judge_rows

    def kth(vals, Kv):
        v = np.sort(np.fromiter(vals, float))[::-1]
        return -np.inf if Kv >= len(v) else v[Kv - 1] if Kv > 0 else np.inf

    outs = {v: [] for v in variants}
    stats = {v: {} for v in variants}
    for v in variants:
        (out_dir / v).mkdir(parents=True, exist_ok=True)
        f = out_dir / v / f"part{k}.json"
        if f.exists():
            outs[v] = json.loads(f.read_text())
        f = out_dir / v / f"stats{k}.json"
        if f.exists():
            stats[v] = json.loads(f.read_text())
    done = set.intersection(*[{r["qid"] for r in outs[v]} for v in variants]) if variants else set()
    J, cur_user, subs = None, None, {}
    BEFORE = {}

    def save_new():
        """Only the answers the lazy 9B added (cache misses) go to <cache>/r59b_cascade/<user>.json."""
        if J is None or not BEFORE:
            return
        new = {k_: v_ for k_, v_ in J.cache.items() if k_ not in BEFORE}
        if new:
            f = cache_dir / "r59b_cascade" / f"{BEFORE['__user__']}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            old = json.loads(f.read_text()) if f.exists() else {}
            f.write_text(json.dumps({**old, **new}))
    for q in Q:
        if q["qid"] in done:
            continue
        u = q["user"]
        if u != cur_user:
            save_new()
            J = CachedJudge(LJ, None)
            for f in sorted(cache_dir.glob(f"*/{u}.json")):
                try:
                    J.cache = {**json.loads(f.read_text()), **J.cache}
                except json.JSONDecodeError:
                    pass
            BEFORE.clear(); BEFORE.update(J.cache); BEFORE["__user__"] = u
            cur_user = u
            JP.clear(); PTV.clear(); PCE.clear()
        if u not in subs:
            subs.clear(); subs[u] = store.subset(idx, np.where(user_of == u)[0])
        sub = subs[u]
        n_lib = int(sub.n_items) if hasattr(sub, "n_items") else len(sub.items)
        lib = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        people = NAMES[:len(G)]

        def refs_for(person, G=G, sub=sub):
            for nm, g in zip(NAMES, G):
                if person and person.lower() == nm.lower():
                    refs = expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3)
                    return nm, refs, int(g["rep"]), len(g["item_ids"])
            raise ValueError(f"unknown person {person}")

        def run_once():
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
                                           albums=[sorted(r.returned.item_id.astype(str)) for r in rs]))
                    turns.append(dict(message=msg, plan=json.loads(P.model_dump_json()), first=rounds[0] if rounds else None,
                                      final=rounds[-1] if rounds else None, n_rounds=len(rounds)))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:400]
            return dict(qid=q["qid"], user=u, cat=q["cat"], turns=turns, error=err, seconds=round(time.time() - t0, 1))

        def reset(mode):
            ST.update(mode=mode, judge=set(), ce=set(), tv=set(), route=Counter(), unseen=0)
            LJ.misses = 0

        def snap():
            return dict(judge=len(ST["judge"]), ce=len(ST["ce"]), tv=len(ST["tv"]), misses=LJ.misses, n_lib=n_lib,
                        route=dict(ST["route"]), unseen=ST["unseen"])
        ST["scope"] = defaultdict(dict); ST["tK"] = {}; ST["scope_j"] = defaultdict(dict); ST["tJ"] = {}; ST["jM"] = None
        ST["record_j"] = any(v.startswith("j") for v in variants)
        reset("A")
        recA = run_once()
        stA = snap()
        for v in variants:
            if v == "k0":
                rec, st = recA, stA
            else:
                if v.startswith("j"):     # jM: facts reader on all non-pixel photos (K = all) + judge on top M of pixel kinds
                    Kv, M = None, int(v[1:])
                    ST["tJ"] = {key: kth(d.values(), M) for key, d in ST["scope_j"].items()}
                else:
                    Kv, M = (None if v == "kall" else int(v[1:]) if v.startswith("k") else
                             math.ceil(int(v[1:]) * n_lib / N_PHONE)), None
                ST["tK"] = {key: (-np.inf if Kv is None else kth(d.values(), Kv)) for key, d in ST["scope"].items()}
                reset("B"); ST["jM"] = M
                rec = run_once(); st = snap(); st["K"] = Kv; st["M"] = M
                ST["jM"] = None
            outs[v].append(rec); stats[v][q["qid"]] = st
            (out_dir / v / f"part{k}.json").write_text(json.dumps(outs[v]))
            (out_dir / v / f"stats{k}.json").write_text(json.dumps(stats[v]))
        print(q["qid"], q["cat"], "err " + recA["error"] if recA["error"] else "ok",
              {v: stats[v][q["qid"]]["judge"] for v in variants}, f"misses {stA['misses']}", flush=True)
    save_new()
    print("CASCADE_DONE", flush=True)


if __name__ == "__main__":
    main()
