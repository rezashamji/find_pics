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
from route import TRIP, _TAIL, PlaceIndex, budget, place_candidates, remainder, route, split_parts  # noqa: E402

NAMES = ["Sam", "Alex"]
# RESULTS 59d variants (route.py header): R59d = R59c + a + b + c1 + c2; ablations = R59c + one item; aj59d = the
# ALL-JUDGE stack + a + b (every leaf judged on its whole scope)
DVARS = {"r59d": dict(a=True, b=True, c1=True, c2=True), "r59d_a": dict(a=True), "r59d_b": dict(b=True),
         "r59d_c": dict(c1=True, c2=True), "aj59d": dict(a=True, b=True, aj=True),
         # attribution only (added before any 59d number): aj0 = the all-judge stack through this harness (must equal the
         # reference), aj59c = all-judge + R59c's metadata-only trip route, ajb = all-judge + (b) parts only
         "aj0": dict(aj=True, notrip=True), "aj59c": dict(aj=True), "ajb": dict(b=True, aj=True, notrip=True)}
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
    if arg("users"):
        users = arg("users").split(",")
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
    IMG = b16_image_vectors(set(facts))
    TVS = {}

    def tv_for(user):
        """--tv may contain {fold}: TRAIN libraries are scored by the fold model that never trained on them."""
        from train import train_folds
        d = arg("tv").replace("{fold}", str(train_folds().get(user, -1)))
        if d not in TVS:
            TVS[d] = Scorer(d, facts=facts, img=IMG, qvec_fn=b16_text)
        return TVS[d]
    TV = None
    CE = Scorer(arg("ce"), facts=facts)
    PTV, PCE, JP = {}, {}, {}

    def cached(S, P, ids, qs):
        S = S if S is not None else CUR["tv"]
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
    CUR = {"tv": None}
    META = {}

    def meta_mask(idx_, question):
        """59c trip route: the first capitalised name that matches >= 1 photo's place in this library (engine.place_mask)."""
        key = (id(idx_), question)
        if key not in META:
            from findpics.engine import place_mask
            META[key] = None
            if "place" in idx_.items:
                for c in place_candidates(question):
                    mk = np.asarray(place_mask(idx_.items["place"], c), bool)
                    if mk.any():
                        META[key] = mk
                        break
        return META[key]
    engine.COMPOSE_QUESTIONS = False     # 59d product flag (main src): the harness composes its own way
    orig = engine._judge_rows
    # ---- RESULTS 59d (route.py header = the pre-registered rule) ----
    # D = flags of the variant being replayed: a (compound trip + tolerant places), b (OR / AND parts), c1 (budget
    # extension), c2 (exclusions judged), aj (all-judge: every leaf judged on its whole scope). R59c's own path is not used
    # by these variants; R59c itself runs unchanged.
    PIX, EXCL = {}, set()

    def pindex(idx_):
        it = idx_.items
        key = (id(idx_), len(it), str(it.item_id.iloc[0]) if len(it) else "", str(it.item_id.iloc[-1]) if len(it) else "")
        if key not in PIX:     # keyed by content too: a garbage-collected subset's id can be reused
            PIX[key] = PlaceIndex(it["place"]) if "place" in it else None
        return PIX[key]

    def decompose(idx_, q, crop, D, depth=0, meta_ok=True, excl=False):
        """-> ('meta', mask, child | None) | (op, [nodes]) | ('leaf', question, judge kind or '')."""
        qb = _TAIL.sub("", q)
        if meta_ok and TRIP.search(qb) and not D.get("notrip"):
            if D.get("a"):
                PI = pindex(idx_)
                for c in (place_candidates(q) if PI is not None else []):
                    mk = PI.mask(c)
                    if mk.any():
                        rem = remainder(q, c)
                        return ("meta", mk, decompose(idx_, rem, crop, D, depth + 1, False, excl) if rem else None)
            elif depth == 0:
                mk = meta_mask(idx_, q)     # R59c's trip route (engine.place_mask, metadata only)
                if mk is not None:
                    return ("meta", mk, None)
        if D.get("b") and depth < 4:
            sp = split_parts(q)
            if sp:
                return (sp[0], [decompose(idx_, x, crop, D, depth + 1, meta_ok and D.get("a", False), excl) for x in sp[1]])
        kind = route(q, crop)
        if not kind and TRIP.search(qb):
            kind = "trip-no-place"
        if not kind and excl:
            kind = "exclude"
        return ("leaf", q, kind)

    def leaves(node, rows):
        """(leaf question, kind, rows it is asked on) for every leaf of a tree."""
        if node[0] == "meta":
            return [] if node[2] is None else leaves(node[2], rows[node[1][rows]])
        if node[0] in ("or", "and"):
            return [x for ch in node[1] for x in leaves(ch, rows)]
        return [(node[1], node[2], rows)]

    def judge_ids(idx_, judge, q, crop, ids_rows_fr, batch):
        """judge (cached) the given (id, row, face row) triples for leaf question q; counted as judge calls."""
        cr = str(crop)
        need = [(i, r_, f) for i, r_, f in ids_rows_fr if (i, q, cr, int(f), False) not in JP]
        if need:
            p = orig(idx_, judge, np.array([r_ for _, r_, _ in need], int), np.array([f for _, _, f in need], int), q,
                     batch=batch, crop_person=crop)
            for (i, _, f), x in zip(need, p):
                JP[(i, q, cr, int(f), False)] = float(x)
        ST["judge"].update((i, q, cr) for i, _, _ in ids_rows_fr)

    def leaf_threshold(idx_, judge, q, crop, D, batch):
        """ms-score threshold above which photos of this leaf's scope are judged: the budget top M (R59c), and with c1
        one more block of M while the previous block was >= 50% judge matches (judged proactively over the pass-A scope)."""
        key = (q, str(crop))
        if key in ST["d_tj"]:
            return ST["d_tj"][key]
        sc = ST["d_scope"].get(key, {})
        if not sc:
            ST["d_tj"][key] = None     # no pass-A record: decided from the first call's rows (fallback)
            return None
        order = sorted(sc.items(), key=lambda kv: -kv[1][0])
        n, M = len(order), budget(len(order))
        F = min(n, M)
        if D.get("c1"):
            judge_ids(idx_, judge, q, crop, [(i, v[1], v[2]) for i, v in order[:F]], batch)
            while F < n:
                blk = order[F - M:F] if F >= M else order[:F]
                rate = np.mean([JP[(i, q, str(crop), int(v[2]), False)] >= .7 for i, v in blk])
                if rate < .5:
                    break
                F2 = min(n, F + M)
                judge_ids(idx_, judge, q, crop, [(i, v[1], v[2]) for i, v in order[F:F2]], batch)
                F = F2
            ST["d_ext"][key] = F
        tj = -np.inf if F >= n else order[F - 1][1][0]
        ST["d_tj"][key] = tj
        return tj

    def evaluate(idx_, judge, node, rows, fr, crop, D, batch):
        if not len(rows):
            return np.zeros(0, np.float32)
        if node[0] == "meta":
            mk = node[1][rows]
            if node[2] is None:
                ST["route"]["trip-metadata"] += len(rows)
                return np.where(mk, 0.99, 0.01).astype(np.float32)
            out = np.full(len(rows), 0.01, np.float32)
            if mk.any():
                out[mk] = evaluate(idx_, judge, node[2], rows[mk], fr[mk], crop, D, batch)
            ST["route"]["trip-metadata+rest"] += len(rows)
            return out
        if node[0] in ("or", "and"):
            P = np.stack([evaluate(idx_, judge, ch, rows, fr, crop, D, batch) for ch in node[1]])
            return (P.max(0) if node[0] == "or" else P.min(0)).astype(np.float32)
        q, kind = node[1], node[2]
        ids = idx_.items.item_id.iloc[rows].astype(str).tolist()
        ST["route"][(kind or "scorer") + ("" if q == ST["d_orig"] else " (part)")] += len(rows)
        if D.get("aj") or not kind:
            if D.get("aj"):
                judge_ids(idx_, judge, q, crop, list(zip(ids, rows, fr)), batch)
                return np.array([JP[(i, q, str(crop), int(f), False)] for i, f in zip(ids, fr)], np.float32)
            qs = scorer_question(q)
            pt = cached(TV, PTV, ids, qs)
            ST["tv"].update((i, qs) for i in ids)
            return pt
        qs = scorer_question(q)
        pt = cached(TV, PTV, ids, qs)
        tj = leaf_threshold(idx_, judge, q, crop, D, batch)
        if tj is None:     # fallback: budget over this call's rows
            tj = kth(pt, budget(len(pt)))
        keep = np.where(pt >= tj)[0]
        out = pt.copy()
        if len(keep):
            judge_ids(idx_, judge, q, crop, [(ids[j], rows[j], fr[j]) for j in keep], batch)
            out[keep] = [JP[(ids[j], q, str(crop), int(fr[j]), False)] for j in keep]
        return out.astype(np.float32)

    def record_d(idx_, rows, fr, question, crop):
        """pass A: the in-scope photos (ms P, row, face row) of every judge leaf any requested R59d variant will ask."""
        for D in ST["d_all"]:
            if D.get("aj"):
                continue
            node = decompose(idx_, question, crop, D, excl=D.get("c2", False) and question in EXCL)
            for q, kind, rr in leaves(node, rows):
                if not kind or not len(rr):
                    continue
                ids = idx_.items.item_id.iloc[rr].astype(str).tolist()
                pt = cached(TV, PTV, ids, scorer_question(q))
                fmap = dict(zip(rows.tolist(), fr.tolist()))
                ST["d_scope"][(q, str(crop))].update({i: (float(x), int(r_), int(fmap[int(r_)])) for i, x, r_ in zip(ids, pt, rr)})

    ST = {"mode": "A", "tK": {}, "scope": defaultdict(dict), "judge": set(), "ce": set(), "tv": set(),
          "route": Counter(), "unseen": 0}

    def judge_rows(idx_, judge, rows, face_rows, question, ref_img=None, batch=96, crop_person=False):
        rows = np.asarray(rows, int)
        if not len(rows):
            return np.zeros(0, np.float32)
        fr = np.asarray(face_rows, int)
        ids = idx_.items.item_id.iloc[rows].astype(str).tolist()
        cr = str(crop_person)
        if ref_img is None and ST["mode"] == "A" and ST.get("d_all"):
            record_d(idx_, rows, fr, question, crop_person)
        if ref_img is None and ST["mode"] == "B" and ST.get("D") is not None:
            D = ST["D"]
            ST["d_orig"] = question
            node = decompose(idx_, question, crop_person, D, excl=D.get("c2", False) and question in EXCL)
            return evaluate(idx_, judge, node, rows, fr, crop_person, D, batch)
        r = route(question, crop_person)
        if ST["mode"] == "B" and ST.get("trip") and TRIP.search(_TAIL.sub("", question)):
            mk = meta_mask(idx_, question)
            if mk is not None:     # answered from place metadata, no judge, no scorer
                ST["route"]["trip-metadata"] += len(ids)
                return np.where(mk[rows], 0.99, 0.01).astype(np.float32)
            r = r or "trip-no-place"
        ST["route"][r or "scorer"] += len(ids)
        if r and ST.get("jM") is not None or (r and ST["mode"] == "A" and ST.get("record_j")):
            # post-hoc variants jM (10-14, after the pre-registered ones were scored): the judge sees only the top M of
            # a pixel-kind question's in-scope photos by towervec P; the others keep towervec P
            qs_ = scorer_question(question)
            ptj = cached(TV, PTV, ids, qs_)
            if ST["mode"] == "A":
                ST["scope_j"][(question, cr)].update(zip(ids, ptj))
            elif ST.get("jM") is not None:
                if ST["jM"] == "budget":     # 59c: top max(50, 5% of scope) by the millisecond score
                    sc = ST["scope_j"].get((question, cr)) or ST["scope"].get((question, cr))
                    tj = kth(sc.values(), budget(len(sc))) if sc else -np.inf
                else:
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
            f = cache_dir / arg("save_sub", "r59b_cascade") / f"{BEFORE['__user__']}.json"     # 59d: one subdir per run
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
            CUR["tv"] = tv_for(u)
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
        ST["record_j"] = any(v.startswith("j") or v == "r59c" for v in variants)
        ST["trip"] = False
        ST["D"] = None; ST["d_scope"] = defaultdict(dict); ST["d_all"] = [DVARS[v] for v in variants if v in DVARS]
        ST["d_ext"] = {}
        EXCL.clear()
        sp_ = stored.get(q["qid"]) or {}
        for t_ in sp_.get("turns") or []:
            for a_ in t_["plan"]["albums"]:
                if a_.get("exclude_question"):
                    EXCL.add(a_["exclude_question"])
        reset("A")
        recA = run_once()
        stA = snap()
        for v in variants:
            if v == "k0":
                rec, st = recA, stA
            else:
                if v in DVARS:      # 59d variants: K = 0 + the R59d tree (route.py header)
                    Kv, M = 0, None
                elif v == "r59c":     # 59c: K = 0 (no facts reader) + trip route + judge budget max(50, 5% of scope)
                    Kv, M = 0, "budget"
                    ST["trip"] = True
                elif v.startswith("j"):     # jM: facts reader on all non-pixel photos (K = all) + judge on top M of pixel kinds
                    Kv, M = None, int(v[1:])
                    ST["tJ"] = {key: kth(d.values(), M) for key, d in ST["scope_j"].items()}
                else:
                    Kv, M = (None if v == "kall" else int(v[1:]) if v.startswith("k") else
                             math.ceil(int(v[1:]) * n_lib / N_PHONE)), None
                ST["tK"] = {key: (-np.inf if Kv is None else kth(d.values(), Kv)) for key, d in ST["scope"].items()}
                reset("B"); ST["jM"] = M
                if v in DVARS:
                    ST["D"] = DVARS[v]; ST["d_tj"] = {}
                rec = run_once(); st = snap(); st["K"] = Kv; st["M"] = M
                if v in DVARS:
                    st["ext"] = {f"{k_[0]}|{k_[1]}": x for k_, x in ST["d_ext"].items()}
                    ST["d_ext"] = {}
                ST["jM"] = None; ST["trip"] = False; ST["D"] = None
            outs[v].append(rec); stats[v][q["qid"]] = st
            (out_dir / v / f"part{k}.json").write_text(json.dumps(outs[v]))
            (out_dir / v / f"stats{k}.json").write_text(json.dumps(stats[v]))
        print(q["qid"], q["cat"], "err " + recA["error"] if recA["error"] else "ok",
              {v: stats[v][q["qid"]]["judge"] for v in variants}, f"misses {stA['misses']}", flush=True)
    save_new()
    print("CASCADE_DONE", flush=True)


if __name__ == "__main__":
    main()
