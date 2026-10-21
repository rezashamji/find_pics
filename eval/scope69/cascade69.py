"""RESULTS 69: the R59d cascade (100% facts, policy V) with a PHONE-FEASIBLE leaf-scope source (pre-registered; JOURNAL
10-16 "RESULTS 69 PRE-REGISTRATION"). = eval/cascade_port/record65.py (cascade65 + R68 logging) with configs
"<scope source>_<score precision>":
  AJ   = the measured harness: pass A answers judge-kind questions with the judge on EVERY photo it is asked about (cached
         in the benchmark, unaffordable on a phone); its leaf scopes feed pass B (control; must reproduce RESULTS 65 100_N_A)
  DRY  = pass A is a DRY RUN of the same plan in which every leaf is answered by the millisecond scorer (the question tree
         with trip metadata; judge-kind leaves get the millisecond P): 0 judge calls; its leaf scopes feed pass B
  ON   = no pass A: one online pass; a judge leaf's scope = every photo the engine has asked that leaf about SO FAR in this
         search (cumulative); its budget, extension and threshold are re-derived at every call over that scope
  bf16 = the scorers as every measured number ran (bf16 autocast on GPU); fp32 = autocast off, TF32 off (the Swift port)
  Every photo has its facts line (100%), so policy V = A here. Scope entries keep the index they were asked in, so a
  proactive judge call reads the photo's own row and face row (the 65 harness used pass-A rows on the current index).
  The recording (--rec) also logs pass-A outputs (needed for the DRY parity test).
Original docstring:
RESULTS 68 RECORDER = eval/partial65/cascade65.py UNCHANGED in behaviour + logging hooks (marked "R68 LOG"): for every
request and config it writes what the cascade saw and decided, so FindPicsCore CascadeRoute.swift can be checked against
the Python harness call by call (pass A calls -> leaf scopes; pass B calls -> output P per photo; scorer P per (photo,
scorer question) for towervec and vecB; every judge answer read; the judged (photo, question) set; budget extensions).
  python eval/cascade_port/record65.py <cascade65 args> --rec=<dir>  -> <dir>/<config>/rec<k>.jsonl.gz + lib<k>.json
Original docstring:
RESULTS 65: the R59d cascade while only PART of each library has a facts line (pre-registered; JOURNAL 10-16
"RESULTS 65 PRE-REGISTRATION"). A copy of eval/facts_scorer/cascade.py (59d harness, variant r59d only, same routing,
budget, extension, judge cache + lazy 9B), with the millisecond score of each photo chosen by a CONFIG:
  config = "<coverage %>_<order>_<policy>": the photos of each library with rank_<order> < ceil(cov x n) (orders.py)
  have their facts line; for the others the POLICY decides the millisecond P:
    A  = current code: towervec with the Scorer's default facts text "no description" (+ the photo's image vector)
    V  = the image-vector-only scorer vecB (RESULTS 59c; TRAIN fold models / full for dev), same ranking and budget
    VS = V + SEPARATE judge budgets: each judge leaf's top max(50, 5%) (+ the c1 extension) among its in-scope photos
         WITH facts by towervec, and separately among those WITHOUT facts by vecB
  Judge answers do not depend on facts, so they are shared across configs within a library (JP); counts are per config.
  python eval/partial65/cascade65.py --split=train --shard=k/K --plans=... --cache=... --tv=<dir {fold}>
      --vec=<dir {fold}> --ce=<dir> --facts=<parquet> --orders=eval/partial65/data/orders.parquet
      --configs=100_N_A,0_N_A,... --out=<dir>   -> <out>/<config>/part<k>.json + stats<k>.json
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facts_scorer"))     # before eval/partial65 (its data/ dir shadows data.py)
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


import contextlib  # noqa: E402


@contextlib.contextmanager
def no_amp():
    """fp32 scorer calls: torch.autocast becomes a no-op and float32 matmuls run at full precision (no TF32)."""
    import torch
    old = torch.autocast, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()
    torch.autocast = lambda *a, **k: contextlib.nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.autocast = old[0]; torch.backends.cuda.matmul.allow_tf32 = old[1]; torch.backends.cudnn.allow_tf32 = old[2]
        torch.set_float32_matmul_precision(old[3])


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
    # R70: a config token "F<tag>" (e.g. DRY_fp32_Fk32) scores with the facts parquet --fvar with {tag} replaced (same
    # towervec model, lines re-encoded); no F token = --facts (R69/69b behaviour unchanged)
    FACTS = {"": facts}

    def facts_of(tag):
        if tag not in FACTS:
            T_ = pd.read_parquet(arg("fvar").replace("{tag}", tag))
            FACTS[tag] = dict(zip(T_.item_id.astype(str), T_.text))
            assert set(FACTS[tag]) == set(facts), f"facts variant {tag}: different photo set"
        return FACTS[tag]
    from train import b16_image_vectors, b16_text
    IMG = b16_image_vectors(set(facts))
    TVS = {}

    def tv_for(user):
        """--tv may contain {fold}: TRAIN libraries are scored by the fold model that never trained on them."""
        from train import train_folds
        d = arg("tv").replace("{fold}", str(train_folds().get(user, -1)))
        kk = (d, CUR["prec"], CUR["ft"])     # R69: one instance per precision; R70: and per facts variant
        if kk not in TVS:
            TVS[kk] = Scorer(d, facts=facts_of(CUR["ft"]), img=IMG, qvec_fn=b16_text)
            TVS[kk].fp32 = CUR["prec"] == "fp32"
        return TVS[kk]
    TV = None
    CE = Scorer(arg("ce"), facts=facts)
    # ---- R68 LOG ----
    LOG = {"on": False, "jp": set(), "tv": set(), "vec": set(), "nof": set()}

    class LogDict(dict):
        def __getitem__(self, key):
            if LOG["on"]:
                LOG["jp"].add(key)
            return dict.__getitem__(self, key)
    PTV, PCE, JP = {}, {}, LogDict()
    NOF, VECS = {}, {}

    def side_for(user):
        """(towervec with NO facts text for any photo = policy A's fallback, vecB) for this library's fold."""
        from train import train_folds
        f = str(train_folds().get(user, -1))
        dt, dv = arg("tv").replace("{fold}", f), arg("vec").replace("{fold}", f)
        pr = CUR["prec"]
        if (dt, pr) not in NOF:
            NOF[(dt, pr)] = Scorer(dt, facts={}, img=IMG, qvec_fn=b16_text); NOF[(dt, pr)].fp32 = pr == "fp32"
        if (dv, pr) not in VECS:
            VECS[(dv, pr)] = Scorer(dv, img=IMG, qvec_fn=b16_text); VECS[(dv, pr)].fp32 = pr == "fp32"
        return NOF[(dt, pr)], VECS[(dv, pr)]
    PFS, PNS, PVS = defaultdict(dict), {"bf16": {}, "fp32": {}}, {"bf16": {}, "fp32": {}}     # R70: PFS keyed (prec, facts tag)

    def _p(S, P, ids, qs):
        tag = "tv" if P is PFS[(CUR["prec"], CUR["ft"])] else "vec" if P is PVS[CUR["prec"]] else "nof" if P is PNS[CUR["prec"]] else None     # R68 LOG
        if tag:
            LOG[tag].update((i, qs) for i in ids)
        miss = [i for i in dict.fromkeys(ids) if (i, qs) not in P]
        if miss:
            with (no_amp() if getattr(S, "fp32", False) else contextlib.nullcontext()):
                pm = S.p(miss, [qs] * len(miss))
            for i, p in zip(miss, pm):
                P[(i, qs)] = float(p)
        return np.array([P[(i, qs)] for i in ids], np.float32)

    def cached(S, P, ids, qs):
        if S is not None:
            return _p(S, P, ids, qs)
        pr = CUR["prec"]
        out = _p(CUR["tv"], PFS[(pr, CUR["ft"])], ids, qs)          # towervec on the facts line (photos WITH facts)
        hf = CUR["hasf"]
        no = np.array([i not in hf for i in ids], bool)
        if no.any():
            sub = [i for i, b in zip(ids, no) if b]
            alt = _p(CUR["nof"], PNS[pr], sub, qs) if CUR["policy"] == "A" else _p(CUR["vec"], PVS[pr], sub, qs)
            out = out.copy(); out[no] = alt
        return out

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
    CUR = {"tv": None, "prec": "bf16", "pa": "AJ", "ft": ""}
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
        """judge (cached) the given (id, row, face row[, index]) tuples for leaf question q; counted as judge calls.
        R69: a tuple may carry the index its row and face row belong to (scope entries); else idx_."""
        cr = str(crop)
        if LOG["on"]:     # R68 LOG
            LOG["jp"].update((t[0], q, cr, int(t[2]), False) for t in ids_rows_fr)
        need = [t for t in ids_rows_fr if (t[0], q, cr, int(t[2]), False) not in JP]
        by = {}
        for t in need:
            ix = t[3] if len(t) > 3 else idx_
            by.setdefault(id(ix), (ix, []))[1].append(t)
        for ix, ts in by.values():
            ts = list({(t[0], int(t[2])): t for t in ts}.values())
            p = orig(ix, judge, np.array([t[1] for t in ts], int), np.array([t[2] for t in ts], int), q,
                     batch=batch, crop_person=crop)
            for t, x in zip(ts, p):
                JP[(t[0], q, cr, int(t[2]), False)] = float(x)
        ST["judge"].update((t[0], q, cr) for t in ids_rows_fr)

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
        def thr(order):
            n, M = len(order), budget(len(order))
            F = min(n, M)
            if D.get("c1"):
                judge_ids(idx_, judge, q, crop, [(i, v[1], v[2], v[3]) for i, v in order[:F]], batch)
                while F < n:
                    blk = order[F - M:F] if F >= M else order[:F]
                    rate = np.mean([JP[(i, q, str(crop), int(v[2]), False)] >= .7 for i, v in blk])
                    if rate < .5:
                        break
                    F2 = min(n, F + M)
                    judge_ids(idx_, judge, q, crop, [(i, v[1], v[2], v[3]) for i, v in order[F:F2]], batch)
                    F = F2
            return (-np.inf if F >= n else order[F - 1][1][0]), F
        order = sorted(sc.items(), key=lambda kv: -kv[1][0])
        if CUR["policy"] == "VS":     # RESULTS 65: separate budgets for photos with / without a facts line
            hf = CUR["hasf"]
            (tf, Ff), (tn, Fn) = thr([kv for kv in order if kv[0] in hf]), thr([kv for kv in order if kv[0] not in hf])
            tj, F = (tf, tn), Ff + Fn
        else:
            tj, F = thr(order)
        if D.get("c1"):
            ST["d_ext"][key] = F
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
        if D.get("ms"):     # R69 DRY pass A: a judge-kind leaf gets the millisecond P (no judge)
            return pt
        if CUR["pa"] == "ON":     # R69 ON: the leaf's scope grows with every call; threshold re-derived each call
            ST["d_scope"][(q, str(crop))].update({i: (float(x), int(r_), int(f_), idx_) for i, x, r_, f_ in zip(ids, pt, rows, fr)})
            ST["d_tj"].pop((q, str(crop)), None)
        tj = leaf_threshold(idx_, judge, q, crop, D, batch)
        if CUR["policy"] == "VS":
            no = np.array([i not in CUR["hasf"] for i in ids], bool)
            if tj is None:     # fallback: budget over this call's rows, per group
                tj = (kth(pt[~no], budget(int((~no).sum()))), kth(pt[no], budget(int(no.sum()))))
            keep = np.where((~no & (pt >= tj[0])) | (no & (pt >= tj[1])))[0]
        else:
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
                ST["d_scope"][(q, str(crop))].update({i: (float(x), int(r_), int(fmap[int(r_)]), idx_) for i, x, r_ in zip(ids, pt, rr)})

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
        if ST["mode"] == "A" and CUR["pa"] == "DRY":     # R69: the dry run answers every call from the millisecond scorer
            if ref_img is not None:
                return cached(TV, PTV, ids, scorer_question(question))
            D = dict(DVARS["r59d"], ms=True)
            ST["d_orig"] = question
            node = decompose(idx_, question, crop_person, D, excl=D.get("c2", False) and question in EXCL)
            return evaluate(idx_, judge, node, rows, fr, crop_person, D, batch)
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
    judge_rows_inner = judge_rows
    CALLS = {"A": [], "B": []}
    SUBS = {}

    def judge_rows(idx_, judge, rows, face_rows, question, ref_img=None, batch=96, crop_person=False):     # R68 LOG
        out = judge_rows_inner(idx_, judge, rows, face_rows, question, ref_img=ref_img, batch=batch, crop_person=crop_person)
        rows_ = np.asarray(rows, int)
        if len(rows_):
            # rows of idx_ (the library or a subset of it) -> LIBRARY rows; sub = which subset (-1 = the library itself),
            # because place masks are built over idx_'s own photos (PlaceIndex results depend on the whole index)
            LR = CUR["librow"]
            ids_all = idx_.items.item_id.astype(str).tolist()
            if len(ids_all) == len(LR):
                sk = -1
            else:
                key = tuple(LR[i] for i in ids_all)
                if key not in SUBS:
                    SUBS[key] = len(SUBS)
                sk = SUBS[key]
            c = dict(q=question, crop=str(crop_person), ref=ref_img is not None, excl=question in EXCL, sub=sk,
                     rows=[LR[ids_all[r_]] for r_ in rows_.tolist()], fr=np.asarray(face_rows, int).tolist())
            c["out"] = [float(x) for x in np.asarray(out, np.float32)]     # R69: pass A outputs too (DRY parity)
            CALLS[ST["mode"]].append(c)
        return out
    engine._judge_rows = converse._judge_rows = agent._judge_rows = judge_rows

    def kth(vals, Kv):
        v = np.sort(np.fromiter(vals, float))[::-1]
        return -np.inf if Kv >= len(v) else v[Kv - 1] if Kv > 0 else np.inf

    configs = arg("configs").split(",")
    variants = configs
    ORD = pd.read_parquet(arg("orders"))
    ORD["item_id"] = ORD.item_id.astype(str)
    RANKS = {u: g for u, g in ORD.groupby("user")}

    import gzip
    rec_dir = Path(arg("rec"))
    LIBS_DONE = set()

    def write_rec(v, q, u, sub, rec, st):     # R68 LOG: one JSON line per (request, config)
        d = rec_dir / v; d.mkdir(parents=True, exist_ok=True)
        if u not in LIBS_DONE:
            f = rec_dir / f"lib_{u}.json"
            if not f.exists():
                it = sub.items
                f.write_text(json.dumps(dict(user=u, ids=it.item_id.astype(str).tolist(),
                                             place=it["place"].fillna("").astype(str).tolist() if "place" in it else None)))
            LIBS_DONE.add(u)
        tv, vec = defaultdict(dict), defaultdict(dict)
        for i, qs in LOG["tv"]:
            tv[qs][CUR["librow"][i]] = PFS[(CUR["prec"], CUR["ft"])][(i, qs)]
        for i, qs in LOG["vec"]:
            vec[qs][CUR["librow"][i]] = PVS[CUR["prec"]][(i, qs)]
        hf = CUR["hasf"]
        touched = {i for i, _ in LOG["tv"]} | {i for i, _ in LOG["vec"]}
        jp = [[CUR["librow"][k_[0]], k_[1], k_[2], k_[3], bool(k_[4]), dict.__getitem__(JP, k_)] for k_ in sorted(LOG["jp"]) if dict.__contains__(JP, k_)]
        out = dict(qid=q["qid"], user=u, cat=q["cat"], cfg=v, policy=CUR["policy"], A=CALLS["A"], B=CALLS["B"],
                   tv=tv, vec=vec, hasf=sorted(CUR["librow"][i] for i in touched if i in hf), jp=jp,
                   judged=sorted([CUR["librow"][x[0]], x[1], x[2]] for x in ST["judge"]), ext={f"{k_[0]}|{k_[1]}": x for k_, x in ST["d_ext"].items()},
                   scope={f"{k_[0]}|{k_[1]}": [[CUR["librow"][i], x[0], x[2]] for i, x in d_.items()] for k_, d_ in ST["d_scope"].items()},
                   error=rec["error"], n_judge=st["judge"], route=st["route"],
                   subs=[list(k_) for k_, _ in sorted(SUBS.items(), key=lambda kv: kv[1])])
        with gzip.open(d / f"rec{k}.jsonl.gz", "at") as fh:
            fh.write(json.dumps(out) + "\n")

    def hasf_for(u, cfg):
        # R69: every photo has its line; cfg = "<AJ|DRY|ON>_<bf16|fp32>". R69b: optional "_<coverage %>" (newest-first
        # order N, policy V for photos without a line), e.g. "DRY_fp32_25"; no suffix = 100% (R69 behaviour unchanged)
        p_ = [x for x in cfg.split("_") if not x.startswith("F")]     # R70: "F<tag>" = facts variant, not coverage
        cov, o, pol = (p_[2] if len(p_) > 2 else "100"), "N", "V"
        g = RANKS[u]
        m = math.ceil(int(cov) / 100 * len(g))
        return set(g.item_id[g[f"rank_{o}"].to_numpy() < m]), pol
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
            JP.clear(); PTV.clear(); PCE.clear()
            for dd in (*PFS.values(), *PNS.values(), *PVS.values()):
                dd.clear()
        if u not in subs:
            subs.clear(); subs[u] = store.subset(idx, np.where(user_of == u)[0])
        sub = subs[u]
        if CUR.get("libu") != u:     # R68 LOG
            CUR["librow"] = {i: r_ for r_, i in enumerate(sub.items.item_id.astype(str).tolist())}; CUR["libu"] = u
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
        EXCL.clear()
        sp_ = stored.get(q["qid"]) or {}
        for t_ in sp_.get("turns") or []:
            for a_ in t_["plan"]["albums"]:
                if a_.get("exclude_question"):
                    EXCL.add(a_["exclude_question"])
        for v in variants:
            CUR["hasf"], CUR["policy"] = hasf_for(u, v)
            CUR["pa"], CUR["prec"] = v.split("_")[:2]
            CUR["ft"] = next((x[1:] for x in v.split("_")[2:] if x.startswith("F")), "")     # R70
            CUR["tv"] = tv_for(u); CUR["nof"], CUR["vec"] = side_for(u)
            ST["scope"] = defaultdict(dict); ST["tK"] = {}; ST["scope_j"] = defaultdict(dict); ST["tJ"] = {}
            ST["jM"] = None; ST["record_j"] = False; ST["trip"] = False
            ST["D"] = None; ST["d_scope"] = defaultdict(dict); ST["d_all"] = [DVARS["r59d"]]; ST["d_ext"] = {}
            CALLS["A"], CALLS["B"] = [], []     # R68 LOG
            SUBS.clear()
            for t_ in ("jp", "tv", "vec", "nof"):
                LOG[t_] = set()
            reset("A")
            if CUR["pa"] == "ON":     # no pass A at all
                recA, stA = dict(error=None), dict(misses=0, judge=0)
            else:
                recA = run_once()
                stA = snap()
            assert CUR["pa"] != "DRY" or (stA["judge"] == 0 and stA["misses"] == 0), "dry run judged a photo"
            ST["tK"] = {key: kth(d.values(), 0) for key, d in ST["scope"].items()}     # K = 0: no facts reader
            reset("B"); ST["D"] = DVARS["r59d"]; ST["d_tj"] = {}
            LOG["on"] = True
            rec = run_once(); st = snap(); st["K"] = 0; st["M"] = None
            LOG["on"] = False
            write_rec(v, q, u, sub, rec, st)
            st["ext"] = {f"{k_[0]}|{k_[1]}": x for k_, x in ST["d_ext"].items()}
            st["n_facts"] = len(CUR["hasf"]); st["errA"] = recA["error"]; st["missA"] = stA["misses"]; st["judgeA"] = stA["judge"]
            st["scope_n"] = {f"{k_[0]}|{k_[1]}": len(d_) for k_, d_ in ST["d_scope"].items()}     # R69
            ST["d_ext"] = {}; ST["D"] = None
            outs[v].append(rec); stats[v][q["qid"]] = st
            (out_dir / v / f"part{k}.json").write_text(json.dumps(outs[v]))
            (out_dir / v / f"stats{k}.json").write_text(json.dumps(stats[v]))
        print(q["qid"], q["cat"], {v: (stats[v][q["qid"]]["judge"], "ERR" if outs[v][-1]["error"] else "ok") for v in variants},
              flush=True)
    save_new()
    print("CASCADE_DONE", flush=True)


if __name__ == "__main__":
    main()
