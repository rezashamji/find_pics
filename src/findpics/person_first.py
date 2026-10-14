"""Person searches get a FIRST ANSWER instead of judging every face-matched photo (RESULTS 62).

Until 10-14 a person album with a condition judged EVERY photo the face filter accepted before showing anything: 41,541
photos for the owner on Reza's phone, ~55 h at the measured ~4.8 s a photo. Ordinary searches show a first answer
after ~480 judge calls with an honest bound and keep looking (RESULTS 43/44/44b). This module is the same idea for
person albums. Which branch a question takes is engine.is_look_question (RESULTS 49c).

FACT ("me at the beach", "me wearing a hat": the judge's yes alone, P >= 0.7). Exactly an ordinary fast-mode search
whose scope is the person's face-matched photos: order them by the album's look vector (fact_order; the phone may blend
in the learned head, HeadRanker.blendScores), run the phone's timed rounds (stream_timed = FindPicsCore.streamTimed,
params = PHONE_FAST = phoneFastParams), the first answer ends where first_answer_done says, every round carries the
Clopper-Pearson bound on what is still hiding in the unjudged rest, and the search keeps looking until every photo is
judged. At the end the album is what judging everything gives (every photo judged, P >= 0.7).

LOOK ("me looking heavier vs fit", "tired": relative to the person's own photos). There is no bound for a relative
album, but the full rule replaces each photo's score by its EVENT's median (photos <= 3 h apart; engine._split_pair),
so a few judged photos per event can sort the whole event. walk_order fixes the judging order before any answer: a
uniform random order of the person's photos (fnv1a64 of seed|id), skipping a photo while its event already has
LOOK_CAP judged photos, the skipped ones after (so big events get 3 photos early, every event gets one eventually, and
in the end everything is judged). After any number of calls:
  * pair_partial (two opposite looks): every photo of an event with >= 1 judged photo gets the median of its event's
    judged log-odds differences; the two-group split is fitted on the judged photos, each weighted by the photos it
    stands for (event size / judged in the event), likelihoods scaled back to the number judged (engine._two_groups
    weights); a covered photo goes to the album its event's posterior says (>= 0.9 and on the right side of the
    midpoint), photos of events not reached yet are "not sorted yet". No clear two groups -> None (the caller uses the
    rank margin on the judged photos, as the full rule falls back to it).
  * single_partial (one look): a judged photo keeps its own P, an unjudged photo of a covered event takes its event's
    median P; the album keeps covered photos ranking above the middle of the person's photos, the middle estimated
    from the judged photos with the same weights.
With every photo judged both are the full rules exactly (weights 1, event medians over all photos).
Parity: FindPicsCore PersonFirst.swift, fixture ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/person_first.json
(eval/person_first/fixtures.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import audit as A

FIRST_CALLS = 480     # a LOOK search's first answer: the same judge-call budget as fast mode's (phoneFastParams)
LOOK_CAP = 3          # first pass of the walk: at most this many judged photos per event
REL_CUT = 0.5         # engine.Thresholds.rel_cut: a look keeps the top half of the person's own photos
PAIR_SURE = 0.9       # engine.PAIR_SURE
PAIR_MIN = 20         # engine.PAIR_MIN: fewer judged photos than this -> no split yet

# FindPicsCore.phoneFastParams (RESULTS 41/43/44), field for field
PHONE_FAST = dict(head_size=150, head_chunk=50, head_window=100, head_max=2000, head_stop_rate=0.03, tail_budget=150,
                  auto_round_hits=3, auto_extra_rounds=1, round_calls=120, round_topup=30, interleave=0,
                  first_alpha_share=0.2, fast_stop_calls=480, fast_stop_topup=150, accept=0.7, alpha=0.05)


# ------------------------------------------------------------------------------------------------------------ FACT
def zscore(x) -> np.ndarray:
    """FindPicsCore zScores / head_ship.zscore: population std over the finite values, std 0 -> 1; non-finite -> -inf."""
    x = np.asarray(x, np.float64); f = np.isfinite(x)
    out = np.full(len(x), -np.inf)
    if f.any():
        m = x[f].mean(); s = x[f].std()
        out[f] = (x[f] - m) / (s if s > 0 else 1.0)
    return out


LOGIT_CLIP = float(np.log((1 - 1e-6) / 1e-6))


def blend_scores(cos, head_logit) -> np.ndarray:
    """FindPicsCore blendScores (RESULTS 48): z(cosine) + z(head logit clipped to +-LOGIT_CLIP); no vector -> -inf."""
    cos = np.asarray(cos, np.float64)
    lg = np.clip(np.asarray(head_logit, np.float64), -LOGIT_CLIP, LOGIT_CLIP)
    lg = np.where(np.isnan(lg), -LOGIT_CLIP, lg)
    lg = np.where(np.isfinite(cos), lg, -np.inf)
    zc, zh = zscore(cos), zscore(lg)
    return np.where(np.isfinite(zc) & np.isfinite(zh), zc + zh, -np.inf)


def fact_order(cos, head_logit=None) -> np.ndarray:
    """Positions of the person's photos, best first: the look score (cosine), or its blend with the learned head's
    logit. Ties keep the input order (FindPicsCore personFactOrder breaks ties by position too)."""
    key = np.asarray(cos, np.float64) if head_logit is None else blend_scores(cos, head_logit)
    key = np.where(np.isnan(key), -np.inf, key)
    return np.argsort(-key, kind="stable")


def max_rounds(n: int, p: dict) -> int:
    """FindPicsCore timedMaxRounds."""
    m0 = max(1, min(p["round_calls"], p["head_size"]))
    k, c = 2, m0
    while c < n:
        c *= 2; k += 1
    return k


def round_alpha(k: int, K: int, p: dict) -> float:
    """FindPicsCore timedAlpha."""
    s = p["first_alpha_share"]
    return s * p["alpha"] if k == 1 else (1 - s) * p["alpha"] / max(K - 1, 1)


@dataclass
class Round:
    k: int
    found: list            # positions the judge said yes to so far
    judged: int
    n_head: int
    cert: dict
    last: bool
    head_done: bool = False
    rounds_after_head_done: int = 0
    calls: list = field(default_factory=list)     # positions judged so far, in call order


def stream_timed(n: int, judge, p: dict = PHONE_FAST, perm=None, seed: int = 0):
    """Port of FindPicsCore streamTimed (StreamParams.roundCalls > 0): yields a Round after every round. judge(pos) ->
    P(yes) for ONE position (0 = best ranked). perm: a uniform permutation of range(n) fixed before any answer (default:
    numpy, seed). Same calls, same round ends, same random sample and bound as the Swift code and as
    eval/tune_stop_time.sim_timed (tests/test_person_first.py checks both)."""
    perm = np.random.default_rng(seed).permutation(n) if perm is None else np.asarray(perm, int)
    perm = [int(q) for q in perm]
    seen: dict[int, float] = {}
    calls: list[int] = []
    K = max_rounds(n, p)
    st = dict(b=0, pp=0, since=0, sat=0)

    def judge_one(q):
        if q not in seen:
            seen[q] = float(judge(q)); calls.append(q)

    def draw():
        while st["pp"] < n and (perm[st["pp"]] < st["b"] or perm[st["pp"]] in seen):
            st["pp"] += 1
        if st["pp"] >= n:
            return False
        judge_one(perm[st["pp"]]); st["pp"] += 1
        return True

    def sample():
        return [q for q in perm[:st["pp"]] if q >= st["b"]]

    def head_rule():
        b = st["b"]
        if b >= min(p["head_max"], n):
            return True
        if b < p["head_size"] or (b - p["head_size"]) % max(p["head_chunk"], 1) != 0:
            return False
        last = [seen[q] for q in range(max(0, b - (p["head_window"] or p["head_chunk"])), b)]
        return sum(v >= p["accept"] for v in last) / max(len(last), 1) < p["head_stop_rate"]

    k = 0
    while True:
        k += 1
        budget = n if k >= K else min(n, p["round_calls"] << min(k - 1, 40) if st["sat"] == 0 else 2 * len(seen))
        while len(seen) < budget and st["b"] < n:
            if p["interleave"] > 0 and st["since"] >= p["interleave"]:
                st["since"] = 0
                if draw():
                    continue
            judge_one(st["b"]); st["b"] += 1; st["since"] += 1
            if st["sat"] == 0 and head_rule():
                st["sat"] = k
                break
        if len(seen) >= n:
            st["b"] = n
        want = p["tail_budget"] if st["sat"] > 0 else p["round_topup"]
        if p["fast_stop_calls"] > 0 and p["fast_stop_topup"] > 0 and len(seen) >= p["fast_stop_calls"]:
            want = max(want, p["fast_stop_topup"])
        while st["b"] < n and len(sample()) < want and draw():
            pass
        ts = sample()
        found = sorted(q for q, v in seen.items() if v >= p["accept"])
        cert = A.certify(found=len(found), n_tail=n - st["b"], tail_labels=[seen[q] >= p["accept"] for q in ts],
                         alpha=round_alpha(k, K, p))
        last = st["b"] >= n
        yield Round(k=k, found=found, judged=len(seen), n_head=st["b"], cert=cert, last=last, head_done=st["sat"] > 0,
                    rounds_after_head_done=k - st["sat"] if st["sat"] > 0 else 0, calls=list(calls))
        if last:
            return


def fast_mode_wants_more(r: Round, p: dict = PHONE_FAST) -> bool:
    """FindPicsCore fastModeWantsMore (timed branch)."""
    if r.last:
        return False
    if p["fast_stop_calls"] > 0 and r.judged >= p["fast_stop_calls"]:
        return False
    if not r.head_done:
        return True
    return (p["auto_round_hits"] > 0 and r.rounds_after_head_done < p["auto_extra_rounds"]
            and r.cert["tail_hits"] >= p["auto_round_hits"])


def first_answer_done(r: Round, p: dict = PHONE_FAST) -> bool:
    """FindPicsCore firstAnswerDone."""
    return r.last or not fast_mode_wants_more(r, p)


# ------------------------------------------------------------------------------------------------------------ LOOK
_FNV_OFFSET, _FNV_PRIME, _MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1


def fnv1a64(s: str) -> int:
    """64-bit FNV-1a over the UTF-8 bytes (FindPicsCore fnv1a64)."""
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h ^= b; h = (h * _FNV_PRIME) & _MASK
    return h


def _event(event_of, i):
    return event_of.get(i, f"_{i}") if event_of is not None else f"_{i}"


def walk_order(ids: list, event_of: dict | None, seed: int = 0, cap: int | None = LOOK_CAP) -> list:
    """The LOOK judging order, fixed before any answer: positions into ids. A uniform random order (photos sorted by
    fnv1a64("<seed>|<id>"), ties by id); a photo is taken while its event has fewer than `cap` taken photos, the rest
    follow in the same order (cap None or 0: the plain random order)."""
    ids = [str(i) for i in ids]
    perm = sorted(range(len(ids)), key=lambda k: (fnv1a64(f"{seed}|{ids[k]}"), ids[k]))
    if not cap:
        return perm
    cnt: dict = {}
    first, rest = [], []
    for k in perm:
        e = _event(event_of, ids[k])
        if cnt.get(e, 0) < cap:
            first.append(k); cnt[e] = cnt.get(e, 0) + 1
        else:
            rest.append(k)
    return first + rest


def _events_of(ids, event_of):
    sizes: dict = {}
    for i in ids:
        e = _event(event_of, i); sizes[e] = sizes.get(e, 0) + 1
    return sizes


def _median(v):
    s = sorted(v); c = len(s)
    return s[c // 2] if c % 2 else (s[c // 2 - 1] + s[c // 2]) / 2


def single_partial(ids: list, event_of: dict | None, p: dict, rel_cut: float = REL_CUT, propagate: bool = True) -> dict:
    """One LOOK album after some photos are judged: {id: True/False} for every SORTED photo (judged, or in an event with a
    judged photo when propagate); photos absent from the result are not sorted yet. See the module docstring."""
    ids = [str(i) for i in ids]
    J = sorted(i for i in ids if i in p)
    if not J:
        return {}
    sizes = _events_of(ids, event_of)
    per: dict = {}
    for i in J:
        per.setdefault(_event(event_of, i), []).append(float(p[i]))
    med = {e: _median(v) for e, v in per.items()}
    vals = np.array([float(p[i]) for i in J]); w = np.array([sizes[_event(event_of, i)] / len(per[_event(event_of, i)]) for i in J])
    o = np.argsort(vals, kind="stable"); sv = vals[o]; cw = np.cumsum(w[o]); W = cw[-1]

    def rel(v):
        k = int(np.searchsorted(sv, v, side="right"))
        return (cw[k - 1] if k else 0.0) / W
    out = {}
    for i in ids:
        if i in p:
            out[i] = bool(rel(float(p[i])) > rel_cut)
        elif propagate and _event(event_of, i) in med:
            out[i] = bool(rel(med[_event(event_of, i)]) > rel_cut)
    return out


def _logit(p):
    q = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(q / (1 - q)))


def pair_partial(ids: list, event_of: dict | None, pa: dict, pb: dict, propagate: bool = True,
                 info: dict | None = None):
    """Two opposite LOOK albums after some photos are judged for both: {id: 0 | 1 | None} for every SORTED photo (0 = the
    first album, 1 = the second, None = not clearly either); absent = not sorted yet. None when fewer than PAIR_MIN
    photos are judged or the judged photos do not form two clear groups (caller: rank margin on the judged photos)."""
    from .engine import _two_groups
    ids = [str(i) for i in ids]
    J = sorted(i for i in ids if i in pa and i in pb)
    if len(J) < PAIR_MIN:
        return None
    sizes = _events_of(ids, event_of)
    per: dict = {}
    for i in J:
        per.setdefault(_event(event_of, i), []).append(_logit(pa[i]) - _logit(pb[i]))
    med = {e: _median(v) for e, v in per.items()}
    ev = [_event(event_of, i) for i in J]
    x = np.array([med[e] for e in ev]); w = np.array([sizes[e] / len(per[e]) for e in ev])
    post, clear, mid = _two_groups(x, info, groups=np.array(ev, dtype=object), weights=w)
    if not clear:
        return None
    side = {}
    for e, v, q in zip(ev, x, post):
        side[e] = 0 if (q >= PAIR_SURE and v > mid) else (1 if (q <= 1 - PAIR_SURE and v < mid) else None)
    out = {}
    for i in ids:
        e = _event(event_of, i)
        if e in side and (propagate or i in pa and i in pb):
            out[i] = side[e]
    return out


def rank_margin_pair(pa: dict, pb: dict, margin: float = 0.3, own_cut: float = REL_CUT) -> dict:
    """FindPicsCore rankMarginPair (engine.make_exclusive's fallback) on the photos judged so far."""
    def rank(p):
        s = np.sort(np.array(list(p.values()), float)); n = max(len(s), 1)
        return {i: float(np.searchsorted(s, v, side="right")) / n for i, v in p.items()}
    ra, rb = rank(pa), rank(pb)
    out = {}
    for i in set(pa) | set(pb):
        a, b = ra.get(i), rb.get(i)
        if a is not None and b is not None:
            out[i] = 0 if a - b >= margin else (1 if b - a >= margin else None)
        elif a is not None:
            out[i] = 0 if a > own_cut else None
        else:
            out[i] = 1 if b > own_cut else None
    return out


def pair_albums(ids: list, event_of: dict | None, pa: dict, pb: dict, propagate: bool = True) -> dict:
    """What a paired LOOK search shows after some photos are judged for both questions: pair_partial, or when that sees no
    clear two groups (or too few judged photos) the rank margin on the judged photos (engine.make_exclusive's fallback,
    FindPicsCore rankMarginPair). {id: 0 | 1 | None}; absent = not sorted yet. With everything judged = the full rule."""
    a = pair_partial(ids, event_of, pa, pb, propagate=propagate)
    if a is None:
        a = rank_margin_pair({i: pa[i] for i in pa if i in pb}, {i: pb[i] for i in pa if i in pb}) if pa else {}
    return a


UPDATE_EVERY = 20         # judge calls between screen updates until the first answer...
UPDATE_EVERY_LATER = 240  # ...and after it (FindPicsCore lookUpdateDue)


def update_due(calls: int, total: int) -> bool:
    """FindPicsCore lookUpdateDue: a LOOK search refreshes its albums every 20 judge calls up to the first answer
    (FIRST_CALLS), every 240 after it, and when everything is judged. Changes only WHEN the screen updates, never what
    it shows for a given number of calls."""
    if calls >= total:
        return True
    return calls % UPDATE_EVERY == 0 if calls <= FIRST_CALLS else (calls - FIRST_CALLS) % UPDATE_EVERY_LATER == 0


def stream_look(ids: list, event_of: dict | None, n_questions: int, judge, seed: int = 0, cap: int | None = LOOK_CAP):
    """A LOOK person search (FindPicsCore streamPersonLook): photos in walk_order, every question per photo
    (judge(id, q) -> P(yes)), and after each photo whose calls make update_due true a dict: calls, photos judged,
    albums (pair: {id: 0|1|None} from pair_albums; single: {id: bool} from single_partial), sorted, first_answer
    (calls >= FIRST_CALLS or everything judged), last. The caller may stop at any update (the generator is lazy)."""
    ids = [str(i) for i in ids]
    pos = walk_order(ids, event_of, seed, cap)
    P = [dict() for _ in range(n_questions)]
    total = n_questions * len(ids)
    calls = 0
    for done, k in enumerate(pos, 1):
        i = ids[k]
        for q in range(n_questions):
            P[q][i] = float(judge(i, q)); calls += 1
        if update_due(calls, total) or done == len(ids):
            alb = pair_albums(ids, event_of, P[0], P[1]) if n_questions == 2 else single_partial(ids, event_of, P[0])
            yield dict(calls=calls, judged=done, albums=alb, sorted=len(alb), total=len(ids),
                       first_answer=calls >= FIRST_CALLS or done == len(ids), last=done == len(ids),
                       p=[dict(x) for x in P])
