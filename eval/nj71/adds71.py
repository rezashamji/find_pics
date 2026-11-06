"""RESULTS 71 (coordinator's reporting request 11-06, within the pre-registered NJ - R59d measure): what the careful check
(R59d-DRY = the phone's cascade, RESULTS 69 DRY_fp32 config rerun in the same jobs) ADDS to (+A) and REMOVES from (-D) the
judge-free answer NJ, per category; + split by the judge-leaf kind the photo was judged under in R59d (route.py kinds;
'trip-no-place' / 'exclude' as the cascade names them; a photo judged under several kinds -> the kinds joined by '+';
never judged -> 'not judged'); + R59d judge calls per search / budget floor (sum over its recorded judge leaves of
min(scope, max(50, ceil(5% of scope))) = route.budget).
Answers = union of the albums of the last message's final round (photos, as the phone counts them).
  python eval/nj71/adds71.py [--t=60,70] [--splits=train,dev] [--out=eval/nj71/adds.txt]"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/facts_scorer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common71 import CATS, final_albums, load, truth  # noqa: E402
from route import TRIP, _TAIL, budget, route  # noqa: E402


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def answers(d, split):
    Q = truth(split); R = load(d); out = {}
    for qid, q in Q.items():
        al = final_albums(R.get(qid), q) or []
        out[qid] = {str(x) for a in al for x in a}
    return out


def recs(d):
    out = {}
    for f in sorted((ROOT / d).glob("rec*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); out[r["qid"]] = r     # requeued shards re-append: the last line wins
    return out


def kind_of(q, crop):
    k = route(q, crop)
    if k:
        return k
    return "trip-no-place" if TRIP.search(_TAIL.sub("", q)) else "exclude"


def main():
    T = [x for x in (arg("t") or (ROOT / "eval/nj71/threshold.txt").read_text().split()[0] + ",70").split(",") if x]
    T = list(dict.fromkeys(T))
    L = ["RESULTS 71: careful check (R59d-DRY) vs judge-free answer (NJ): photos ADDED (+A: in R59d, not in NJ) and REMOVED "
         "(-D: in NJ, not in R59d), summed over requests; (requests with >= 1 add / >= 1 removal / requests)"]
    for split in arg("splits", "train,dev").split(","):
        Q = truth(split)
        lib = {}
        for f in (ROOT / "data/public/nj71/record/s1" / split).glob("lib_*.json"):
            x = json.loads(f.read_text()); lib[x["user"]] = x["ids"]
        R59 = answers(f"eval/nj71/replay/s1/{split}/DRY_fp32", split)
        RC = recs(f"data/public/nj71/record/s1/{split}/DRY_fp32")
        assert set(RC) >= set(R59), "missing R59d recordings"
        J = {}
        for qid, r in RC.items():
            ids = lib[r["user"]]
            jk = defaultdict(set)
            for row, lq, crop in r["judged"]:
                jk[ids[row]].add(kind_of(lq, crop))
            J[qid] = jk
        for t in T:
            NJ = answers(f"eval/nj71/replay/s1/{split}/NJ_fp32_T{t}", split)
            L += ["", f"{split}, NJ t=.{t}: cat | +A photos (requests) | -D photos (requests) | requests | R59d answer photos | NJ answer photos"]
            KA, KD = Counter(), Counter()
            for c in CATS + ["ALL"]:
                qs = [q for q in Q if c == "ALL" or Q[q]["cat"] == c]
                A = [len(R59[q] - NJ[q]) for q in qs]; Dd = [len(NJ[q] - R59[q]) for q in qs]
                L.append(f"  {c} | +{sum(A)} ({sum(x > 0 for x in A)}) | -{sum(Dd)} ({sum(x > 0 for x in Dd)}) | {len(qs)} | "
                         f"{sum(len(R59[q]) for q in qs)} | {sum(len(NJ[q]) for q in qs)}")
            for q in Q:
                for i in R59[q] - NJ[q]:
                    KA["+".join(sorted(J[q].get(i, {"not judged"})))] += 1
                for i in NJ[q] - R59[q]:
                    KD["+".join(sorted(J[q].get(i, {"not judged"})))] += 1
            L.append(f"  by the judge-leaf kind the photo was judged under in R59d (all categories): kind | +A | -D")
            for k in sorted(set(KA) | set(KD), key=lambda k: -(KA[k] + KD[k])):
                L.append(f"    {k} | +{KA[k]} | -{KD[k]}")
        # judge calls vs budget floor
        st = {}
        for f in sorted((ROOT / f"eval/nj71/replay/s1/{split}/DRY_fp32").glob("stats*.json")):
            st.update(json.loads(f.read_text()))
        PER = defaultdict(list)
        zero_floor, n_calls, n_floor, n_ext = 0, [], [], 0
        for qid in Q:
            r = RC[qid]
            calls = st[qid]["judge"] + st[qid].get("judgeA", 0)
            floor = sum(budget(len(v)) for v in r["scope"].values())
            n_ext += any(r["ext"].get(k, 0) > budget(len(v)) for k, v in r["scope"].items())
            n_calls.append(calls); n_floor.append(floor)
            if floor > 0:
                PER[Q[qid]["cat"]].append(calls / floor); PER["ALL"].append(calls / floor)
            elif calls > 0:
                zero_floor += 1
        L += ["", f"{split}: R59d judge calls per search / budget floor (sum over its recorded judge leaves of route.budget(scope) = "
                  f"min(scope, max(50, ceil(5% of scope)))); requests with floor > 0 only. cat | requests | median | p90 | mean | max | "
                  f"ratio > 1 (budget extension or unrecorded leaf)"]
        for c in CATS + ["ALL"]:
            rat = np.array(PER[c])
            if not len(rat):
                L.append(f"  {c} | 0 | - | - | - | - | -"); continue
            L.append(f"  {c} | {len(rat)} | {np.median(rat):.2f} | {np.percentile(rat, 90):.2f} | {rat.mean():.2f} | {rat.max():.2f} | {(rat > 1 + 1e-9).sum()}")
        L.append(f"  requests with judge calls but no recorded leaf {zero_floor}; with a c1 budget extension beyond the floor {n_ext}; "
                 f"calls mean / median / p90 {np.mean(n_calls):.0f} / {np.median(n_calls):.0f} / {np.percentile(n_calls, 90):.0f}, "
                 f"floor {np.mean(n_floor):.0f} / {np.median(n_floor):.0f} / {np.percentile(n_floor, 90):.0f} (all {len(Q)} requests)")
    txt = "\n".join(L)
    print(txt)
    if arg("out"):
        Path(arg("out")).write_text(txt + "\n")


if __name__ == "__main__":
    main()
