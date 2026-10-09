"""Phone-judge weight compression (10-09): score every candidate's eye-label run against the q4 baseline.
Same frozen labels + questions as RESULTS 32-33 (eval_judge_distill.py in .cache/snap_jd; 261 right / 141 wrong).
Prints: kept right / wrong at P >= t; agreement with the 9B on held-out questions; the six "Is this a photo of X?"
subject queries at the 0.7 and 0.99 cuts (RESULTS 35's recommendation; NOTE: this label pool is the snap_jd one, not
RESULTS 35 (2)'s 126/117 outside-the-4-libraries pool, so compare candidates within this table only); and the paired
flips vs the baseline at the two shipped cuts (0.7, 0.95). Writes eval/compress/flips_<tag>.json (item ids + P's).
Usage (CPU): python eval/compress_report.py q3vl4b_vis4 q3vl4b_vis8 ...      (baseline = qwen3vl_4b_q4)
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / ".cache/snap_jd"
OUT = ROOT / "eval/compress"
os.chdir(SNAP); sys.path.insert(0, str(SNAP / "eval"))
from eval_question_variants import labels  # noqa: E402

SUBJ = ["food photos", "photos of a cat", "beach photos", "photos of flowers", "sunset photos", "photos of a church"]
QS = SUBJ + ["photos with a dog", "photos with a car", "photos with a bicycle", "photos with a boat"]
T = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def split_of(q):
    return "test" if int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 == 0 else "train"


held = [json.loads(l) for f in sorted(glob.glob("data/public/judge_distill/part*.jsonl")) for l in open(f)]
p9 = {f"{r['question']}\t{r['item_id']}": r["p9"] for r in held if split_of(r["question"]) == "test"}
lab = {(q, i): v for (q, i), v in labels().items() if q in QS}


def load(name):
    return json.load(open(f"eval/judge_distill_test/{name}.json"))


def kept(d, t, qs=QS):
    r = w = 0
    for (q, i), v in lab.items():
        k = f"{q}\t{i}"
        if q in qs and k in d["eye"] and d["eye"][k] >= t:
            r += v == "right"; w += v == "wrong"
    return r, w


def totals(d, qs=QS):
    r = sum(1 for (q, i), v in lab.items() if q in qs and f"{q}\t{i}" in d["eye"] and v == "right")
    w = sum(1 for (q, i), v in lab.items() if q in qs and f"{q}\t{i}" in d["eye"] and v == "wrong")
    return r, w


def agree(d):
    ks = [k for k in d["held"] if k in p9]
    return sum((d["held"][k] >= 0.7) == (p9[k] >= 0.7) for k in ks), len(ks)


def main(names):
    B0 = os.environ.get("FP_CMP_BASE", "qwen3vl_4b_q4"); base = load(B0)
    R, W = totals(base); sr, sw = totals(base, SUBJ)
    print(f"# eye labels: {R} right / {W} wrong (subject queries: {sr} / {sw}); kept right r / wrong w at P >= t")
    for n in [B0] + names:
        d = load(n); a, na = agree(d)
        row = " | ".join(f"t{t}: {kept(d, t)[0]}r {kept(d, t)[1]}w" for t in T)
        s7, s99 = kept(d, 0.7, SUBJ), kept(d, 0.99, SUBJ)
        print(f"{n:22s} agree9B {a}/{na} | {row} | subject q: 0.7 {s7[0]}/{s7[1]}, 0.99 {s99[0]}/{s99[1]}")
    print(f"\n# paired flips vs {B0} (eye label of each flipped photo): +kept = candidate keeps, base drops")
    for n in names:
        d = load(n); flips = {}
        for t in (0.7, 0.95):
            f = [(q, i, v, base["eye"][f"{q}\t{i}"], d["eye"][f"{q}\t{i}"]) for (q, i), v in sorted(lab.items())
                 if f"{q}\t{i}" in d["eye"] and (base["eye"][f"{q}\t{i}"] >= t) != (d["eye"][f"{q}\t{i}"] >= t)]
            gain = [x for x in f if x[4] >= t]; loss = [x for x in f if x[4] < t]
            c = lambda xs, v: sum(x[2] == v for x in xs)
            print(f"{n:22s} t{t}: +kept {len(gain)} ({c(gain, 'right')} right, {c(gain, 'wrong')} wrong, "
                  f"{c(gain, 'unsure')} unsure) | -dropped {len(loss)} ({c(loss, 'right')} right, "
                  f"{c(loss, 'wrong')} wrong, {c(loss, 'unsure')} unsure)")
            flips[str(t)] = [{"query": q, "item_id": i, "eye": v, "p_base": pb, "p_cand": pc} for q, i, v, pb, pc in f]
        import statistics
        ks = [k for k in d["eye"] if k in base["eye"]]
        diffs = sorted(abs(d["eye"][k] - base["eye"][k]) for k in ks)
        print(f"{'':22s} |P - P_base| over {len(ks)} eye photos: median {statistics.median(diffs):.4f}, "
              f"p90 {diffs[int(0.9 * len(diffs))]:.4f}, max {diffs[-1]:.4f}")
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(flips, open(OUT / f"flips_{n}.json", "w"), indent=0)


if __name__ == "__main__":
    main(sys.argv[1:])
