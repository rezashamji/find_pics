"""find pics vs Apple Photos search on the recall-audit test set (RESULTS 34: 4 DISBench libraries, 7,886 photos,
6 queries, 1,950 blind eye labels + earlier eye labels). Both systems are scored by eval_recall.score_set: the same
strata, labels and estimator as RESULTS 34.

Results files (one per system / query style), TSV with a header line:
    query <TAB> search_text <TAB> n_returned <TAB> filenames
  query      = dog | car | bicycle | beach | sunset | food
  filenames  = the returned photos' filenames (<item_id>.jpg) in the system's order, separated by commas
  The Mac writes eval/apple_photos/results_apple_<style>.tsv (MAC_INBOX M29); `findpics` below writes
  eval/apple_photos/results_findpics_<mode>.tsv in the same format.

  looks    (GPU, main or vLLM env): PE-Core-B-16 (the phone's image-text model) look scores of the 7,886 photos
           -> eval/apple_photos/looks_b16.parquet (fast mode ranks by these, as the phone does)
  findpics (CPU): the phone pipeline from the judge's stored P(yes) on every photo (eval/recall_audit/scores_<tag>/):
           exhaustive = every photo judged; fast = round 1 of FindPicsCore.streamRounds with the app's parameters
           (Search.swift: head 150, chunks of 50 up to 1,500 while >= 3% of the last chunk is yes, random tail sample
           150), cut = FindPicsCore.judgeCutoff (0.99 for "Is this a photo of X?", else 0.7), all 7,886 photos as ONE
           library (as imported into Photos). Fast mode's tail sample is random: 20 seeds, the TSV holds seed 0.
  score    (CPU): python eval/score_apple_photos.py score [files...]  (default: every results_*.tsv) -> report.txt
  selftest (CPU): score_set on the RESULTS 34 judge's own kept set == eval_recall.estimate() point estimates.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
from eval_recall import QUERIES, _labels_eye, load_scores, old_labels, plans, score_set  # noqa: E402

D = Path("eval/apple_photos")
SHORT = dict(zip(["dog", "car", "bicycle", "beach", "sunset", "food"], QUERIES))
# what the Mac types, per style (M29). keyword = the Photos search-bar habit; phrases = natural-language search.
STYLES = {
    "keyword": {k: k for k in SHORT},
    "phrase_of": {"dog": "photos of a dog", "car": "photos of a car", "bicycle": "photos of a bicycle",
                  "beach": "photos of a beach", "sunset": "photos of a sunset", "food": "photos of food"},
    "phrase_with": {"dog": "photos with a dog", "car": "photos with a car", "bicycle": "photos with a bicycle",
                    "beach": "photos with a beach", "sunset": "photos with a sunset", "food": "photos with food"},
}
SUBJECT = re.compile(r"^\s*is\s+this\s+(a\s+|an\s+)?(photo|picture|image|pic)\s+of\b", re.I)   # JudgeCutoff.swift


def judge_cutoff(question):
    return 0.99 if SUBJECT.search(question) else 0.7


def looks():
    from PIL import Image
    from findpics.models import ImageTextEncoder
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path("eval/recall_audit/scores").glob("part*.parquet"))])
    man = sc.drop_duplicates("item_id")[["item_id", "path"]].reset_index(drop=True)   # the package's source bytes
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    P = plans()
    T = {q: enc.texts(P[q].albums[0].looks) for q in QUERIES}
    V = []
    for b in range(0, len(man), 256):
        ims = [Image.open(f).convert("RGB") for f in man.path[b:b + 256]]
        V.append(enc.images(ims).astype(np.float32))
        print(b, flush=True)
    V = np.concatenate(V)
    out = pd.DataFrame({"item_id": man.item_id})
    for q in QUERIES:
        out[q] = (V @ T[q].T).mean(1)     # no avoid words in these plans; photos only (one vector each)
    out.to_parquet(D / "looks_b16.parquet")


def stream_round1(p, order, cut, seed, head=150, chunk=50, head_max=1500, stop=0.03, tail_budget=150):
    """FindPicsCore.streamRounds, round 1 only (= what the app shows when not exhaustive). p: P(yes) by position in
    `order` (best fast score first). Returns found positions and the number of judge calls."""
    n = len(order)
    n_head = min(head, n)
    while n_head < min(head_max, n):
        last = p[max(0, n_head - chunk):n_head]
        if (last >= cut).mean() < stop:
            break
        n_head = min(n_head + chunk, n)
    rng = np.random.default_rng(seed)
    ts = rng.permutation(np.arange(n_head, n))[:min(n - n_head, tail_budget)]
    found = [i for i in range(n_head) if p[i] >= cut] + [int(i) for i in ts if p[i] >= cut]
    return sorted(found), n_head + len(ts), n_head


def write_tsv(path, sets, texts=None):
    with open(path, "w") as f:
        f.write("query\tsearch_text\tn_returned\tfilenames\n")
        for k, q in SHORT.items():
            ids = sets[q]
            f.write(f"{k}\t{(texts or {}).get(k, q)}\t{len(ids)}\t{','.join(f'{i}.jpg' for i in ids)}\n")


def findpics(tag="q3vl4b_real"):
    sc = pd.concat([pd.read_parquet(f) for f in sorted(Path(f"eval/recall_audit/scores_{tag}").glob("part*.parquet"))])
    lk = pd.read_parquet(D / "looks_b16.parquet").set_index("item_id")
    ex, fast, info = {}, {}, []
    seeds = {}
    for q in QUERIES:
        d = sc[sc["query"] == q].set_index("item_id")
        assert len(d) == 7886 and d.question.nunique() == 1, (q, len(d))
        cut = judge_cutoff(d.question.iloc[0])
        # exhaustive: every photo judged; shown best-first by P then fast score (the app's grid order is not scored)
        look = lk.loc[d.index, q]
        dd = d.assign(look_b16=look.to_numpy())
        ex[q] = dd[dd.p >= cut].sort_values(["p", "look_b16"], ascending=False).index.tolist()
        order = dd.sort_values("look_b16", ascending=False, kind="stable")
        p = order.p.to_numpy()
        runs = [stream_round1(p, order.index, cut, s) for s in range(20)]
        fast[q] = [order.index[i] for i in runs[0][0]]
        seeds[q] = [set(order.index[i] for i in r[0]) for r in runs]
        info.append(dict(query=q, question=d.question.iloc[0], cut=cut, exhaustive_kept=len(ex[q]),
                         fast_head=runs[0][2], fast_judge_calls=runs[0][1],
                         fast_found_seed0=len(fast[q]),
                         fast_found_20seeds=f"{min(len(r[0]) for r in runs)}-{max(len(r[0]) for r in runs)}"))
    write_tsv(D / "results_findpics_exhaustive.tsv", ex)
    write_tsv(D / "results_findpics_fast.tsv", fast)
    I = pd.DataFrame(info)
    I.to_csv(D / "findpics_runs.tsv", sep="\t", index=False)
    print(I.to_string())
    return seeds


def read_results(path):
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    keep, order = {}, {}
    for r in t.itertuples():
        q = SHORT[r.query.strip()]
        fns = [f for f in re.split(r"[,|\s]+", r.filenames.strip()) if f]
        ids = [re.sub(r"\.(jpe?g|heic|png)$", "", f, flags=re.I) for f in fns]
        if str(r.n_returned).strip() and int(r.n_returned) != len(ids):
            print(f"WARNING {path} {r.query}: n_returned {r.n_returned} != {len(ids)} filenames")
        keep[q] = set(ids); order[q] = ids
    return keep, order


def score(files=None, seeds=None):
    files = files or sorted(D.glob("results_*.tsv"))
    df = load_scores(); old = old_labels(); eye = _labels_eye()
    pd.set_option("display.width", 250)
    lines, allr = [], []
    for f in files:
        keep, _ = read_results(f)
        for um in (False, True):
            R = score_set(df, old, eye, keep, unsure_match=um)
            R.insert(0, "system", Path(f).stem.replace("results_", ""))
            R.insert(1, "unsure_as", "match" if um else "no_match")
            allr.append(R)
    A = pd.concat(allr, ignore_index=True)
    A.to_csv(D / "scores.tsv", sep="\t", index=False)
    for um in ("no_match", "match"):
        lines.append(f"\n== unsure counted as {um}")
        for sysname, R in A[A.unsure_as == um].groupby("system", sort=False):
            lines.append(f"-- {sysname}")
            for r in R.itertuples():
                lines.append(
                    f"  {r.query:22s} returned {r.returned:5d}"
                    f"{'' if not r.returned_not_in_library else f' ({r.returned_not_in_library} not in the library!)'}"
                    f" | eye-labelled returns right {r.returned_labelled_match}/{r.returned_labelled}"
                    f" (unsure {r.returned_labelled_unsure})"
                    f" | precision est {r.precision_est:.2f} [{r.precision_boot_lo:.2f}, {r.precision_boot_hi:.2f}]"
                    f" | real returned est {r.real_returned_est:7.1f} of {r.real_est:7.1f}"
                    f" | recall {r.recall:.3f} boot [{r.recall_boot_lo:.3f}, {r.recall_boot_hi:.3f}]"
                    f" bayes [{r.recall_bayes_lo:.3f}, {r.recall_bayes_hi:.3f}]")
            F, T, N = R.real_returned_est.sum(), R.real_est.sum(), R.returned_in_library.sum()
            lines.append(f"  pooled (24 searches): returned {N}, real returned est {F:.1f} of {T:.1f}, "
                         f"recall {F / T:.3f}, precision est {F / max(N, 1):.2f}; mean of 6 recalls "
                         f"{R.recall.mean():.3f}")
    if seeds:
        lines.append("\n== fast mode, 20 tail-sample seeds (unsure = no match): recall min-median-max per query")
        for q in QUERIES:
            vals = [score_set(df, old, eye, {q: s}, B=10).set_index("query").loc[q, "recall"] for s in seeds[q]]
            lines.append(f"  {q:22s} {np.min(vals):.3f} {np.median(vals):.3f} {np.max(vals):.3f}")
    txt = "\n".join(lines)
    print(txt)
    (D / "report.txt").write_text(txt)


def selftest():
    from eval_recall import estimate
    df = load_scores(); old = old_labels(); eye = _labels_eye()
    keep = {q: set(df[(df["query"] == q) & (df.stratum == "kept")].item_id) for q in QUERIES}
    for um in (False, True):
        R0, _ = estimate(df, old, eye, unsure_match=um)
        R1 = score_set(df, old, eye, keep, unsure_match=um)
        for a, b in zip(R0.itertuples(), R1.itertuples()):
            assert abs(a.found - b.real_returned_est) < 0.11 and abs(a.found + a.missed - b.real_est) < 0.11, (a, b)
            assert abs(a.recall - b.recall) < 1e-3, (a.query, a.recall, b.recall)
            print(f"{a.query:22s} unsure={'match' if um else 'no':5s} estimate {a.recall:.4f} "
                  f"[{a.boot_lo:.3f},{a.boot_hi:.3f}] bayes [{a.bayes_lo:.3f},{a.bayes_hi:.3f}]   score_set {b.recall:.4f} "
                  f"[{b.recall_boot_lo:.3f},{b.recall_boot_hi:.3f}] bayes [{b.recall_bayes_lo:.3f},{b.recall_bayes_hi:.3f}]")
    print("selftest OK: score_set == estimate() on the RESULTS 34 judge's kept sets")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "looks":
        looks()
    elif cmd == "findpics":
        s = findpics(*(sys.argv[2:3]))
        score([D / "results_findpics_fast.tsv", D / "results_findpics_exhaustive.tsv"], seeds=s)
    elif cmd == "score":
        score(sys.argv[2:] or None)
    elif cmd == "selftest":
        selftest()
