"""Score the blind audits (audit/ and audit2/) -> per-category quality of the candidate answer sets, per truth version.

Labels are about (owner's words, photo) only, so the same labels score every truth version; membership of each
labelled photo is recomputed from queries_v1/ (as generated), queries_v2/ (hidden time/place dropped) and queries/
(v3 = v2 + hidden visual filters dropped in "photos from <moment>" requests).
Stratified estimate, pooled per category: each labelled photo was drawn from a sampling stratum of known size per request
(audit/: v1 answers, the 30 in-scope non-answers most similar to the words, the other in-scope non-answers; audit2/:
v2-added answers, out-of-v1-scope hard / other non-answers, decoy answers). For a version V:
  TP_V = sum over strata of (stratum size) x (share of labelled photos in it that are in V AND match the words)
  FP_V = ... in V and do NOT match;   FN_V = ... NOT in V and match
  precision = TP / (TP + FP), recall = TP / (TP + FN). unsure labels are left out (counted).
g (two albums): a photo in V is right only if the label names the album V puts it in; a photo in no album is a miss if
labelled album0/album1.
Usage: python eval/complex_bench/audit_score.py  -> eval/complex_bench/audit/quality.txt
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402

VERSIONS = [("v1", "queries_v1"), ("v2", "queries_v2"), ("v3", "queries")]
_va = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--versions=")), None)
if _va:
    VERSIONS = [tuple(x.split(":")) for x in _va.split(",")]
SPLIT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--split=")), "all")
HALF = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--half=")), "all")   # even | odd request index
OUTF = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), "quality.txt")


def labels(folder, pattern="labels_*.txt"):
    L = {}
    for f in sorted((ROOT / folder).glob(pattern)):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(None, 3)
            if len(p) >= 3:
                L[(p[0], p[1])] = p[2].strip().lower()
    return L


def load(path):
    out = {}
    for s in ("test", "dev"):
        for q in json.loads((ROOT / path / f"{s}.json").read_text())["queries"]:
            out[q["qid"]] = q
    return out


def albums(q):
    return [set(s) for s in q["answers"][-1]] if q else []


def main():
    k1 = json.loads((ROOT / "audit" / "key.json").read_text())
    k2p = ROOT / "audit2" / "key.json"
    k2 = json.loads(k2p.read_text()) if k2p.exists() else {}
    L1, L2 = labels("audit"), labels("audit2")
    Q = {v: load(p) for v, p in VERSIONS}
    Q1 = load("queries_v1")      # stratum sizes come from the audit's sampling design (v1 answer sets)
    # records: (cat, qid, stratum, item_id, label)
    recs, pops = [], defaultdict(float)
    anchors = defaultdict(lambda: [0, 0])
    unl = unsure = 0
    def ok(qid):
        return qid in Q1 and all(qid in Q[v] for v, _ in VERSIONS) and (SPLIT == "all" or Q[VERSIONS[-1][0]][qid]["split"] == SPLIT)

    half_ok = lambda name: HALF == "all" or (int(name[1:4]) % 2 == 0) == (HALF == "even")

    for name, k in k1.items():
        c, qid = k["cat"], k["qid"]
        if not ok(qid) or not half_ok(name):
            continue
        for j in range(1, k.get("n_anchor", 0) + 1):
            lab = L1.get((name, f"A{j}"))
            if lab in ("match", "no_match"):
                anchors[c][0 if lab == "match" else 1] += 1
        a1 = set().union(*albums(Q1[qid]))
        any_lab = False
        for x in k["items"]:
            lab = L1.get((name, str(x["n"])))
            if lab is None:
                unl += 1; continue
            any_lab = True
            if lab == "unsure":
                unsure += 1; continue
            s = {"answer": "ans", "album0": "ans", "album1": "ans", "hard": "hard_in", "random": "rand_in"}[x["role"]]
            recs.append((c, qid, s, x["item_id"], lab))
        if any_lab:
            pops[(c, "ans")] += len(a1)
            pops[(c, "hard_in")] += k["n_hard_pool"]
            pops[(c, "rand_in")] += k["n_rand_pool"]
            if name in k2:
                pops[(c, "added")] += k2[name]["n_added"]
                pops[(c, "out_hard")] += k2[name]["n_out_hard_pool"]
                pops[(c, "out_rand")] += max(0, k2[name]["n_out"] - k2[name]["n_out_hard_pool"])
    for name, k in k2.items():
        c, qid = k["cat"], k["qid"]
        if not ok(qid) or not half_ok(name):
            continue
        for x in k["items"]:
            lab = L2.get((name, str(x["n"])))
            if lab is None:
                unl += 1; continue
            if lab == "unsure":
                unsure += 1; continue
            s = {"added": "added", "out_hard": "out_hard", "out_rand": "out_rand", "decoy": "ans"}[x["role"]]
            recs.append((c, qid, s, x["item_id"], lab))
    nq = defaultdict(set)
    for c, qid, *_ in recs:
        nq[c].add(qid)
    out = [f"[split filter: {SPLIT}; request half: {HALF}; versions: {VERSIONS}]",
           "Blind eye audit of the candidate answer sets. Labellers: 5 independent Claude subagents (A-E) + audit2 labellers,",
           "protocol AUDIT_PROTOCOL.txt, photos at native <= 500 px, never shown which photos are answers.",
           "v1 = programs as generated; v2 = time/place constraints the words never state dropped; v3 = v2 + visual filters",
           "the words never state dropped in 'photos from <moment>' requests (category d). Estimates are stratified (see header).", ""]
    raw = ["Raw labelled counts per stratum (match / labelled):"]
    cats = sorted({r[0] for r in recs})
    for c in cats:
        raw.append(f"  [{c}] " + "  ".join(
            f"{s}: {sum(1 for r in recs if r[0] == c and r[2] == s and r[4] in ('match', 'album0', 'album1'))}/"
            f"{sum(1 for r in recs if r[0] == c and r[2] == s)} (pop {pops[(c, s)]:.0f})"
            for s in ("ans", "hard_in", "rand_in", "added", "out_hard", "out_rand")))
    out.append("cat | requests | " + " | ".join(f"{v} precision / recall" for v, _ in VERSIONS) + " | anchors right")
    for c in cats:
        row = [c, str(len(nq[c]))]
        for v, _ in VERSIONS:
            TP = FP = FN = 0.0
            for s in ("ans", "hard_in", "rand_in", "added", "out_hard", "out_rand"):
                R = [r for r in recs if r[0] == c and r[2] == s]
                if not R:
                    continue
                tp = fp = fn = 0
                for _, qid, _, iid, lab in R:
                    A = albums(Q[v][qid])
                    inV = [k for k, a in enumerate(A) if iid in a]
                    if c == "g":
                        right = bool(inV) and lab == f"album{inV[0]}"
                        tp += right; fp += bool(inV) and not right
                        fn += (not inV) and lab in ("album0", "album1")
                    else:
                        y = lab == "match"
                        tp += bool(inV) and y; fp += bool(inV) and not y; fn += (not inV) and y
                N = pops[(c, s)]
                TP += N * tp / len(R); FP += N * fp / len(R); FN += N * fn / len(R)
            p = TP / (TP + FP) if TP + FP else 0.0
            r_ = TP / (TP + FN) if TP + FN else 0.0
            row.append(f"{p:.2f} / {r_:.2f}")
        row.append(f"{anchors[c][0]}/{sum(anchors[c])}" if sum(anchors[c]) else "-")
        out.append(" | ".join(row))
    out += [""] + raw + [f"\nunlabelled items: {unl}; unsure labels (excluded): {unsure}"]
    ov = labels("audit", "overlap_main.txt")
    agree = tot = 0; dis = []
    for key_, lab in ov.items():
        other = L1.get(key_)
        if other is None or "unsure" in (lab, other):
            continue
        tot += 1
        if lab == other:
            agree += 1
        else:
            dis.append(f"{key_[0]} {key_[1]}: main={lab} subagent={other}")
    out.append(f"inter-labeller agreement (main agent vs subagents on 32 overlapping requests, unsure excluded): {agree}/{tot}")
    out += ["  " + d for d in dis[:40]]
    (ROOT / "audit" / OUTF).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
