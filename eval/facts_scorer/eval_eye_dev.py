"""RESULTS 59: system level, DEV requests with photos labelled by eye against the request's WORDS (truth-independent):
b55 (RESULTS 55: dev b failures, 125 photos), audit5 (RESULTS 56, b), audit3 (RESULTS 52, h: last turn), audit4
(RESULTS 53, g: album0/album1 = a match, neither = no match; which album is not checked). Only split == dev entries are
kept (test entries are dropped on load, never looked at). For each run: returned = the final round's albums of the last
turn; precision / recall among the labelled photos (unsure = no match). Samples are NOT uniform: paired comparison only.
Usage: python eval/facts_scorer/eval_eye_dev.py name=<run dir> ... [--out=file]
"""
import json
import sys
from pathlib import Path

import pandas as pd

CB = Path("eval/complex_bench")


def labels(files):
    out = {}
    for f in files:
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 3 and p[1].isdigit():
                out[(p[0].replace(".jpg", ""), int(p[1]))] = p[2]
    return out


def table():
    dev = set(json.loads((CB / "splits.json").read_text())["dev"])
    rows = []
    L = labels([CB / "b55" / "labels.txt"])
    for line in open(CB / "b55" / "key.txt"):
        if line.startswith("#") or len(line.split()) < 4:     # "bwcolour" lines: no request
            continue
        sheet, tile, kind, iid = line.split()[:4]
        u, cat = sheet.rsplit("_", 1)
        user = u.replace("_N", "@N")
        if user not in dev or (sheet, int(tile)) not in L:
            continue
        rows.append(dict(src="b55", qid=f"{user}:{cat}", item_id=iid, eye=L[(sheet, int(tile))]))
    for a in ("audit5", "audit3", "audit4"):
        K = {k: v for k, v in json.loads((CB / a / "key.json").read_text()).items() if v.get("split") == "dev"}
        L = labels(sorted((CB / a).glob("labels_*.txt")))
        for sheet, v in K.items():
            for it in v["items"]:
                lab = L.get((sheet, int(it["n"])))
                if lab is None:
                    continue
                if a == "audit4":
                    lab = "match" if lab.startswith("album") else "no_match" if lab == "neither" else lab
                rows.append(dict(src=a, qid=v["qid"], item_id=str(it["item_id"]), eye=lab))
    return pd.DataFrame(rows).drop_duplicates(["src", "qid", "item_id"])


def returned(run_dir):
    R = {r["qid"]: r for f in sorted(Path(run_dir).glob("*part*.json")) for r in json.loads(f.read_text())}
    out = {}
    for q, r in R.items():
        t = r["turns"][-1] if r.get("turns") else None
        out[q] = set().union(*map(set, t["final"]["albums"])) if t and t.get("final") else set()
    return out


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    T = table()
    known = {q["qid"] for q in json.loads((CB / "queries" / "dev.json").read_text())["queries"]}
    T = T[T.qid.isin(known)].reset_index(drop=True)
    lines = [f"Dev eye labels (request words): {len(T)} photos in {T.qid.nunique()} requests; "
             + ", ".join(f"{s} {n}" for s, n in T.src.value_counts().items())
             + f"; match {int((T.eye == 'match').sum())}", "run | src | labelled returned | P | R | F1"]
    for name, d in runs:
        Rt = returned(d)
        for src in ["ALL"] + sorted(T.src.unique()):
            X = T if src == "ALL" else T[T.src == src]
            got = [i in Rt.get(q, set()) for q, i in zip(X.qid, X.item_id)]
            y = (X.eye == "match").tolist()
            tp = sum(g and t for g, t in zip(got, y)); ng = sum(got)
            P = tp / max(1, ng); Rr = tp / max(1, sum(y))
            lines.append(f"{name} | {src} | {ng}/{len(X)} | {P:.3f} | {Rr:.3f} | {2 * P * Rr / max(1e-9, P + Rr):.3f}")
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
