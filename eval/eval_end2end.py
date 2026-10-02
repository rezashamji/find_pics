"""End-to-end: engine (fast scores -> VLM judge on head -> random tail audit -> certificate) vs known answers.

The question this answers: when find_pics says "at least X% complete", is that true against ground truth?
Writes eval/end2end/<query>/ (judged.parquet, contact sheets of results / misses / false positives) and
eval/results_end2end.json. Usage: python eval/eval_end2end.py <index_dir> <testlib_dir> [head_size] [tail_budget]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from findpics import store
from findpics.contact import sheet
from findpics.engine import Thresholds, run_album
from findpics.models import ImageTextEncoder
from findpics.people import refs_from_items
from findpics.planner import AlbumSpec
from findpics.vlm import VLLMJudge



def main():

    idx_dir, tl = Path(sys.argv[1]), Path(sys.argv[2])
    head = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    tail = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    OUT = Path(__file__).parent / "end2end"; OUT.mkdir(exist_ok=True)
    idx = store.load(idx_dir)
    gt = json.load(open(tl / "ground_truth.json"))
    row = {iid: i for i, iid in enumerate(idx.items.item_id)}
    enc = ImageTextEncoder()
    judge = VLLMJudge()
    th = Thresholds(head_size=head, tail_budget=tail)

    QUERIES = [
        ("bread", AlbumSpec(name="Bread", looks=["a photo of bread", "a loaf of bread", "slices of bread"],
                            judge_question="Is there bread (a loaf, slices, rolls, baguette or buns) visible in this image?"), "concept:Bread", None),
        ("dog", AlbumSpec(name="Dogs", looks=["a photo of a dog"], judge_question="Is there a dog in this image?"), "concept:Dog", None),
        ("kevin_bacon", AlbumSpec(name="Kevin Bacon", person="Kevin Bacon", judge_question="Is a person visible?"), "person:Kevin Bacon", None),
        ("drew_barrymore", AlbumSpec(name="Drew Barrymore", person="Drew Barrymore", judge_question="Is a person visible?"), "person:Drew Barrymore", None),
        ("cage_1995_2005", AlbumSpec(name="Nicolas Cage 1995-2005", person="Nicolas Cage", date_from="1995-01-01", date_to="2006-01-01",
                                     judge_question="Is a person visible?"), "person:Nicolas Cage", ("1995-01-01", "2006-01-01")),
    ]
    results = {}
    for qname, spec, gkey, dr in QUERIES:
        t0 = time.time()
        refs = ref_face = None
        if spec.person:
            trows = [i for i, ps in enumerate(idx.items.apple_persons) if ps is not None and spec.person in list(ps)]
            refs, frows = refs_from_items(idx, trows, return_rows=True)
            ref_face = int(frows[0])
        g = gt[gkey]
        truth = set(row[u] for u in (g["pos"] if isinstance(g, dict) else g) if u in row)
        if dr:
            import pandas as pd
            tt = pd.to_datetime(idx.items.taken, utc=True, format="ISO8601")
            truth = {r for r in truth if pd.Timestamp(dr[0], tz="UTC") <= tt[r] < pd.Timestamp(dr[1], tz="UTC")}
        res = run_album(idx, spec, enc, judge, refs, ref_face_row=ref_face, th=th)
        got = set(res.returned.item_row)
        poss = set(getattr(res, "possible", []).item_row) if len(getattr(res, "possible", [])) else set()
        tp = got & truth
        true_recall = len(tp) / max(len(truth), 1)
        c = res.cert or {}
        r = dict(n_truth=len(truth), returned=len(got), true_pos=len(tp), false_pos_vs_labels=len(got - truth),
                 true_recall_vs_labels=true_recall, cert=c, seconds=round(time.time() - t0, 1),
                 lower_bound_holds=(c.get("recall_lower", 0) <= true_recall + 1e-9) if c else None, report=res.report,
                 possible=len(poss), possible_true=len(poss & truth),
                 recall_with_possible=len((got | poss) & truth) / max(len(truth), 1))
        results[qname] = r
        d = OUT / qname; d.mkdir(exist_ok=True)
        res.judged.to_parquet(d / "judged.parquet")
        P = idx.items.path.to_numpy()
        fp = sorted(got - truth)[:36]; miss = sorted(truth - got)[:36]; okk = sorted(tp)[:36]
        if okk: sheet([P[i] for i in okk], d / "true_pos.jpg", title=f"{qname}: returned AND labeled true (first 36 of {len(tp)})")
        if fp: sheet([P[i] for i in fp], d / "false_pos.jpg", labels=[f"{i}" for i in fp], title=f"{qname}: returned but NOT labeled true (first 36 of {len(got-truth)})")
        if miss: sheet([P[i] for i in miss], d / "missed.jpg", labels=[f"{i}" for i in miss], title=f"{qname}: labeled true but NOT returned (first 36 of {len(truth-got)})")
        print(f"\n### {qname}: truth={len(truth)} returned={len(got)} tp={len(tp)} fp(vs labels)={len(got-truth)} "
              f"true_recall={true_recall:.3f} cert_lower={c.get('recall_lower', float('nan')):.3f} point={c.get('recall_point', float('nan')):.3f} "
              f"holds={r['lower_bound_holds']} possible={len(poss)} (true {len(poss & truth)}) recall_incl_possible={r['recall_with_possible']:.3f} ({r['seconds']}s)")
        print(res.report)
    (Path(__file__).parent / "results_end2end.json").write_text(json.dumps(results, indent=1, default=str))



if __name__ == "__main__":
    main()
