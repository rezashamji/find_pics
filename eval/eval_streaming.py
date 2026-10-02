"""Streaming curve: how fast does the album approach "judge looked at every photo"? (CPU; replays stored oracle answers)
For each of the 20 oracle concepts (eval/oracle/<qid>.parquet holds the 9B judge's P(yes) on ALL 19,218 items and the
fast score), run engine.stream_album with th.stream=True and a judge that returns the stored answers. Per round:
cumulative judge calls, recall of the oracle set, stated lower bound (must never exceed the true recall).
Usage: python eval/eval_streaming.py   -> eval/results_streaming.json + printed table
"""
import json

import numpy as np
import pandas as pd


def main():
    import findpics.engine as E
    from findpics import store
    from findpics.planner import AlbumSpec
    from eval_oracle import IDX, OUT, QUERIES
    idx = store.load(IDX)
    E._frame_for = lambda idx, r, fr: r
    E._boxed = lambda idx, im, fr: im
    res, viol = {}, 0
    for Q in QUERIES:
        f = OUT / f"{Q['qid']}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f); P = d.p.to_numpy(); L = d.look.to_numpy()
        oracle = set(np.where(P >= 0.7)[0])

        class J:
            calls = 0

            def p_yes(self, ims, q):
                J.calls += len(ims); return [float(P[int(i)]) for i in ims]
        E.look_scores = lambda idx, enc, looks, avoid, _L=L: _L
        rows = []
        for r in E.stream_album(idx, AlbumSpec(name=Q["qid"], looks=Q["looks"], judge_question=Q["q"]), None, J(), None,
                                th=E.Thresholds(stream=True)):
            rec = len(set(r.returned.item_row) & oracle) / max(len(oracle), 1)
            lo = r.cert["recall_lower"]; viol += lo > rec + 1e-9
            rows.append(dict(round=r.cert["round"], judge_calls=J.calls, frac_library=J.calls / len(P), recall=rec,
                             stated_lower=lo, returned=len(r.returned)))
        res[Q["qid"]] = dict(oracle=len(oracle), rounds=rows)
        print(f"{Q['qid']:18s} oracle={len(oracle):5d} | " + "  ".join(
            f"{x['frac_library']:.0%}:{x['recall']:.2f}/{x['stated_lower']:.2f}" for x in rows), flush=True)
    # recall reached at fixed fractions of the library judged (interpolated on rounds)
    for frac in (0.1, 0.25, 0.5):
        vals = []
        for q in res.values():
            ok = [x["recall"] for x in q["rounds"] if x["frac_library"] <= frac + 1e-9]
            vals.append(ok[-1] if ok else np.nan)
        print(f"recall of oracle once <= {frac:.0%} of library judged: median {np.nanmedian(vals):.2f}, min {np.nanmin(vals):.2f} "
              f"(n={np.sum(~np.isnan(vals))} concepts)")
    n_rounds = sum(len(q["rounds"]) for q in res.values())
    print(f"stated lower bound above true recall: {viol} of {n_rounds} rounds")
    json.dump(res, open("eval/results_streaming.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "eval")
    main()
