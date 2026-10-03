"""Does adding 2x2 tile vectors to the cheap first stage help the STREAMING product on small objects? (CPU replay)
Same stored 9B answers (eval/oracle), same text query; only the ranking that decides which photos the judge sees first
changes: whole-photo vector vs max(whole, 4 quarter tiles) (eval/tiles/tile_vectors.npy, [N, 14, D]: 0 = whole, 1-4 = 2x2).
Per concept: round-1 recall of the oracle set, judge calls in round 1, recall once 10% / 25% of the library is judged.
Usage: PYTHONPATH=src python eval/eval_streaming_tiles.py
"""
import sys

import numpy as np
import pandas as pd


def main():
    sys.path.insert(0, "eval")
    import findpics.engine as E
    from findpics import store
    from findpics.models import ImageTextEncoder
    from findpics.planner import AlbumSpec
    from eval_oracle import IDX, OUT, QUERIES
    idx = store.load(IDX)
    V = np.load("eval/tiles/tile_vectors.npy", mmap_mode="r")
    W5 = np.asarray(V[:, :5], np.float32)                     # whole + 2x2, [N, 5, D]
    enc = ImageTextEncoder(idx.clip_model, device="cpu")
    E._frame_for = lambda idx, r, fr: r
    E._boxed = lambda idx, im, fr: im
    rows = []
    for Q in QUERIES:
        f = OUT / f"{Q['qid']}.parquet"
        if not f.exists():
            continue
        P = pd.read_parquet(f).p.to_numpy(); oracle = set(np.where(P >= 0.7)[0])
        t = enc.texts(Q["looks"]).astype(np.float32).mean(0)
        S = W5 @ t
        for name, L in (("whole", S[:, 0]), ("whole+2x2", S.max(1))):
            class J:
                calls = 0

                def p_yes(self, ims, q):
                    J.calls += len(ims); return [float(P[int(i)]) for i in ims]
            E.look_scores = lambda idx, enc, looks, avoid, _L=L: _L
            curve = []
            for r in E.stream_album(idx, AlbumSpec(name=Q["qid"], looks=Q["looks"], judge_question=Q["q"]), None, J(), None,
                                    th=E.Thresholds(stream=True)):
                curve.append((J.calls / len(P), len(set(r.returned.item_row) & oracle) / max(len(oracle), 1), J.calls))
            at = lambda fr: max([c[1] for c in curve if c[0] <= fr + 1e-9] or [np.nan])
            rows.append(dict(concept=Q["qid"], oracle=len(oracle), ranking=name, round1_recall=curve[0][1],
                             round1_calls=curve[0][2], at10=at(0.10), at25=at(0.25)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["concept", "oracle"], columns="ranking", values=["round1_recall", "at25"]).round(3).to_string())
    print(df.groupby("ranking")[["round1_recall", "round1_calls", "at10", "at25"]].median().round(3).to_string())
    df.to_json("eval/results_streaming_tiles.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
