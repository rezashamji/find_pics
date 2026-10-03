"""Tiles vs whole-photo ranking, judged by HUMAN labels (Open Images positives), not by the 9B judge (whose errors
inflated the tile gain: 10-03 raw look). Recall of human-labeled positives inside the top K by each ranking.
Usage (CPU): PYTHONPATH=src python eval/eval_tiles_human.py"""
import json
import sys

import numpy as np

sys.path.insert(0, "eval")


def main():
    from eval_oracle import QUERIES
    from findpics import store
    from findpics.models import ImageTextEncoder
    idx = store.load("data/public/index_testlib"); row = {u: i for i, u in enumerate(idx.items.item_id)}
    gt = json.load(open("data/public/testlib/ground_truth.json"))
    W5 = np.asarray(np.load("eval/tiles/tile_vectors.npy", mmap_mode="r")[:, :5], np.float32)
    enc = ImageTextEncoder(idx.clip_model, device="cpu")
    print(f"{'concept':15s} {'human pos':>9s} | " + " | ".join(f"top{K} whole/tiles" for K in (1000, 2000)))
    for Q in QUERIES:
        if not Q["gt"] or Q["gt"] not in gt:
            continue
        pos = np.array([row[u] for u in gt[Q["gt"]]["pos"] if u in row])
        t = enc.texts(Q["looks"]).astype(np.float32).mean(0); S = W5 @ t
        res = []
        for K in (1000, 2000):
            for L in (S[:, 0], S.max(1)):
                top = np.argsort(-L)[:K]; res.append(np.isin(pos, top).mean())
        print(f"{Q['qid']:15s} {len(pos):9d} | " + " | ".join(f"{res[i]:.3f}/{res[i+1]:.3f}" for i in (0, 2)), flush=True)


if __name__ == "__main__":
    main()
