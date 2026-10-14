"""RESULTS 59: predictions of a trained scorer on a pair set, optionally reading a DIFFERENT facts file (e.g. the
27B-trained scorer reading phone facts with no retraining).
  python eval/facts_scorer/predict.py --model=<dir> --set=dev|eye|hold [--facts=<facts parquet>] [--name=<suffix>]
  -> <dir>/pred_<set><suffix>.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import scorer_question  # noqa: E402
from scorers import Scorer  # noqa: E402
from train import b16_image_vectors, b16_text, train_folds  # noqa: E402

D = Path("eval/facts_scorer/data")


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    mdir = Path(arg("model")); which = arg("set", "dev")
    cfg = json.loads((mdir / "config.json").read_text())
    facts_f = arg("facts", cfg["facts"])
    FT = pd.read_parquet(facts_f)
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    use_img = cfg["kind"] in ("vec", "towervec")
    S = Scorer(mdir, facts=facts, img=b16_image_vectors(set(facts)) if use_img else None,
               qvec_fn=b16_text if use_img else None, max_len=cfg.get("max_len", 256))
    if which == "eye":
        X = pd.read_parquet(D / "eye_sets.parquet")
        X = X.assign(q=X.question, qs=X.question.map(scorer_question), p=np.nan)
        X = X[["user", "item_id", "q", "qs", "p", "set", "eye", "weight", "query"]]
    elif which == "hold":
        X = pd.read_parquet(D / "pairs_train.parquet")
        X = X[X.user.map(train_folds()).isin(cfg["hold"])][["user", "item_id", "q", "qs", "p"]]
    else:
        X = pd.read_parquet(D / f"pairs_{which}.parquet")[["user", "item_id", "q", "qs", "p"]]
    X = X.copy(); X["item_id"] = X.item_id.astype(str)
    X = X.sort_values(["user", "qs", "item_id"]).reset_index(drop=True)
    X["pred"] = S.p(X.item_id.tolist(), X.qs.tolist(), bs=2048)
    X.to_parquet(mdir / f"pred_{which}{arg('name', '')}.parquet")
    print("PREDICT_DONE", len(X), flush=True)


if __name__ == "__main__":
    main()
