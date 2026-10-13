"""RESULTS 55: blind contact sheets for the dev b failures (native pixels; DISBench photos are <= 500 px = what the judge
sees). Per request: up to 6 returned-not-truth (FP), 6 truth-not-returned (FN) and 2 hits (TP) of the m54 dev S1 run,
SHUFFLED, each tile shows only a number + the photo's date and place; which tile is which goes to key.json, read only
after the labels are written. Sheets go to .cache/tmp/b55_sheets (inside the project, never committed).
Usage: python eval/complex_bench/b55/sheets.py <qid> [<qid> ...]
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, load_items  # noqa: E402

OUT = Path(".cache/tmp/b55_sheets")
RUN = ROOT / "runs" / "m54" / "m54_dev_s1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Q = {q["qid"]: q for q in json.loads((ROOT / "queries" / "dev.json").read_text())["queries"]}
    R = {r["qid"]: r for f in sorted(RUN.glob("part*.json")) for r in json.loads(f.read_text())}
    it = load_items().set_index("item_id")
    kf = OUT / "key.json"
    key = json.loads(kf.read_text()) if kf.exists() else {}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    n_fp = int(next((a.split("=")[1] for a in sys.argv if a.startswith("fp=")), 6))
    n_fn = int(next((a.split("=")[1] for a in sys.argv if a.startswith("fn=")), 6))
    for qid in [a for a in sys.argv[1:] if "=" not in a]:
        q, r = Q[qid], R[qid]
        got = set().union(*map(set, r["turns"][-1]["final"]["albums"]))
        gt = set().union(*map(set, q["answers"][-1]))
        rnd = random.Random(qid)
        fp, fn, tp = sorted(got - gt), sorted(gt - got), sorted(got & gt)
        pick = ([("FP", i) for i in rnd.sample(fp, min(n_fp, len(fp)))] + [("FN", i) for i in rnd.sample(fn, min(n_fn, len(fn)))]
                + [("TP", i) for i in rnd.sample(tp, min(2, len(tp)))])
        rnd.shuffle(pick)
        name = qid.replace("@", "_").replace(":", "_")
        key[name] = dict(qid=qid, words=q["turns"], tiles=[dict(n=k + 1, kind=kd, item_id=i) for k, (kd, i) in enumerate(pick)],
                         n_fp=len(fp), n_fn=len(fn), n_tp=len(tp))
        for s in range(0, len(pick), 6):
            T = pick[s:s + 6]
            ims = []
            for k, (kd, i) in enumerate(T):
                im = Image.open(it.loc[i, "path"]).convert("RGB")
                row = it.loc[i]
                lab = f"#{s + k + 1}  {str(row.t)[:16]}  {str(row.place or '')[-60:]}"
                ims.append((im, lab))
            W = 3 * 506
            rows = [ims[j:j + 3] for j in range(0, len(ims), 3)]
            H = sum(max(im.size[1] for im, _ in rw) + 26 for rw in rows) + 28
            S = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(S)
            dr.text((4, 4), f"{name}: {' || '.join(q['turns'])}", fill="black", font=font)
            y = 28
            for rw in rows:
                for c, (im, lab) in enumerate(rw):
                    dr.text((c * 506 + 2, y), lab[:62], fill="blue", font=font)
                    S.paste(im, (c * 506, y + 22))
                y += max(im.size[1] for im, _ in rw) + 26
            f = OUT / f"{name}_{s // 6}.jpg"
            S.save(f, quality=95)
            print(f)
    kf.write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
