"""RESULTS 49c: photos the FACT rule removes from a person album although the (fixed) truth has them: is the judge
(P < 0.7) or the 27B oracle (truth) right? Native-pixel sheets with the red box on the named person's best face, as the
judge saw it; no scores on the sheets (index.json keeps them).
Usage: python eval/complex_bench/c_diag/s6_rule_lost.py <old-rule scored dir> <new-rule scored dir> <run parts dir> <new-rule parts json>
  -> eval/complex_bench/c_diag/rule_lost/<qid>_NN.jpg + index.json (jpg not committed)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from findpics import store
from findpics.engine import _boxed
from findpics.people import expand_refs, item_person_scores

OUT = Path("eval/complex_bench/c_diag/rule_lost")
NAMES = ["Sam", "Alex"]


def main():
    old_dir, new_dir, parts, new_parts = map(Path, sys.argv[1:5])
    Q = {q["qid"]: q for q in json.load(open("eval/complex_bench/queries/dev.json"))["queries"]}
    R_old = {r["qid"]: r for f in sorted(parts.glob("part*.json")) for r in json.load(open(f))}
    R_new = {r["qid"]: r for r in json.load(open(new_parts))}
    so = {r["qid"]: r for r in json.load(open(old_dir / "scored.json"))}
    sn = {r["qid"]: r for r in json.load(open(new_dir / "scored.json"))}
    idx = store.load("data/public/index_disbench")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    rng = np.random.default_rng(4949)
    index = {}
    for qid in sorted(R_new):
        if Q[qid]["cat"] != "c" or sn[qid]["final"]["r"] >= so[qid]["final"]["r"]:
            continue
        a_old = set().union(*map(set, R_old[qid]["turns"][-1]["final"]["albums"]))
        a_new = set().union(*map(set, R_new[qid]["turns"][-1]["final"]["albums"]))
        truth = set().union(*map(set, Q[qid]["answers"][-1]))
        lost = sorted((a_old - a_new) & truth)
        show = list(rng.choice(lost, size=min(12, len(lost)), replace=False))
        u = qid.split(":")[0]
        sub = store.subset(idx, np.where(user_of == u)[0])
        lib = json.load(open(f"eval/complex_bench/libraries/{u}.json"))
        G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
        person = R_new[qid]["turns"][-1]["plan"]["albums"][0]["person"]
        g = G[NAMES.index(person)]
        best = item_person_scores(sub, expand_refs(sub, sub.face_emb[np.array(g["face_rows"], int)], rounds=3))[1]
        row = {iid: i for i, iid in enumerate(sub.items.item_id.astype(str))}
        ims = []
        for iid in show:
            k = row[iid]; im = Image.open(sub.items.path.iloc[k]).convert("RGB")
            ims.append(_boxed(sub, im, int(best[k])) if best[k] >= 0 else im)
        names = []
        for b in range(0, len(ims), 6):
            ch = ims[b:b + 6]; W = max(i.width for i in ch); H = max(i.height for i in ch)
            sh = Image.new("RGB", (2 * W + 6, 3 * H + 12), (255, 255, 255)); dr = ImageDraw.Draw(sh)
            for j, im in enumerate(ch):
                x, y = (j % 2) * (W + 6), (j // 2) * (H + 6)
                sh.paste(im, (x, y)); dr.rectangle([x, y, x + 24, y + 26], fill=(255, 255, 0))
                dr.text((x + 6, y + 1), str(j + 1), fill=(0, 0, 0), font=font)
            n = f"{qid.replace('@', '_').replace(':', '_')}_{b // 6 + 1:02d}.jpg"; sh.save(OUT / n, quality=95); names.append(n)
        index[qid] = dict(words=Q[qid]["turns"], question=R_new[qid]["turns"][-1]["plan"]["albums"][0]["judge_question"],
                          lost_truth=len(lost), shown=show, sheets=names)
        print(qid, Q[qid]["turns"][-1][:60], "lost truth photos", len(lost), names)
    json.dump(index, open(OUT / "index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
