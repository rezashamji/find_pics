"""Which labeled-positive Open Images photos does the 'real X' question reject that the plain question accepted?
(eval_real_subject: recall drops 1-8 points; OI labels often count toys/statues/drawings as the object.) Saves per-photo
scores and 2x2 sheets at 800 px of up to 8 such photos per concept for an eye check (public data; sheets gitignored)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from findpics import store
from findpics.engine import _judge_rows
from findpics.media import load_image
from findpics.vlm import VLLMJudge
REAL = "Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
OUT = Path("eval/real_subject")


def main():
    (OUT / "misses").mkdir(parents=True, exist_ok=True)
    idx = store.load("data/public/index_testlib"); row = {u: i for i, u in enumerate(idx.items.item_id)}
    gt = json.load(open("data/public/testlib/ground_truth.json")); J = VLLMJudge(gpu_mem=0.85)
    for c in ["Horse", "Sunglasses", "Wine glass", "Guitar", "Dog", "Cake"]:
        pos = np.array(sorted(row[u] for u in gt[f"concept:{c}"]["pos"] if u in row), int)
        plain = pd.read_parquet(f"eval/oracle/{c.lower().replace(' ', '_')}.parquet").set_index("item_row").p.reindex(pos).to_numpy()
        real = np.array(_judge_rows(idx, J, pos, np.full(len(pos), -1), REAL.format(x=c.lower()), batch=96))
        lost = pos[(plain >= 0.7) & (real < 0.7)]
        pd.DataFrame(dict(item_row=pos, plain=plain, real=real)).to_parquet(OUT / f"oi_{c.lower().replace(' ', '_')}.parquet")
        pick = list(np.random.default_rng(0).choice(lost, min(8, len(lost)), replace=False)) if len(lost) else []
        for s in range(0, len(pick), 4):
            S = Image.new("RGB", (1620, 1620), (30, 30, 30)); d = ImageDraw.Draw(S)
            for j, r in enumerate(pick[s:s + 4]):
                im = load_image(idx.items.path.iloc[int(r)], max_side=800); x, y = (j % 2) * 810, (j // 2) * 810
                S.paste(im, (x + (800 - im.width) // 2, y + (800 - im.height) // 2)); d.text((x + 5, y + 5), str(s + j + 1), fill=(255, 255, 0))
            S.save(OUT / "misses" / f"{c.lower().replace(' ', '_')}_{s // 4 + 1}.jpg", quality=85)
        print(c, "labeled positives", len(pos), "plain yes -> real no:", len(lost), flush=True)


if __name__ == "__main__":
    main()
