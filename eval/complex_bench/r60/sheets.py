"""RESULTS 60: contact sheets of the photos the exemption moved, per changed dev request (native pixels; DISBench photos
are <= 500 px = what the judge sees): returned by the new run and not the old (ADDED) and the reverse (DROPPED), up to
n of each, SHUFFLED; tiles carry only a number + date; kind and truth membership go to key.json, read after labelling.
Sheets in .cache/tmp/r60_sheets (inside the project, never committed).
Usage: python eval/complex_bench/r60/sheets.py <old run dir> <new run dir> <qid> [<qid> ...] [n=9]"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, load_items  # noqa: E402

OUT = Path(".cache/tmp/r60_sheets")


def final(r):
    return set().union(*map(set, r["turns"][-1]["final"]["albums"])) if r["turns"] and r["turns"][-1]["final"] else set()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if "=" not in a]
    n = int(next((a.split("=")[1] for a in sys.argv if a.startswith("n=")), 9))
    old_d, new_d, qids = Path(args[0]), Path(args[1]), args[2:]
    Q = {q["qid"]: q for q in json.loads((ROOT / "queries" / "dev.json").read_text())["queries"]}
    load = lambda d: {r["qid"]: r for f in sorted(d.glob("part*.json")) for r in json.loads(f.read_text())}
    O, N = load(old_d), load(new_d)
    it = load_items().set_index("item_id")
    kf = OUT / "key.json"
    key = json.loads(kf.read_text()) if kf.exists() else {}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for qid in qids:
        q = Q[qid]
        gt = set().union(*map(set, q["answers"][-1]))
        go, gn = final(O[qid]), final(N[qid])
        rnd = random.Random(qid)
        add, drop = sorted(gn - go), sorted(go - gn)
        pick = [("ADDED", i) for i in rnd.sample(add, min(n, len(add)))] + [("DROPPED", i) for i in rnd.sample(drop, min(n, len(drop)))]
        rnd.shuffle(pick)
        name = qid.replace("@", "_").replace(":", "_")
        key[name] = dict(qid=qid, words=q["turns"], n_added=len(add), n_dropped=len(drop), n_old=len(go), n_new=len(gn),
                         n_truth=len(gt), added_in_truth=len(set(add) & gt), dropped_in_truth=len(set(drop) & gt),
                         tiles=[dict(n=k + 1, kind=kd, item_id=i, in_truth=i in gt) for k, (kd, i) in enumerate(pick)])
        for s in range(0, len(pick), 6):
            T = pick[s:s + 6]
            ims = []
            for k, (kd, i) in enumerate(T):
                im = Image.open(it.loc[i, "path"]).convert("RGB")
                ims.append((im, f"#{s + k + 1}  {str(it.loc[i].t)[:16]}"))
            W = 3 * 506
            rows = [ims[j:j + 3] for j in range(0, len(ims), 3)]
            H = sum(max(im.size[1] for im, _ in rw) + 26 for rw in rows) + 28
            S = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(S)
            dr.text((4, 4), f"{name}: {' || '.join(q['turns'])}"[:180], fill="black", font=font)
            y = 28
            for rw in rows:
                for c, (im, lab) in enumerate(rw):
                    dr.text((c * 506 + 2, y), lab, fill="blue", font=font)
                    S.paste(im, (c * 506, y + 22))
                y += max(im.size[1] for im, _ in rw) + 26
            f = OUT / f"{name}_{s // 6}.jpg"
            S.save(f, quality=95)
            print(f)
        print(qid, "old", len(go), "new", len(gn), "added", len(add), "dropped", len(drop), "truth", len(gt))
    kf.write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
