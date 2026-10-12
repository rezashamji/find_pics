"""Complex-query benchmark, step 4b: BLIND eye audit of the candidate answer sets (protocol: AUDIT_PROTOCOL.txt).

make : stratified sample of queries per category (test + dev), sheets of native 500 px photos with date/time/place
       under each photo, answers mixed with in-scope NON-answers (hard = highest image-text similarity to the owner's
       words; random = the rest) so the labeller cannot tell which is which. Key -> audit/key.json (never open it while
       labelling).
score: labels (audit/labels_*.txt) + key -> per-category precision of the candidate sets, miss rates among non-answers,
       estimated recall, anchor / known-item / two-album checks. -> audit/score.txt
Usage (main env): python eval/complex_bench/audit.py make [N_PER_CAT] | score
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from build import Lib  # noqa: E402
from common import ROOT, city, load_items  # noqa: E402

A = ROOT / "audit"
TILE = 500
N_ANS, N_HARD, N_RAND = 6, 4, 2


def font(sz=18):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def tile(path, label, caption):
    im = Image.open(path).convert("RGB")
    im.thumbnail((TILE, TILE))                       # native photos are <= 500 px: no resampling
    canvas = Image.new("RGB", (TILE, TILE + 46), "white")
    canvas.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 44, 30], fill=(255, 230, 0)); d.text((6, 4), label, fill="black", font=font(20))
    d.text((6, TILE + 4), caption[:60], fill="black", font=font(15)); d.text((6, TILE + 24), caption[60:120], fill="black", font=font(15))
    return canvas


def sheet(tiles, cols=3):
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * TILE + (cols - 1) * 6, rows * (TILE + 46) + (rows - 1) * 6), (90, 90, 90))
    for k, t in enumerate(tiles):
        S.paste(t, ((k % cols) * (TILE + 6), (k // cols) * (TILE + 46 + 6)))
    return S


def scope_mask(lib, a, cat):
    """Photos the album's metadata/anchor/person constraints allow (no visual condition)."""
    m = lib.meta_mask(a)
    anc = a.get("anchor")
    if isinstance(anc, dict) and anc.get("window"):
        rows = [lib.pos[str(i)] for i in anc.get("ids") or [] if str(i) in lib.pos]
        if rows:
            m &= lib.window(rows, anc["window"])
    if a.get("person"):
        m &= np.isin(lib.ids, list(lib.persons.get(a["person"], {})))
    return m


def make(n_per_cat):
    from findpics import store
    from findpics.models import ImageTextEncoder
    it = load_items()
    Q = []
    for s in ("test", "dev"):
        Q += json.loads((ROOT / "queries" / f"{s}.json").read_text())["queries"]
    idx = store.load("data/public/index_disbench")
    row_of = {str(i): k for k, i in enumerate(idx.items.item_id.astype(str))}
    enc = ImageTextEncoder(idx.clip_model, device="cpu")
    X = idx.clip.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    unit_item = idx.units.item_row.to_numpy()
    rng = random.Random(0)
    by_cat = defaultdict(list)
    for q in Q:
        by_cat[q["cat"]].append(q)
    A.mkdir(exist_ok=True); (A / "sheets").mkdir(exist_ok=True)
    key, todo = {}, []
    for c in sorted(by_cat):
        qs = by_cat[c][:]
        rng.shuffle(qs)
        # stratify by library: round-robin over users
        users = defaultdict(list)
        for q in qs:
            users[q["user"]].append(q)
        pick = []
        while len(pick) < n_per_cat and any(users.values()):
            for u in sorted(users):
                if users[u] and len(pick) < n_per_cat:
                    pick.append(users[u].pop())
        todo += pick
    libs = {}
    import time as _time
    TT = enc.texts([" ".join(q["turns"]) for q in todo]).astype(np.float32)
    TT /= np.linalg.norm(TT, axis=1, keepdims=True)
    print("texts encoded", flush=True)
    for qn, q in enumerate(todo):
        _t0 = _time.time()
        u = q["user"]
        if u not in libs:
            libs.clear(); libs[u] = Lib(u, it)
        lib = libs[u]
        progs = q["program"][-1] if q["cat"] == "h" else q["program"]
        fin = q["answers"][-1]
        words = " ".join(q["turns"])
        T = TT[qn]
        rows = np.array([row_of[i] for i in lib.ids])
        u_mask = np.isin(unit_item, rows)
        S = X[u_mask] @ T
        dfu = pd.DataFrame({"r": unit_item[u_mask], "s": S}).groupby("r").s.max()
        simv = dfu.reindex(rows).fillna(-1.0).to_numpy()
        ans = set().union(*map(set, fin))
        scope = np.zeros(len(lib.ids), bool)
        for a in progs:
            scope |= scope_mask(lib, a, q["cat"])
        neg = np.where(scope & ~np.isin(lib.ids, list(ans)))[0]
        neg_sorted = neg[np.argsort(-simv[neg])]
        hard_pool, rand_pool = neg_sorted[:30], neg_sorted[30:]
        items = []   # (item_id, role)
        if q["cat"] == "g":
            for k, s in enumerate(fin):
                for i in rng.sample(s, min(4, len(s))):
                    items.append((i, f"album{k}"))
        elif q["cat"] == "f":
            for i in fin[0][:3]:
                items.append((i, "answer"))
        else:
            for i in rng.sample(fin[0] if len(fin) == 1 else sorted(ans), min(N_ANS, len(ans))):
                items.append((i, "answer"))
        nh = N_HARD + (N_ANS - min(N_ANS, len(ans)) if q["cat"] not in ("g", "f") else 2)
        for r in rng.sample(list(hard_pool), min(nh, len(hard_pool))):
            items.append((lib.ids[r], "hard"))
        for r in rng.sample(list(rand_pool), min(N_RAND, len(rand_pool))):
            items.append((lib.ids[r], "random"))
        rng.shuffle(items)
        tiles = []
        anchor_tiles = []
        for a in progs:
            anc = a.get("anchor") if isinstance(a, dict) else None
            if isinstance(anc, dict) and anc.get("ids"):
                for i in [str(x) for x in anc["ids"][:2]]:
                    if i in lib.pos:
                        r = lib.pos[i]
                        anchor_tiles.append(tile(lib.L.path.iloc[r], "A", f"{lib.L.t.iloc[r]:%Y-%m-%d %a %H:%M} | {city(lib.L.place.iloc[r])}"))
        for k, (i, role) in enumerate(items, 1):
            r = lib.pos[i]
            tiles.append(tile(lib.L.path.iloc[r], str(k), f"{lib.L.t.iloc[r]:%Y-%m-%d %a %H:%M} | {city(lib.L.place.iloc[r])}"))
        ref_tiles = []
        for a in progs:
            nm = a.get("person") if isinstance(a, dict) else None
            if nm in ("Sam", "Alex") and not ref_tiles:
                meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
                G = [g for g in meta.get("face_groups", []) if g["n_items"] >= 25][:2]
                gi = ["Sam", "Alex"].index(nm)
                if gi < len(G):
                    from findpics.engine import reference_crop
                    sub = store.subset(idx, np.array([row_of[i] for i in lib.ids]))
                    for fr in [G[gi]["rep"]] + G[gi]["face_rows"][:2]:
                        im = reference_crop(sub, int(fr)).convert("RGB")
                        k_ = 300 / max(im.size)
                        im = im.resize((max(1, int(im.width * k_)), max(1, int(im.height * k_))), Image.LANCZOS)
                        c = Image.new("RGB", (TILE, TILE + 46), "white"); c.paste(im, ((TILE - im.width) // 2, 0))
                        d = ImageDraw.Draw(c); d.rectangle([0, 0, 44, 30], fill=(255, 120, 0))
                        d.text((6, 4), "R", fill="black", font=font(20)); d.text((6, TILE + 4), f"reference: {nm}", fill="black", font=font(15))
                        ref_tiles.append(c)
        name = f"q{qn:03d}_{q['cat']}"
        allt = ref_tiles + anchor_tiles + tiles
        for s in range(0, len(allt), 6):
            sheet(allt[s:s + 6]).save(A / "sheets" / f"{name}_{s // 6}.jpg", quality=92)
        key[name] = dict(qid=q["qid"], cat=q["cat"], user=u, turns=q["turns"], today=q["today"], n_anchor=len(anchor_tiles),
                         n_ref=len(ref_tiles),
                         items=[dict(n=k, item_id=i, role=role) for k, (i, role) in enumerate(items, 1)],
                         answer_sizes=[len(s) for s in fin], scope=int(scope.sum()), n_hard_pool=len(hard_pool),
                         n_rand_pool=len(rand_pool))
        print(name, q["qid"], len(items), f"{_time.time() - _t0:.1f}s", flush=True)
    (A / "key.json").write_text(json.dumps(key, indent=1))
    # the labeller's view: only the words (no program, no key)
    with open(A / "requests.txt", "w") as fh:
        for name, k in key.items():
            fh.write(f"{name} [{k['cat']}] (today {k['today']}) " + " || ".join(k["turns"]) + "\n")


def score():
    key = json.loads((A / "key.json").read_text())
    lab = {}
    for f in sorted(A.glob("labels_*.txt")):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(None, 3)
            if len(p) >= 3:
                lab[(p[0], p[1])] = (p[2], p[3].strip() if len(p) > 3 else "", f.name)
    out = []
    per = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # cat -> role -> [match, no, unsure]
    anchor_ok = defaultdict(lambda: [0, 0])
    est = defaultdict(lambda: dict(found=0.0, miss=0.0))
    qrows = []
    for name, k in key.items():
        c = k["cat"]
        for x in [dict(n=f"A{j}", role="anchor") for j in range(1, k["n_anchor"] + 1)] + k["items"]:
            L = lab.get((name, str(x["n"])))
            if L is None:
                continue
            l = L[0]
            if x["role"] == "anchor":
                anchor_ok[c][0 if l == "match" else 1] += 1
                continue
            if c == "g":
                good = (l == x["role"]) if x["role"].startswith("album") else (l in ("no_match", "neither"))
                per[c][x["role"]][0 if (l == x["role"] or (x["role"] in ("hard", "random") and l.startswith("album"))) else 2 if l == "unsure" else 1] += 1
                continue
            per[c][x["role"]][0 if l == "match" else 2 if l == "unsure" else 1] += 1
    lines = ["Blind eye audit of the candidate answer sets (labels vs key). match / no_match / unsure per role.",
             "answer = photo in the candidate set; hard = in-scope non-answer among the 30 most text-similar; random = other in-scope non-answer.", ""]
    for c in sorted(per):
        r = per[c]
        lines.append(f"[{c}] " + "  ".join(f"{role}: {v[0]} match / {v[1]} no / {v[2]} unsure" for role, v in sorted(r.items())))
        if c in anchor_ok:
            lines.append(f"     anchors: {anchor_ok[c][0]} right / {anchor_ok[c][0] + anchor_ok[c][1]}")
    (A / "score.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    make(int(sys.argv[2]) if len(sys.argv) > 2 else 25) if sys.argv[1] == "make" else score()
