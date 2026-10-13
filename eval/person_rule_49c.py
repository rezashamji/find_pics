"""RESULTS 49c (a): person albums, FACT rule (judge P >= 0.7 alone) vs the OLD rule (top half of the person's own photos
OR P >= 0.7) vs the LOOK rule (top half only), on eval/vector_first_person (32 DISBench people, 8 conditions, phone judge
Qwen3-VL-4B on EVERY photo of each person, red box on the person; eval/eval_vector_first_person.py gen).

Each rule keeps whole strata of a person's photos (rel = within-person rank of P, as engine.stream_album):
  K_top  P >= 0.7, rel > 0.5      kept by OLD, FACT, LOOK
  K_bot  P >= 0.7, rel <= 0.5     kept by OLD, FACT          (exists when > half the person's photos are judge-yes)
  D_hi   0.1 <= P < 0.7, rel > 0.5   kept by OLD, LOOK
  D_lov  P < 0.1, rel > 0.5, look vector in the condition's top 10% of D_lo   kept by OLD, LOOK
  D_lor  P < 0.1, rel > 0.5, the rest                                        kept by OLD, LOOK
  R_v    P < 0.7, rel <= 0.5, look vector in the top 10% of R                kept by none (recall denominator)
  R_r    the rest of R
The vector split only makes the sample efficient (real matches the judge missed sit high on the look vector: within-person
AUC snow 0.99 ... sunglasses 0.83, eval/vector_first_person/report.txt); every stratum is sampled uniformly and weighted by
its size. Eye labels: blind, native pixels, red box, LABEL_PROTOCOL.txt criteria; unsure = not a match.

  python eval/person_rule_49c.py sample   -> eval/vector_first_person/sheets49c/*.jpg + key49c.json (do NOT open the key
                                            before labels49c_<cond>.txt are written)
  python eval/person_rule_49c.py score    -> eval/vector_first_person/rule49c.txt
  python eval/person_rule_49c.py imdb     -> appends (b), the IMDB-WIKI eras (looks must be unchanged)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
sys.path.insert(0, "src")
OUT = Path("eval/vector_first_person")
ACC = 0.7
FACTS = ["beach", "eating", "hat", "outdoors", "smiling", "snow", "sunglasses"]   # "tired" is a look under both codes
N = {"K_top": 6, "K_bot": 6, "D_hi": 8, "D_lov": 8, "D_lor": 6, "R_v": 6, "R_r": 4}
KEEP = {"OLD": ["K_top", "K_bot", "D_hi", "D_lov", "D_lor"], "FACT": ["K_top", "K_bot"],
        "LOOK": ["K_top", "D_hi", "D_lov", "D_lor"]}
STRATA = list(N)
OLD_REV = "3b0fae3"   # the last commit before RESULTS 49c (old person rule)


def strata():
    from eval_vector_first_person import load
    from findpics.engine import Thresholds, person_keep
    D = load()
    D = D[D.cond.isin(FACTS)].copy()
    rel = np.zeros(len(D))
    for _, g in D.groupby(["person", "cond"]):
        p = g.p.to_numpy(); s = np.sort(p)
        rel[D.index.get_indexer(g.index)] = np.searchsorted(s, p, side="right") / len(p)
    D["rel"] = rel
    yes, top = D.p >= ACC, D.rel > 0.5
    st = np.where(yes & top, "K_top", np.where(yes, "K_bot", np.where(top, np.where(D.p >= 0.1, "D_hi", "D_lo"), "R")))
    D["st"] = st
    for c in FACTS:
        for base, hi, lo in (("D_lo", "D_lov", "D_lor"), ("R", "R_v", "R_r")):
            m = (D.cond == c) & (D.st == base)
            t = D.loc[m, "s"].quantile(0.9)
            D.loc[m, "st"] = np.where(D.loc[m, "s"] >= t, hi, lo)
    # the strata are exactly the product rule's sets (engine.person_keep): FACT keeps K_*, LOOK keeps rel > 0.5
    th = Thresholds()
    assert (person_keep(D.p, D.rel, "Is the person in the red box wearing a hat?", th) == D.st.str.startswith("K")).all()
    assert (person_keep(D.p, D.rel, "Does the person in the red box look tired?", th) ==
            D.st.isin(KEEP["LOOK"])).all()
    return D


def sample():
    from PIL import Image, ImageDraw, ImageFont
    from findpics.vlm import draw_box
    D = strata()
    rng = np.random.default_rng(4949)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    (OUT / "sheets49c").mkdir(parents=True, exist_ok=True)
    key = []
    for c in FACTS:
        its = []
        for s_, n in N.items():
            g = D[(D.cond == c) & (D.st == s_)]
            for _, r in g.iloc[rng.choice(len(g), size=min(n, len(g)), replace=False)].iterrows():
                its.append(r)
        its = [its[k] for k in rng.permutation(len(its))]
        ims = []
        for r in its:
            im = Image.open(r.path).convert("RGB")
            sx, sy = im.width / r.img_w, im.height / r.img_h
            ims.append((r, draw_box(im, (r.x1 * sx, r.y1 * sy, r.x2 * sx, r.y2 * sy))))
        s = 0
        for orient in ("land", "port"):
            o = [(r, im) for r, im in ims if (im.width >= im.height) == (orient == "land")]
            cols, rows = (2, 3) if orient == "land" else (3, 2)
            for b in range(0, len(o), 6):
                chunk = o[b:b + 6]
                W = max(im.width for _, im in chunk); H = max(im.height for _, im in chunk)
                sheet = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * H + (rows - 1) * 6), (255, 255, 255))
                dr = ImageDraw.Draw(sheet)
                s += 1; name = f"{c}_{s:02d}.jpg"
                for k, (r, im) in enumerate(chunk):
                    x, y = (k % cols) * (W + 6), (k // cols) * (H + 6)
                    sheet.paste(im, (x, y))
                    dr.rectangle([x, y, x + 24, y + 26], fill=(255, 255, 0))
                    dr.text((x + 6, y + 1), str(k + 1), fill=(0, 0, 0), font=font)
                    key.append(dict(sheet=name, number=k + 1, cond=c, person=r.person, item_id=r.item_id, path=r.path,
                                    stratum=r.st, p=float(r.p), rel=float(r.rel)))
                sheet.save(OUT / "sheets49c" / name, quality=95)
        print(c, len(its), "photos,", s, "sheets", flush=True)
    json.dump(key, open(OUT / "key49c.json", "w"), indent=1)
    sizes = D.groupby(["cond", "st"]).size().unstack(fill_value=0)
    sizes.to_csv(OUT / "strata49c.tsv", sep="\t")


def labels():
    key = json.load(open(OUT / "key49c.json"))
    by = {(k["sheet"], k["number"]): k for k in key}
    lab = {}
    for f in sorted(OUT.glob("labels49c_*.txt")):
        for line in open(f):
            if not line.strip() or line.startswith("#"):
                continue
            sh, n, v = line.split()[:3]
            assert v in ("match", "no_match", "unsure"), line
            k = by[(sh if sh.endswith(".jpg") else sh + ".jpg", int(n))]
            lab[(k["cond"], k["person"], k["item_id"])] = v
    return key, lab


def estimate(sizes, labs, keep, unsure_match=False):
    """Stratified estimate: matches in stratum h = N_h * (eye rate in its uniform sample)."""
    m = {h: sizes.get(h, 0) * (np.mean([(v == "match") or (unsure_match and v == "unsure") for v in labs[h]])
                               if labs.get(h) else 0.0) for h in STRATA}
    kept_n = sum(sizes.get(h, 0) for h in keep); kept_m = sum(m[h] for h in keep); tot_m = sum(m.values())
    p = kept_m / kept_n if kept_n else 0.0
    r = kept_m / tot_m if tot_m else 0.0
    return dict(n=kept_n, m=kept_m, p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0.0, tot=tot_m)


def score():
    key, lab = labels()
    sizes = pd.read_csv(OUT / "strata49c.tsv", sep="\t", index_col=0)
    rng = np.random.default_rng(0)
    L = ["RESULTS 49c (a): person-album rule on eval/vector_first_person, blind eye labels (eval/person_rule_49c.py).",
         "OLD = top half of the person's own photos OR P >= 0.7 (facts until 10-12); FACT = P >= 0.7 alone (new, facts);",
         "LOOK = top half only (looks, unchanged). Pooled over the people of each condition; photos = estimated from the",
         "stratified blind sample (unsure = not a match); [95% bootstrap over the labelled photos within strata].", ""]
    labs_all, rows = {}, []
    for c in FACTS:
        sz = sizes.loc[c].to_dict()
        labs = {h: [lab[(k["cond"], k["person"], k["item_id"])] for k in key
                    if k["cond"] == c and k["stratum"] == h and (k["cond"], k["person"], k["item_id"]) in lab] for h in STRATA}
        labs_all[c] = (sz, labs)
        cnt = " ".join(f"{h} {sum(v == 'match' for v in labs[h])}/{len(labs[h])}"
                       f"{'(u' + str(sum(v == 'unsure' for v in labs[h])) + ')' if any(v == 'unsure' for v in labs[h]) else ''}"
                       f" of {int(sz.get(h, 0))}" for h in STRATA if sz.get(h, 0))
        L.append(f"{c}: eye matches per stratum (labelled / stratum size): {cnt}")
    L.append("")

    def boot(c_list, rule, stat, unsure_match=False, B=2000):
        vals = []
        for _ in range(B):
            tot = {"n": 0, "m": 0.0, "tot": 0.0}
            for c in c_list:
                sz, labs = labs_all[c]
                lb = {h: list(rng.choice(v, size=len(v), replace=True)) if v else v for h, v in labs.items()}
                e = estimate(sz, lb, KEEP[rule], unsure_match)
                tot["n"] += e["n"]; tot["m"] += e["m"]; tot["tot"] += e["tot"]
            p = tot["m"] / tot["n"] if tot["n"] else 0; r = tot["m"] / tot["tot"] if tot["tot"] else 0
            vals.append(dict(p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0)[stat] if isinstance(stat, str) else stat(p, r))
        return np.percentile(vals, [2.5, 97.5])

    def pooled(c_list, rule, unsure_match=False):
        tot = {"n": 0, "m": 0.0, "tot": 0.0}
        for c in c_list:
            sz, labs = labs_all[c]
            e = estimate(sz, labs, KEEP[rule], unsure_match)
            tot["n"] += e["n"]; tot["m"] += e["m"]; tot["tot"] += e["tot"]
        p = tot["m"] / tot["n"] if tot["n"] else 0; r = tot["m"] / tot["tot"] if tot["tot"] else 0
        return dict(n=tot["n"], m=tot["m"], tot=tot["tot"], p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0)

    L.append("| condition | rule | photos kept | est. right (eye) | precision [CI] | recall [CI] | F1 [CI] |")
    L.append("|---|---|---|---|---|---|---|")
    for c_list, name in [([c], c) for c in FACTS] + [([c for c in FACTS if c != "smiling"], "5 facts + outdoors (no smiling)"),
                                                     (FACTS, "ALL 7")]:
        for rule in ("OLD", "FACT", "LOOK"):
            if rule == "LOOK" and name not in ("smiling",):
                continue
            e = pooled(c_list, rule)
            ci = {s: boot(c_list, rule, s, B=1000) for s in ("p", "r", "f1")}
            L.append(f"| {name} | {rule} | {e['n']:,} | {e['m']:.0f} of {e['tot']:.0f} | {e['p']:.3f} [{ci['p'][0]:.3f}, "
                     f"{ci['p'][1]:.3f}] | {e['r']:.3f} [{ci['r'][0]:.3f}, {ci['r'][1]:.3f}] | {e['f1']:.3f} "
                     f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}] |")
    L.append("")
    L.append("Sensitivity, unsure counted as a match (ALL 7): " + "; ".join(
        f"{rule} P {pooled(FACTS, rule, True)['p']:.3f} R {pooled(FACTS, rule, True)['r']:.3f} F1 {pooled(FACTS, rule, True)['f1']:.3f}"
        for rule in ("OLD", "FACT")))
    txt = "\n".join(L)
    (OUT / "rule49c.txt").write_text(txt + "\n")
    print(txt)


def imdb():
    """(b) RESULTS 10/18 IMDB-WIKI eras (Pratt / Hill / Rogen, face-filtered sim_expanded >= 0.40, cached 9B P for
    'heavier' and 'fit'): albums under the engine before 49c (OLD_REV: rule + make_exclusive) vs now, single and paired,
    for the cached questions and for the phrasings the planners write. Looks must be unchanged."""
    import importlib.util
    import re
    import subprocess
    from types import SimpleNamespace as NS
    from findpics import engine as E
    from findpics.planner import AlbumSpec
    src = subprocess.run(["git", "show", f"{OLD_REV}:src/findpics/engine.py"], capture_output=True, text=True,
                         check=True).stdout
    path = Path(f".cache/engine_{OLD_REV}_49c.py"); path.write_text(src)
    spec = importlib.util.spec_from_file_location("findpics.engine_head49c", path)
    O = importlib.util.module_from_spec(spec); sys.modules[spec.name] = O; spec.loader.exec_module(O)
    T = Path("data/public/transformation")
    s = pd.read_csv(T / "scored_fit.csv", index_col=0)
    fs = pd.read_csv(T / "face_sims.csv").set_index("key")
    s["sim"] = fs.loc[s.index, "sim_expanded"].to_numpy()
    s = s[s.sim >= 0.40].copy()
    s["item_id"] = s.index.astype(str)
    th = E.Thresholds()
    pairs = [("cached IMDB questions", "Does this person's face look heavier or fuller than average?",
              "Does this person look fit and lean in this photo?"),
             ("adapter planner (gate)", "Is the person in the red box looking heavier?", "Is the person in the red box looking fit?"),
             ("9B / base-4B planner (gate)", "Is this a photo of a person with a heavy build?",
              "Is this a photo of a person with a fit build?")]

    def albums(mod, keep_fn, qh, qf, paired):
        out = {}
        for name, g in s.groupby("name"):
            rs = []
            for alb, q, col in (("heavier", qh, "p_heavier"), ("fit", qf, "p_fit")):
                p = g[col].to_numpy(); srt = np.sort(p); rel = np.searchsorted(srt, p, side="right") / len(p)
                j = pd.DataFrame(dict(item_id=g.item_id, where="identity_match", p_attr=p, rel=rel, era=g.era.to_numpy()))
                j["y"] = keep_fn(p, rel, q)
                rs.append(NS(spec=AlbumSpec(name=alb, person=name, judge_question=q), judged=j,
                             returned=j[j.y].copy(), report=f"Album '{alb}': {int(j.y.sum())} items."))
            if paired:
                mod.make_exclusive(rs)
            for r in rs:
                out[(name, r.spec.name)] = sorted(r.returned.item_id)
        return out

    old_keep = lambda p, rel, q: (rel > th.rel_cut) | ((not O.LOOK_WORDS.search(q)) & (p >= th.judge_accept))  # OLD_REV
    new_keep = lambda p, rel, q: E.person_keep(p, rel, q, th)
    era = dict(zip(s.item_id, s.era))
    L = ["", "(b) IMDB-WIKI eras (RESULTS 10/18; 239 face-filtered crops: Pratt / Hill / Rogen; cached 9B P(heavier), P(fit)).",
         f"Albums under the rule + pairing before 49c ({OLD_REV}) vs now; counts = heavy-era / lean-era photos per album.",
         "| questions as written by | look? | albums | before: heavier H/L, fit L/H | now: heavier H/L, fit L/H | identical |",
         "|---|---|---|---|---|---|"]
    for who, qh, qf in pairs:
        for paired in (True, False):
            a = albums(O, old_keep, qh, qf, paired); b = albums(E, new_keep, qh, qf, paired)

            def cnt(x):
                h = [era[i] for k, v in x.items() if k[1] == "heavier" for i in v]
                f = [era[i] for k, v in x.items() if k[1] == "fit" for i in v]
                return f"{h.count('heavy')}/{h.count('lean')}, {f.count('lean')}/{f.count('heavy')}"
            L.append(f"| {who} | {E.is_look_question(qh)}/{E.is_look_question(qf)} | {'paired' if paired else 'single'} | "
                     f"{cnt(a)} | {cnt(b)} | {'YES' if a == b else 'NO'} |")
    txt = "\n".join(L)
    with open(OUT / "rule49c.txt", "a") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    {"sample": sample, "score": score, "imdb": imdb}[sys.argv[1]]()
