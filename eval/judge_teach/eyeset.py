"""RESULTS 58 step 0: every existing BLIND EYE LABEL that can score a photo judge, as one table of judge inputs.

One row = (photo, the product's question for that search, optional red-box face, eye label). Sources (all public
DISBench photos, labelled blind at native pixels; the files are read, never edited):
  everyday  eval_question_variants.labels()  (RESULTS 26-28/32 audits: the 261 right / 141 wrong + later everyday audits)
  recall    eval/recall_audit/labels.txt      (RESULTS 34 strata, 1,950 labels)
  census    eval/vector_enough/labels_*.txt   (RESULTS 45 census + ev16 draws)
  scene     eval/scene_questions/labels_extra (RESULTS 47 V3)
  person    eval/vector_first_person labels_*.txt + labels49c_*.txt (RESULTS 45 M45 / 49c; red box on the person)
  cb_b      eval/complex_bench/b55/labels.txt (RESULTS 55, dev b failures; question = the S1 dev plan's album question)
Question per everyday search = what the product asks TODAY (planner question + RESULTS 47 grounding: beach -> "clearly
visible"); cut = score_apple_photos.judge_cutoff (0.7 objects / 0.999 subjects / 0.9999 food). Person facts 0.7.
When two audits label the same (search, photo), the newer protocol wins (recall > census > scene > everyday) and the
disagreement is counted.
Library tag: 'heldout' = never usable for training (section 26's 8 libraries incl. audit4, all complex_bench dev/test);
'ev16' = the 16 libraries the DISBench training pool may come from (their labelled photos and near-duplicates are
excluded from training; eval on them is photo-disjoint only).
Usage (CPU): python eval/judge_teach/eyeset.py  -> eval/judge_teach/eyeset.parquet (+ counts printed)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
OUT = ROOT / "eval/judge_teach"
REAL = "Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
QUESTION = {"photos with a dog": REAL.format(x="dog"), "photos with a car": REAL.format(x="car"),
            "photos with a bicycle": REAL.format(x="bicycle"), "photos with a boat": REAL.format(x="boat"),
            "food photos": "Is this a photo of food?", "photos of a cat": "Is this a photo of a cat?",
            "beach photos": "Is a beach clearly visible in this photo?",     # RESULTS 47 grounding (shipped)
            "photos of flowers": "Is this a photo of flowers?", "sunset photos": "Is this a photo of a sunset?",
            "photos of a church": "Is this a photo of a church?", "photos taken at night": "Is this photo taken at night?",
            "selfies": "Is this a selfie?"}
KIND = {"photos with a dog": "object", "photos with a car": "object", "photos with a bicycle": "object",
        "photos with a boat": "object", "food photos": "subject", "photos of a cat": "subject", "beach photos": "subject",
        "photos of flowers": "subject", "sunset photos": "subject", "photos of a church": "subject",
        "photos taken at night": "fact", "selfies": "fact"}
SHORT = {"dog": "photos with a dog", "car": "photos with a car", "bicycle": "photos with a bicycle",
         "boat": "photos with a boat", "food": "food photos", "cat": "photos of a cat", "beach": "beach photos",
         "flowers": "photos of flowers", "sunset": "sunset photos", "church": "photos of a church",
         "night": "photos taken at night"}
NORM = {"right": "match", "wrong": "no_match", "unsure": "unsure", "match": "match", "no_match": "no_match"}


def library_tags():
    sp = json.load(open(ROOT / "eval/complex_bench/splits.json"))
    u26 = {r["user"] for f in glob.glob(str(ROOT / "eval/everyday_v2/part*.json")) for r in json.load(open(f))}
    tag = {u: "heldout" for u in sp["dev"] + sp["test"]} | {u: "heldout" for u in u26}
    tag |= {u: "ev16" for u in sp["train"] if u not in u26}
    split = {u: s for s in ("test", "dev", "train") for u in sp[s]}
    return tag, split


def main():
    from eval_question_variants import labels as everyday_labels
    from eval_recall import _labels_eye
    from score_apple_photos import judge_cutoff
    import eval_vector_enough as ve
    import eval_scene_questions as sq
    import eval_vector_first_person as vf
    from findpics import store
    idx = store.load(str(ROOT / "data/public/index_disbench"))
    items = idx.items.assign(item_id=idx.items.item_id.astype(str)).reset_index(drop=True)
    row_of = {i: k for k, i in enumerate(items.item_id)}
    user_of = items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    tag, split = library_tags()

    # ---- everyday-style searches: (query, item) -> label, newest protocol wins
    srcs = [("everyday", {k: NORM[v] for k, v in everyday_labels().items()}),
            ("scene", {k: NORM[v] for k, v in sq.extra_labels().items()}),
            ("census", {k: NORM[v] for k, v in ve.new_labels()[0].items()}),
            ("recall", {k: NORM[v] for k, v in _labels_eye().items()})]
    lab, src, conflicts = {}, {}, []
    for name, d in srcs:                     # later sources overwrite earlier ones
        for k, v in d.items():
            k = (k[0], str(k[1]))
            if k in lab and lab[k] != v:
                conflicts.append((k, src[k], lab[k], name, v))
            lab[k], src[k] = v, name
    rows = []
    for (q, i), v in lab.items():
        if q not in QUESTION or i not in row_of:
            continue
        r = row_of[i]
        rows.append(dict(set="everyday", kind=KIND[q], search=q, question=QUESTION[q], cut=judge_cutoff(QUESTION[q]),
                         item_id=i, item_row=r, face_row=-1, user=user_of[r], label=v, src=src[(q, i)]))
    # ---- person facts with the red box
    pl, _ = vf.eye_labels()
    k49 = json.load(open(ROOT / "eval/vector_first_person/key49c.json"))
    by49 = {(k["sheet"], k["number"]): k for k in k49}
    for f in sorted((ROOT / "eval/vector_first_person").glob("labels49c_*.txt")):
        for line in open(f):
            if not line.strip() or line.startswith("#"):
                continue
            sh, n, v = line.split()[:3]
            k = by49[(sh if sh.endswith(".jpg") else sh + ".jpg", int(n))]
            pl.setdefault((k["person"], k["cond"], k["item_id"]), (v, ""))
    fs = sorted(glob.glob(str(ROOT / "eval/vector_first_person/pp/*.parquet"))) + \
        sorted(glob.glob(str(ROOT / "eval/vector_first_person/part*.parquet")))
    box = pd.concat([pd.read_parquet(f, columns=["person", "item_id", "x1", "y1"]) for f in fs]) \
        .drop_duplicates(["person", "item_id"]).set_index(["person", "item_id"])
    F = idx.faces.reset_index(drop=True)
    face_at = {}
    for fr, (ir, x1, y1) in enumerate(zip(F.item_row.to_numpy(), F.x1.to_numpy(), F.y1.to_numpy())):
        face_at.setdefault(int(ir), []).append((fr, float(x1), float(y1)))
    miss = 0
    for (person, cond, i), (v, _) in pl.items():
        r = row_of[str(i)]
        bx = box.loc[(person, str(i))]
        cands = [fr for fr, x1, y1 in face_at.get(r, []) if abs(x1 - bx.x1) < 0.01 and abs(y1 - bx.y1) < 0.01]
        if not cands:
            miss += 1
            continue
        q = vf.COND[cond][1]
        rows.append(dict(set="person", kind="person_look" if cond == "tired" else "person_fact", search=f"person {cond}",
                         question=q, cut=0.7, item_id=str(i), item_row=r, face_row=int(cands[0]), user=user_of[r],
                         label=v, src="person", person=person))
    # ---- complex bench dev b (request-level labels; question = the shipped stack's album question)
    plans = json.load(open(ROOT / "eval/complex_bench/runs/m54/dev_s1_plan/plans.json"))["plans"]
    key = {}
    for line in open(ROOT / "eval/complex_bench/b55/key.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 4:            # the 12-photo b&w-colour side sheet has no request: not used
            continue
        sh, n, kind, i = f[:4]
        key[(sh, int(n))] = (kind, i)
    for line in open(ROOT / "eval/complex_bench/b55/labels.txt"):
        if line.startswith("#") or not line.strip():
            continue
        sh, n, v = line.split()[:3]
        if (sh, int(n)) not in key or v not in NORM:
            continue
        kind, i = key[(sh, int(n))]
        u, b = sh.rsplit("_", 1)
        qid = u.replace("_N", "@N") + ":" + b
        al = plans[qid]["turns"][-1]["plan"]["albums"]
        qs = [a.get("judge_question") for a in al if a.get("judge_question")]
        if not qs or str(i) not in row_of:
            continue
        r = row_of[str(i)]
        rows.append(dict(set="cb_b", kind="attribute", search=qid, question=qs[0], cut=0.7, item_id=str(i), item_row=r,
                         face_row=-1, user=user_of[r], label=NORM[v], src=f"b55:{kind}",
                         request=plans[qid]["turns"][-1]["message"]))
    D = pd.DataFrame(rows)
    D["lib"] = D.user.map(tag).fillna("other")
    D["split"] = D.user.map(split).fillna("none")
    D["path"] = items.path.to_numpy()[D.item_row.to_numpy()]
    D.to_parquet(OUT / "eyeset.parquet")
    print("rows", len(D), "| everyday conflicts (newer protocol kept)", len(conflicts), "| person rows without a face match", miss)
    print(D.groupby(["set", "kind", "label"]).size().unstack(fill_value=0).to_string())
    print(D.groupby(["set", "lib"]).size().unstack(fill_value=0).to_string())
    print(D.groupby(["search", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
