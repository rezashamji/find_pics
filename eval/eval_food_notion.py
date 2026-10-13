"""FOOD NOTION (after M62): what is food precision under the OWNER's idea of "food photos", and does a shop-front veto help?

Reza labelled 35 public audit4 photos ("Is this a food photo?"; eval/food_notion/reza_answers.tsv; hidden key
data/public/food_notion_35/key.json): 25 judge-yes photos the blind labellers called no_match/unsure, each tagged with a
failure kind F1-F7 of eval/vector_enough/false_positives_scene.txt, + 10 clear-food controls.

  notion (CPU): the owner's notion as a second column of food precision / recall in RESULTS 45 (1a), 46 and 47.
                Every labelled food photo TAGGED with a kind in false_positives_scene.txt whose label is not a match
                counts as w of a match, w = Reza's yes-rate for its kind (point; sensitivity: every kind at its 95%
                Clopper-Pearson low / high end). Untagged labels keep their label. Each table is re-scored exactly as it
                was made (same labels V1 / V2 / V3, same cut configuration, same systems scored jointly, same seed) via
                eval_recall.score_systems(weight=...), so the labellers' column reproduces the published numbers.
  gen   (vLLM env, GPU): shop-front veto wordings on every audit4 food photo with q0 P >= 0.99 (a superset of what the
                shipped rule "Is this a photo of food?" >= 0.9999 keeps) + the RESULTS 35 (2) other-library food photos.
                Judge = EXACTLY the pipeline of eval/eval_scene_questions.py gen (Qwen3-VL-4B at the phone's real 4-bit
                weights, models/q3vl4b_real; VLLMJudge + engine._judge_rows; native DISBench pixels; P = py/(py+pn)).
                -> eval/food_notion/veto_p.parquet
  veto  (CPU): keep = shipped food rule AND veto P >= cut; cut chosen on one half of the photos (item_id parity),
                reported on the other half, both ways (2-fold cross-fit) -> one out-of-fold keep set scored on the
                RESULTS 47 truth (V3) under both notions; sheets of every photo the veto drops (native pixels).
  -> eval/food_notion/notion_tables.txt, veto_tables.txt (report.txt is written by hand from these).
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/food_notion")
KEY = Path("data/public/food_notion_35/key.json")
FPS = Path("eval/vector_enough/false_positives_scene.txt")
SQ = Path("eval/scene_questions")
MODEL = os.path.abspath("models/q3vl4b_real")
Q = "food photos"
VETO = {   # keep the photo when P(yes) >= cut (both are "is there real food" questions, so yes = keep)
    "v_real": "Is real food or a drink visible in this photo (not just a sign, menu or shop front)?",
    "v_any": "Can you see any actual food or drink in this photo, even a small amount, rather than only a restaurant, "
             "shop front, sign, menu or food van?",
}
VETO_CUTS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
SHIP_CUT = 0.9999          # JudgeCutoff foodSubjectAccept (RESULTS 47)


# ------------------------------------------------------------------------------------------------------- the owner
def _cp(m, n):
    from scipy.stats import beta
    lo = 0.0 if m == 0 else beta.ppf(0.025, m, n - m + 1)
    hi = 1.0 if m == n else beta.ppf(0.975, m + 1, n - m)
    return lo, hi


def answers():
    """The 35 photos: key (set, eye label, kind, note) joined with Reza's yes/no."""
    a = pd.read_csv(OUT / "reza_answers.tsv", sep="\t", dtype=str)
    a["item_id"] = a.filename.str.replace(".jpg", "", regex=False)
    k = pd.DataFrame(json.load(open(KEY)))
    k["item_id"] = k.item_id.astype(str)
    d = k.merge(a[["item_id", "reza"]], on="item_id", validate="1:1")
    assert len(d) == 35
    return d


def owner_rates():
    """kind -> (yes, n, point, lo, hi) over the 25 disputed photos; 'control' too (not used as a weight)."""
    d = answers()
    out = {}
    for kd, g in d.groupby("kind"):
        m, n = int((g.reza == "yes").sum()), len(g)
        lo, hi = _cp(m, n)
        out[kd] = (m, n, m / n, lo, hi)
    return out


def kinds():
    """item_id -> (kind, eye label, label source, P, my call, note) for the 38 food photos of false_positives_scene.txt."""
    rx = re.compile(r"^\s+(\d+) \| ([\d.]+) \| (\w+)\s+\((\w+)\s*\) \| rank\s+(\d+) \| (F\d) \| (\w+)\s+\| (.*)$")
    out, on = {}, False
    for line in open(FPS):
        if line.startswith("== "):
            on = line.startswith("== food")
            continue
        m = rx.match(line) if on else None
        if m:
            out[m.group(1)] = dict(kind=m.group(6), eye=m.group(3), src=m.group(4), p=float(m.group(2)),
                                   call=m.group(7), note=m.group(8).strip())
    assert len(out) == 38, len(out)
    return out


def weights(which="point"):
    """{(food photos, item_id): w} for every tagged photo; which = point / lo / hi (all kinds at that end);
    'point+mine' = point + my_kind_tags.tsv for audit4 photos false_positives_scene.txt does not tag ('none' = 0)."""
    R, K = owner_rates(), kinds()
    j = {"point": 2, "lo": 3, "hi": 4, "point+mine": 2, "point+mine+rej": 2}[which]
    w = {(Q, i): float(R[v["kind"]][j]) for i, v in K.items()}
    if which.startswith("point+mine"):
        mt = dict(my_tags())
        if which == "point+mine+rej":
            mt.update(my_tags(MY_TAGS_REJ))
        w.update({(Q, i): float(R[k][j]) if k in R else 0.0 for (s, i), k in mt.items()
                  if s == "audit4" and i not in K})
    return w


# -------------------------------------------------------------------------------------------------------------- gen
def veto_photos():
    """(set, item_id): audit4 food photos with q0 or stored q0 P >= 0.99, + every other-library food photo."""
    p = pd.read_csv(SQ / "p.tsv", sep="\t", dtype={"item_id": str})
    f = p[(p["query"] == Q) & ((p.q0 >= 0.99) | (p.q0s >= 0.99))]
    o = pd.read_csv(SQ / "other_libraries_p.tsv", sep="\t", dtype={"item_id": str})
    o = o[o["query"] == Q]
    J = [("audit4", i) for i in f.item_id] + [("other", i) for i in o.item_id]
    ids35 = set(answers().item_id)
    assert ids35 <= set(f.item_id)
    return pd.DataFrame(J, columns=["set", "item_id"])


def gen():
    os.environ["FP_VLM_MODEL"] = MODEL
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    J = veto_photos()
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    miss = (~J.item_id.isin(drow)).sum()
    assert miss == 0, miss
    jud = VLLMJudge(model=MODEL, gpu_mem=0.8)
    rows = np.array([drow[i] for i in J.item_id], int)
    out = []
    for n, qq in VETO.items():
        p = _judge_rows(di, jud, rows, np.full(len(rows), -1), qq)
        out.append(J.assign(qname=n, question=qq, p=p))
        print(n, len(J), flush=True)
    # pipeline check: the shipped question re-judged on the same photos (vs p.tsv q0 / other_libraries_p.tsv q0)
    p = _judge_rows(di, jud, rows, np.full(len(rows), -1), "Is this a photo of food?")
    out.append(J.assign(qname="q0_check", question="Is this a photo of food?", p=p))
    tmp = OUT / f".veto_p.{os.getpid()}.tmp"
    pd.concat(out, ignore_index=True).to_parquet(tmp)
    os.replace(tmp, OUT / "veto_p.parquet")
    print("done", flush=True)


# ----------------------------------------------------------------------------------------------- notion: re-scoring
NOTIONS = [("lab", None), ("own", "point"), ("own_lo", "lo"), ("own_hi", "hi"), ("own_mine", "point+mine"),
           ("own_rej", "point+mine+rej")]


def _setcut(c):
    """The subject / food cut the census screens and judged sets were defined with when a table was made."""
    import score_apple_photos as sap
    sap.SUBJECT_CUT = c
    sap.FOOD_CUT = c


def _base():
    from eval_recall import _labels_eye, load_scores, old_labels
    df = load_scores()
    df["item_id"] = df.item_id.astype(str)
    return df, old_labels(), _labels_eye()


def _score_all(df, old, eye, systems):
    """Every notion, same systems / labels / seed -> {notion: food rows indexed by system}."""
    from eval_recall import score_systems
    out = {}
    for tag, which in NOTIONS:
        r = score_systems(df, old, eye, systems, weight=None if which is None else weights(which))
        out[tag] = r[r["query"] == Q].set_index("system")
        print(" scored", tag, flush=True)
    return out


def r45():
    """RESULTS 45 (1a): V1 labels, cut 0.99, vector top N / judged top N / fast first answer (phoneFastParams seed 0)."""
    _setcut(0.99)
    import eval_vector_enough as VE
    import tune_stop_time as TT
    df, old, eye = _base()
    _, r = TT.parse(VE.FAST)
    A = VE.audit4()
    systems, scr = {}, {}
    for q, d in A.items():
        ids, y = d.index.astype(str).to_numpy(), d.y.to_numpy()
        order, _ = TT.sim_timed(y, r, 0)
        yes_pos = order[y[order]]
        for N in VE.NS:
            systems.setdefault(f"vector top {N}", {})[q] = set(ids[:N])
            systems.setdefault(f"judged top {N}", {})[q] = set(ids[yes_pos[:N]])
        systems.setdefault("fast first answer (all)", {})[q] = set(ids[yes_pos])
        if q == Q:
            scr = {f"{k} {S}": list(s) for S in VE.SCREENS for k, s in (("vector", ids[:S]), ("judged", ids[y][:S]))}
    # (1b) first screens: census labels (truth_audit4 order), fractional under the owner's notion
    new, _ = VE.new_labels()
    st = df.set_index(["query", "item_id"]).stratum
    census = {}
    WN = {tag: (weights(w) if w else {}) for tag, w in NOTIONS}
    for nm, ids in scr.items():
        L = [VE.truth_audit4(Q, i, st[(Q, i)], old, eye, new) for i in ids]
        census[nm] = {tag: (sum(1.0 if v == "match" else WN[tag].get((Q, i), 0.0)
                                for v, i in zip(L, ids) if v is not None), sum(v is not None for v in L))
                      for tag, _ in NOTIONS}
    return _score_all(df, old, eye, systems), census


def r46():
    """RESULTS 46: V2 labels (census at cut 0.99), nested cuts + P bins on the stored real-weight P, all 6 searches."""
    _setcut(0.99)
    import eval_judge_cut_sweep as CS
    df, old, eye = _base()
    sc = CS.real_scores()
    P = {q: sc[sc["query"] == q].set_index("item_id").p for q in CS.QUERIES}
    cex, *_ = CS.census_exact(df, sc, old, eye)
    old2 = dict(cex); old2.update(old)
    systems = {}
    for q in CS.QUERIES:
        p = P[q]
        for name, c in CS.CUTS:
            systems.setdefault(f"cut {name}", {})[q] = set(p.index[CS.keep_at(p.to_numpy(), c)])
        for k in range(len(CS.CUTS), -1, -1):
            hi = CS.keep_at(p.to_numpy(), CS.CUTS[k][1]) if k < len(CS.CUTS) else np.zeros(len(p), bool)
            lo = CS.keep_at(p.to_numpy(), CS.CUTS[k - 1][1]) if k > 0 else np.ones(len(p), bool)
            systems.setdefault(f"bin {CS.bin_name(k)}", {})[q] = set(p.index[lo & ~hi])
    return _score_all(df, old2, eye, systems)


def v3_labels():
    """RESULTS 47 truth V3: census (judged screens at that day's cut 0.999) + earlier labels + labels_extra.txt."""
    _setcut(0.999)
    import eval_scene_questions as ESQ
    from eval_judge_cut_sweep import census_exact
    df, old, eye = _base()
    sc = ESQ.real_scores()
    cex, *_ = census_exact(df, sc, old, eye)
    old2 = dict(cex); old2.update(old)
    ex = ESQ.extra_labels()
    old3 = dict(old2); old3.update({k: v for k, v in ex.items() if k not in old3})
    return df, old3, eye


def food_W():
    """RESULTS 47 per-photo P of every food wording (exact parquet values, not the rounded p.tsv) + stored q0s."""
    import eval_scene_questions as ESQ
    new = ESQ.load_new()
    new["item_id"] = new.item_id.astype(str)
    a4 = new[(new.set == "audit4") & (new["query"] == Q)]
    W = a4.pivot_table(index="item_id", columns="qname", values="p")
    sc = ESQ.real_scores()
    W["q0s"] = sc[sc["query"] == Q].set_index("item_id").p.reindex(W.index)
    return W


def r47():
    """RESULTS 47 (1) food: V3 labels, every food rule of eval_scene_questions.rules, scored jointly (food only)."""
    import eval_scene_questions as ESQ
    df, old3, eye = v3_labels()
    W = food_W()
    R = ESQ.rules(Q, W)
    systems = {nm: {Q: set(W.index[m.to_numpy()])} for nm, m in R.items()}
    return _score_all(df, old3, eye, systems), (df, old3, eye, W)


def label_of(q, i, stratum, old, eye):
    """The label score_systems counts for one photo (kept stratum: earlier label first; else blind eye label first)."""
    if stratum == "kept" and (q, i) in old:
        return old[(q, i)]
    return eye.get((q, i)) or old.get((q, i))


def _row(tab, nm, show_ci=True):
    a = {t: tab[t].loc[nm] for t, _ in NOTIONS}
    L, O = a["lab"], a["own"]
    s = (f"{L.returned_in_library:4d} | {L.precision_est:.2f} [{L.precision_boot_lo:.2f}-{L.precision_boot_hi:.2f}] / "
         f"{L.recall:.3f} | {O.precision_est:.2f} [{O.precision_boot_lo:.2f}-{O.precision_boot_hi:.2f}] / {O.recall:.3f} | "
         f"{a['own_lo'].precision_est:.2f}-{a['own_hi'].precision_est:.2f} / "
         f"{min(a['own_lo'].recall, a['own_hi'].recall):.3f}-{max(a['own_lo'].recall, a['own_hi'].recall):.3f} | "
         f"{a['own_mine'].precision_est:.2f} / {a['own_mine'].recall:.3f} | "
         f"{a['own_rej'].precision_est:.2f} / {a['own_rej'].recall:.3f} | "
         f"{L.real_est:.1f} / {O.real_est:.1f} / {a['own_mine'].real_est:.1f} / {a['own_rej'].real_est:.1f}")
    return s


HDR = ("| rule | kept | LABELLERS precision [95% boot] / recall | OWNER (point) precision [95% boot] / recall | "
       "OWNER sensitivity (all kinds at CP low .. high end) precision / recall | OWNER + my tags (P >= 0.99) P / R | "
       "OWNER + my tags incl. judge-rejected P / R | real photos in the 7,886: labellers / owner / + my tags / + rejected |")


def notion():
    OUT.mkdir(parents=True, exist_ok=True)
    R = owner_rates()
    K = kinds()
    L = ["FOOD PRECISION UNDER THE OWNER'S NOTION (eval/eval_food_notion.py notion; public DISBench audit4; 2026-10-12)",
         "Owner's yes-rate per failure kind (25 disputed photos of eval/food_notion; 95% Clopper-Pearson):"]
    for kd in sorted(R):
        m, n, p, lo, hi = R[kd]
        ntag = sum(v["kind"] == kd for v in K.values())
        L.append(f"  {kd:8s} {m}/{n} = {p:.2f} [{lo:.3f}-{hi:.3f}]   tagged food photos in false_positives_scene.txt: {ntag}")
    L += ["Weight of a tagged photo whose label is not a match = its kind's rate; matches stay 1; untagged labels keep",
          "their label. 'sensitivity' = every kind at its CP low end / high end at once (F3, F6, F7 rest on ONE answer each,",
          "F4 on two, F5 on two: their CP ranges are ~0.02-1.0, so the low end is a floor, not a likely value).", ""]
    # ---- RESULTS 45 (1a) + (1b)
    T45, census = r45()
    L += ["RESULTS 45 (1a): audit4 as one 7,886-photo library, V1 labels (RESULTS 34 estimator), food cut 0.99, fast",
          "mode phoneFastParams seed 0 ('judged top N' = its first N judge-yes).", HDR]
    for nm in ["vector top 100", "vector top 200", "vector top 500", "judged top 100", "judged top 200",
               "fast first answer (all)"]:
        L.append(f"| {nm} | " + _row(T45, nm) + " |")
    L += ["RESULTS 45 (1b) first screen, census labels (matches / labelled photos; owner = fractional):"]
    for nm, c in census.items():
        L.append(f"  {nm:10s} labellers {c['lab'][0]:.0f}/{c['lab'][1]}   owner {c['own'][0]:.2f}/{c['own'][1]} "
                 f"(range {c['own_lo'][0]:.2f}-{c['own_hi'][0]:.2f})")
    L.append("")
    # ---- RESULTS 46
    T46 = r46()
    L += ["RESULTS 46 (2): V2 labels (V1 + RESULTS 45 census exact), cuts on the stored real-weight P(yes) of 'Is this a",
          "photo of food?' (all 6 searches scored jointly, as published).", HDR]
    import eval_judge_cut_sweep as CS
    for name, _ in CS.CUTS:
        L.append(f"| cut {name} | " + _row(T46, f"cut {name}") + " |")
    L.append("")
    # ---- RESULTS 47
    T47, (df, old3, eye, W) = r47()
    L += ["RESULTS 47 (1) FOOD: V3 labels (V2 with the census at that day's cut 0.999 + 401 blind labels_extra), the",
          "re-judged P of each wording, every food rule scored jointly (as published). SHIPPED = q0 >= 0.9999.", HDR]
    main = ["q0>=0.7", "q0>=0.99", "q0>=0.999", "q0>=0.9999", "g_main>=0.999", "f_meal>=0.99", "g_vis>=0.9999",
            "f_served>=0.7", "q0>=0.9999 & people<0.5", "g_main>=0.999 & people<0.5"]
    for nm in main:
        L.append(f"| {nm}{' (SHIPPED)' if nm == 'q0>=0.9999' else ''} | " + _row(T47, nm) + " |")
    f1 = {}
    for t, _ in NOTIONS:
        tb = T47[t].drop(index=[i for i in T47[t].index if "stored" in i])
        F = 2 * tb.precision_est * tb.recall / (tb.precision_est + tb.recall)
        f1[t] = (F.idxmax(), F.max())
    L.append("Best rule by F1 (all RESULTS 47 food rules): " + "; ".join(f"{t}: {nm} ({v:.3f})" for t, (nm, v) in f1.items()))
    for t in ("lab", "own", "own_mine", "own_rej"):
        tb = T47[t]
        F = 2 * tb.precision_est * tb.recall / (tb.precision_est + tb.recall)
        L.append(f"  F1 {t:4s}: " + ", ".join(f"{nm} {F[nm]:.3f}" for nm in main))
    L.append("")
    # ---- what the shipped rule keeps, by label and tag
    st = df[df["query"] == Q].set_index("item_id").stratum
    kept = W.index[(W.q0 >= SHIP_CUT).to_numpy()]
    rows = []
    for i in kept:
        v = label_of(Q, i, st[i], old3, eye)
        rows.append(dict(item_id=i, label=v, kind=K.get(i, {}).get("kind")))
    kd = pd.DataFrame(rows)
    nm_ = kd[kd.label.isin(["no_match", "unsure"])]
    L += [f"SHIPPED RULE (q0 >= 0.9999) keeps {len(kd)} photos: labels match {int((kd.label == 'match').sum())}, "
          f"no_match {int((kd.label == 'no_match').sum())}, unsure {int((kd.label == 'unsure').sum())}, "
          f"unlabelled {int(kd.label.isna().sum())}.",
          f"  not-a-match {len(nm_)}: tagged {int(nm_.kind.notna().sum())} ("
          + ", ".join(f"{k} {c}" for k, c in nm_.kind.value_counts().sort_index().items())
          + f"), UNTAGGED {int(nm_.kind.isna().sum())} (keep their label; listed below)."]
    for r in nm_[nm_.kind.isna()].itertuples():
        L.append(f"    untagged {r.label:8s} {r.item_id}  q0 P {W.q0[r.item_id]:.7f}  {_note(Q, r.item_id)}")
    pd.DataFrame(kd).to_csv(OUT / "shipped_kept_labels.tsv", sep="\t", index=False)
    txt = "\n".join(L)
    (OUT / "notion_tables.txt").write_text(txt)
    for t, tb in {"r45": T45, "r46": T46, "r47": T47}.items():
        pd.concat([v.assign(notion=k) for k, v in tb.items()]).to_csv(OUT / f"notion_{t}.tsv", sep="\t", float_format="%.5g")
    print(txt)


def _notes():
    """(query, item_id) -> the labeller's free-text note, from every blind-label file that has one."""
    import glob
    out = {}
    keyfiles = {"eval/recall_audit/labels.txt": sorted(glob.glob("eval/recall_audit/key*.json")),
                "eval/scene_questions/labels_extra.txt": ["eval/scene_questions/key_extra.json"]}
    for f in sorted(glob.glob("eval/vector_enough/labels_*.txt")):
        keyfiles[f] = ["eval/vector_enough/key.json"]
    for f, kfs in keyfiles.items():
        key = {}
        for kf in kfs:
            for r in json.load(open(kf)):
                sh = r["sheet"] if r["sheet"].endswith(".jpg") else r["sheet"] + ".jpg"
                key[(sh, int(r["number"]))] = r
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(None, 3)
            sh = parts[0] if parts[0].endswith(".jpg") else parts[0] + ".jpg"
            r = key.get((sh, int(parts[1])))
            if r is not None:
                out[(r["query"], str(r["item_id"]))] = parts[3].strip() if len(parts) > 3 else ""
    return out


_NOTES = None


def _note(q, i):
    global _NOTES
    if _NOTES is None:
        _NOTES = _notes()
    return _NOTES.get((q, i), "")


# -------------------------------------------------------------------------------------------------------------- veto
MY_TAGS = OUT / "my_kind_tags.tsv"     # kinds I (Claude) gave photos that false_positives_scene.txt does not tag


MY_TAGS_REJ = OUT / "my_kind_tags_rejected.tsv"   # ... and food-adjacent labels of photos the judge rejects (P < 0.99)


def my_tags(path=MY_TAGS):
    if not path.exists():
        return {}
    t = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    return {(r.set, r.item_id): r.kind for r in t.itertuples()}


def veto_table():
    """One row per judged photo: set, item_id, half, q0 (shipped question P), veto P per wording, labellers' value,
    owner's value (point / lo / hi), label, kind."""
    from eval_question_variants import labels as lab35
    V = pd.read_parquet(OUT / "veto_p.parquet")
    V["item_id"] = V.item_id.astype(str)
    T = V.pivot_table(index=["set", "item_id"], columns="qname", values="p").reset_index()
    df, old3, eye = v3_labels()
    W = food_W()
    st = df[df["query"] == Q].set_index("item_id").stratum
    o = pd.read_csv(SQ / "other_libraries_p.tsv", sep="\t", dtype={"item_id": str})
    o = o[o["query"] == Q].set_index("item_id")
    lab = lab35()
    K, R, MT = kinds(), owner_rates(), my_tags()
    rows = []
    for r in T.itertuples():
        if r.set == "audit4":
            q0 = float(W.q0[r.item_id])
            v = label_of(Q, r.item_id, st[r.item_id], old3, eye)
        else:
            q0 = float(o.q0[r.item_id])
            v = {"right": "match", "wrong": "no_match", "unsure": "unsure"}[lab[(Q, r.item_id)]]
        kd = K[r.item_id]["kind"] if (r.set == "audit4" and r.item_id in K) else None
        mk = MT.get((r.set, r.item_id))
        row = dict(set=r.set, item_id=r.item_id, half=int(r.item_id) % 2, q0=q0, label=v, kind=kd, my_kind=mk,
                   **{n: float(getattr(r, n)) for n in list(VETO) + ["q0_check"]})
        row["lab"] = None if v is None else float(v == "match")
        for t, j in (("own", 2), ("own_lo", 3), ("own_hi", 4)):
            k_ = kd or mk
            row[t] = None if v is None else (1.0 if v == "match" else (float(R[k_][j]) if k_ in R else 0.0))
        row["own_tagonly"] = None if v is None else (1.0 if v == "match" else (float(R[kd][2]) if kd else 0.0))
        rows.append(row)
    return pd.DataFrame(rows), (df, old3, eye, W)


def choose_cut(D, n, notion="own"):
    """On photos D (shipped-rule kept, labelled): the veto cut maximising (not-a-match dropped - match dropped) under
    the notion; 0 = no veto; ties -> the lowest cut (fewest drops)."""
    best = (0.0, 0.0)
    for c in VETO_CUTS:
        d = D[D[n] < c]
        net = float((1 - 2 * d[notion]).sum())
        if net > best[1] + 1e-9:
            best = (c, net)
    return best


def veto():
    from eval_recall import score_systems
    T, (df, old3, eye, W) = veto_table()
    T.to_csv(OUT / "veto_p.tsv", sep="\t", index=False, float_format="%.7g")
    ship = T[T.q0 >= SHIP_CUT].copy()
    L = ["SHOP-FRONT VETO for food searches (eval/eval_food_notion.py veto; public DISBench; 2026-10-12)",
         "Judge = Qwen3-VL-4B at the phone's real 4-bit weights, the pipeline of eval/eval_scene_questions.py gen.",
         "Veto wordings (keep when P(yes) >= cut):"] + [f"  {n}: '{q}'" for n, q in VETO.items()]
    a4 = T[T.set == "audit4"]
    same = int(((a4.q0_check >= SHIP_CUT) == (a4.q0 >= SHIP_CUT)).sum())
    L += [f"Pipeline check: 'Is this a photo of food?' re-judged in this job vs the RESULTS 47 P, keep at 0.9999 the same "
          f"for {same}/{len(a4)} audit4 photos; max |dP| {np.abs(a4.q0_check - a4.q0).max():.2g}.",
          f"Photos judged: {len(a4)} audit4 (q0 P >= 0.99), {int((T.set == 'other').sum())} other-library food photos.",
          f"Shipped rule (q0 >= 0.9999) keeps {int((ship.set == 'audit4').sum())} audit4 + {int((ship.set == 'other').sum())} "
          "other-library photos; the veto can only remove photos from these.",
          "Owner's value of a not-a-match label = its kind's yes-rate (false_positives_scene.txt tags; my_kind_tags.tsv for",
          "photos that file does not tag, see report); untagged = 0.", ""]
    # ---- veto P distribution by label / kind on the shipped set
    L.append("Veto P on the shipped set, by label (median; how many < 0.5):")
    for n in VETO:
        parts = []
        for nm, g in (("match", ship[ship.label == "match"]), ("not-a-match", ship[ship.label.isin(["no_match", "unsure"])]),
                      ("F2 (shopfront)", ship[(ship.kind == "F2") | (ship.my_kind == "F2")]),
                      ("unlabelled", ship[ship.label.isna()])):
            parts.append(f"{nm} {len(g)}: med {g[n].median():.3f}, <0.5 {int((g[n] < 0.5).sum())}")
        L.append(f"  {n}: " + "; ".join(parts))
    L.append("")
    # ---- 2-fold cut choice
    lab_ship = ship[ship.lab.notna()]
    oof = {}
    L.append("CUT CHOICE (2-fold by item_id parity; candidates 0 = no veto, " + ", ".join(map(str, VETO_CUTS)) + "):")
    L.append("  objective on the choosing half = owner's (not-a-match dropped - match dropped), labelled shipped photos of")
    L.append("  both sets; reported on the OTHER half.")
    for n in VETO:
        cuts = {}
        for h in (0, 1):
            c, net = choose_cut(lab_ship[lab_ship.half == h], n)
            cuts[1 - h] = c                       # cut chosen on half h applies to half 1-h
            L.append(f"  {n}: chosen on half {h}: cut {c} (net {net:+.2f} on that half)")
        oof[n] = cuts
    L.append("")
    # ---- held-out tallies
    def tally(g, n, cutmap):
        drop = g[g.apply(lambda r: r[n] < cutmap[r.half], axis=1)] if len(g) else g
        d_l = drop[drop.lab.notna()]
        return drop, dict(dropped=len(drop), dropped_unlabelled=int(drop.lab.isna().sum()),
                          lab_match_lost=float(d_l.lab.sum()), lab_wrong_removed=float((1 - d_l.lab).sum()),
                          own_match_lost=float(d_l.own.sum()), own_wrong_removed=float((1 - d_l.own).sum()),
                          own_lo_lost=float(d_l.own_lo.sum()), own_hi_lost=float(d_l.own_hi.sum()))
    dropped_all = {}
    L.append("HELD-OUT EFFECT (each photo judged by the cut chosen on the other half), shipped-rule photos:")
    L.append("| wording | set | kept by rule (labelled) | dropped (unlabelled) | labellers: real lost / wrong removed |"
             " owner: real lost / wrong removed | owner real lost, CP low..high |")
    for n in VETO:
        for s in ("audit4", "other"):
            g = ship[ship.set == s]
            drop, t = tally(g, n, oof[n])
            dropped_all.setdefault(n, []).append(drop)
            L.append(f"| {n} | {s} | {len(g)} ({int(g.lab.notna().sum())}) | {t['dropped']} ({t['dropped_unlabelled']}) | "
                     f"{t['lab_match_lost']:.0f} / {t['lab_wrong_removed']:.0f} | {t['own_match_lost']:.2f} / "
                     f"{t['own_wrong_removed']:.2f} | {t['own_lo_lost']:.2f}..{t['own_hi_lost']:.2f} |")
    L.append("")
    # ---- out-of-fold keep sets scored on the RESULTS 47 truth (V3), both notions
    import eval_scene_questions as ESQ
    keep_ship = set(W.index[(W.q0 >= SHIP_CUT).to_numpy()])
    # scored jointly with every RESULTS 47 food rule, so the shipped row matches the RESULTS 47 table's cells
    systems = {nm: {Q: set(W.index[m.to_numpy()])} for nm, m in ESQ.rules(Q, W).items()}
    systems = {"SHIPPED q0>=0.9999": systems.pop("q0>=0.9999"), "q0>=0.999": systems.pop("q0>=0.999"), **systems}
    n_r47 = len(systems)
    A4 = ship[ship.set == "audit4"].set_index("item_id")
    for n in VETO:
        cutv = A4.apply(lambda r: oof[n][r.half], axis=1)
        systems[f"SHIPPED & {n} (out-of-fold cut)"] = {Q: keep_ship - set(A4.index[(A4[n] < cutv).to_numpy()])}
        for c in (0.1, 0.5):
            systems[f"SHIPPED & {n}>={c}"] = {Q: keep_ship - set(A4.index[(A4[n] < c).to_numpy()])}
    # exploratory: a LOOSER food cut + the veto at 0.01 (the cut both halves chose for v_real on the shipped set; the
    # photos in [0.99, 0.9999) played no part in choosing it)
    Aall = T[T.set == "audit4"].set_index("item_id")
    for base in (0.999, 0.99):
        kb = set(W.index[(W.q0 >= base).to_numpy()])
        systems[f"q0>={base} & v_real>=0.01"] = {Q: kb - set(Aall.index[(Aall.v_real < 0.01).to_numpy()])}
    res = {}
    K = kinds()
    MT = {i: k for (s, i), k in my_tags().items() if s == "audit4"}
    for tag, which in NOTIONS:
        if which is None:
            w = None
        else:
            w = weights(which)
        r = score_systems(df, old3, eye, systems, weight=w)
        res[tag] = r[r["query"] == Q].set_index("system")
    L.append("AUDIT4 (one 7,886-photo library), RESULTS 47 truth V3; the veto rules scored jointly with all RESULTS 47 food")
    L.append("rules (so the shipped row matches RESULTS 47 up to the finer cells). precision [95% boot] / recall:")
    L.append("| rule | kept | LABELLERS P / R | OWNER (tags of false_positives_scene.txt) P / R | OWNER CP low..high P | "
             "OWNER + my tags P / R | OWNER + my tags incl. judge-rejected P / R |")
    for nm in [k for k in systems if k.startswith(("SHIPPED", "q0>=0.999")) or "v_real>=0.01" in k]:
        a, o, lo, hi, m, rj = (res[t].loc[nm] for t in ("lab", "own", "own_lo", "own_hi", "own_mine", "own_rej"))
        L.append(f"| {nm} | {a.returned_in_library} | {a.precision_est:.3f} [{a.precision_boot_lo:.2f}-{a.precision_boot_hi:.2f}]"
                 f" / {a.recall:.3f} | {o.precision_est:.3f} [{o.precision_boot_lo:.2f}-{o.precision_boot_hi:.2f}] / "
                 f"{o.recall:.3f} | {lo.precision_est:.3f}..{hi.precision_est:.3f} | {m.precision_est:.3f} / {m.recall:.3f} | "
                 f"{rj.precision_est:.3f} / {rj.recall:.3f} |")
    L.append("")
    # ---- other libraries
    L.append("OTHER LIBRARIES (RESULTS 35 (2) labels; biased toward judge disagreements): kept right / wrong / unsure")
    oth = ship[ship.set == "other"]
    L.append(f"  shipped rule alone: {int((oth.label == 'match').sum())} / {int((oth.label == 'no_match').sum())} / "
             f"{int((oth.label == 'unsure').sum())} (of all 50 labelled: 11 / 29 / 10)")
    for n in VETO:
        d = pd.concat(dropped_all[n])
        dd = d[d.set == "other"]
        k = oth[~oth.item_id.isin(dd.item_id)]
        L.append(f"  & {n} (out-of-fold): {int((k.label == 'match').sum())} / {int((k.label == 'no_match').sum())} / "
                 f"{int((k.label == 'unsure').sum())}")
    L.append("")
    # ---- every dropped photo
    L.append("EVERY PHOTO THE VETO DROPS (out-of-fold cut; shipped-rule photos), viewed at native pixels (sheets below):")
    drops = []
    for n in VETO:
        d = pd.concat(dropped_all[n])
        for r in d.sort_values(n).itertuples():
            kd = next((v for v in (r.kind, r.my_kind) if isinstance(v, str) and v), "-")
            src = "fp" if isinstance(r.kind, str) else ("mine" if kd != "-" else "")
            drops.append(dict(wording=n, set=r.set, item_id=r.item_id, veto_p=getattr(r, n), q0=r.q0, label=r.label,
                              kind=kd, kind_src=src, note=_note(Q, r.item_id) if r.set == "audit4" else ""))
            L.append(f"  {n:8s} {r.set:6s} {r.item_id:>12s} veto P {getattr(r, n):.4f}  q0 {r.q0:.7f}  label {str(r.label):8s} "
                     f"kind {kd:4s}({src:4s})  {drops[-1]['note']}")
    pd.DataFrame(drops).to_csv(OUT / "veto_drops.tsv", sep="\t", index=False, float_format="%.6g")
    # ---- F2 photos the veto keeps
    f2 = ship[(ship.kind == "F2") | (ship.my_kind == "F2")]
    L.append("")
    L.append("Shop-front (F2) photos in the shipped set and their veto P: " + "; ".join(
        f"{r.set}/{r.item_id} " + ", ".join(f"{n} {getattr(r, n):.3f}" for n in VETO) for r in f2.itertuples()))
    txt = "\n".join(L)
    (OUT / "veto_tables.txt").write_text(txt)
    print(txt)


def sheets(tsv="veto_drops.tsv", tag="drops", extra=None):
    """Native-pixel sheets (4 per sheet, 2 x 2) of photos in a TSV (set, item_id) -> .cache/tmp/food_notion_sheets/."""
    import glob
    from PIL import Image, ImageDraw, ImageFont
    d = pd.read_csv(OUT / tsv, sep="\t", dtype={"item_id": str}) if extra is None else extra
    ids = list(dict.fromkeys(d.item_id))
    outd = Path(".cache/tmp/food_notion_sheets"); outd.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    key = []
    for b in range(0, len(ids), 4):
        chunk = ids[b:b + 4]
        ims = [Image.open(glob.glob(f"data/public/raw/disbench/images/images/*/{i}.jpg")[0]).convert("RGB") for i in chunk]
        Wd = max(im.width for im in ims); Hd = max(im.height for im in ims)
        sh = Image.new("RGB", (2 * Wd + 6, 2 * Hd + 6), (255, 255, 255))
        dr = ImageDraw.Draw(sh)
        for j, (i, im) in enumerate(zip(chunk, ims)):
            x, y = (j % 2) * (Wd + 6), (j // 2) * (Hd + 6)
            sh.paste(im, (x, y))
            dr.rectangle([x, y, x + 26, y + 26], fill=(255, 255, 0)); dr.text((x + 7, y + 2), str(j + 1), fill=(0, 0, 0), font=font)
            key.append(f"{tag}_{b // 4 + 1:02d}.jpg {j + 1} {i} {im.width}x{im.height}")
        sh.save(outd / f"{tag}_{b // 4 + 1:02d}.jpg", quality=95)
    (outd / f"{tag}_key.txt").write_text("\n".join(key) + "\n")
    print("\n".join(key))


if __name__ == "__main__":
    {"gen": gen, "notion": notion, "veto": veto, "sheets": lambda: sheets(*sys.argv[2:])}[sys.argv[1]]()
