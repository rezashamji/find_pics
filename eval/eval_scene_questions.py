"""SCENE QUESTION WORDING (MAC_INBOX M44 (b)): does another wording of the phone judge's question fix "food / beach /
sunset photos"? RESULTS 46 showed the judge answers about the SETTING (diners with incidental plates, food shops, pets
eating, beach-trip people with no beach in frame) or LOOK-ALIKES (rocky coast / lake shore as beach, golden-hour light
as sunset), and that no cut fixes beach (28 of its 230 P == 1.0 photos are wrong).

Judge = EXACTLY the pipeline of eval/recall_audit/scores_q3vl4b_real (eval_recall.py gen --model=models/q3vl4b_real
--tag=q3vl4b_real): Qwen3-VL-4B at the phone's real mlx-community 4-bit weights (models/q3vl4b_real), VLLMJudge
(engine._judge_rows: native DISBench pixels, ~500 px long side, nothing resized here), P(yes) = py/(py+pn).
Photos: ALL 7,886 photos of the 4 RESULTS 34 libraries for each of the 3 searches, every wording (q0 re-judged as a
pipeline check against the stored P). Plus every earlier eye-labelled photo of OTHER libraries for six "X photos"
searches (RESULTS 35 (2) labels: food, beach, sunset, flowers, cat, church) with q0 and the two GENERIC templates, to
test whether a template could be a planner rule for any X.

  gen    (vLLM env, GPU): python eval/eval_scene_questions.py gen --shard=k/K  -> eval/scene_questions/part{k}.parquet
  report (CPU):           python eval/eval_scene_questions.py report         -> eval/scene_questions/report.txt + p.tsv
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
OUT = Path("eval/scene_questions")
MODEL = os.path.abspath("models/q3vl4b_real")
REAL = Path("eval/recall_audit/scores_q3vl4b_real")
X = {"food photos": "food", "beach photos": "a beach", "sunset photos": "a sunset", "photos of flowers": "flowers",
     "photos of a cat": "a cat", "photos of a church": "a church"}
SCENE = ["food photos", "beach photos", "sunset photos"]
GENERIC = {"q0": "Is this a photo of {x}?",                              # the planner's subject form today
           "g_main": "Is {x} the main subject of this photo?",
           "g_vis": "Is {x} clearly visible in this photo?"}
SPECIFIC = {
    "food photos": {
        "f_meal": "Is food (a meal, dish or snack) the main subject of this photo?",
        "f_served": "Does this photo show plated or served food up close?",
        "x_people": "Are people the main subject of this photo?"},
    "beach photos": {
        "b_sandy": "Is this photo taken on a sandy beach, with the sand or shoreline visible?",
        "b_sandsea": "Is a beach (sand next to the sea) clearly visible in this photo?",
        "b_shore": "Is a beach (a sand or pebble shore by the sea or a lake) clearly visible in this photo?",
        "x_people": "Are people the main subject of this photo?"},
    "sunset photos": {
        "s_sunsky": "Is the sun setting or the sky lit by a sunset visible in this photo?",
        "s_sky": "Does the sky in this photo show a sunset (the sun low at the horizon, or the sky coloured orange or pink "
                 "by it)?"},
}
CUTS = [0.7, 0.99, 0.999, 0.9999]


def questions(q):
    out = {k: v.format(x=X[q]) for k, v in GENERIC.items()}
    out |= SPECIFIC.get(q, {})
    return out


def real_scores():
    sc = pd.concat([pd.read_parquet(f) for f in sorted(REAL.glob("part*.parquet"))], ignore_index=True)
    sc["item_id"] = sc.item_id.astype(str)
    return sc


def jobs():
    from eval_question_variants import labels
    sc = real_scores()
    in4 = set(sc.item_id)
    J = [(q, i, n, "audit4") for q in SCENE for i in sc[sc["query"] == q].item_id for n in questions(q)]
    held = sorted({(q, i) for (q, i) in labels() if q in X and i not in in4})
    J += [(q, i, n, "other") for q, i in held for n in GENERIC]
    return pd.DataFrame(J, columns=["query", "item_id", "qname", "set"])


def gen():
    """--shard=k/K, or --shard=a,b,c/K to run several shards in one job (model loaded once; a shard whose part file
    already exists is skipped; parts are written to a temp name then renamed, so duplicate runs are harmless)."""
    os.environ["FP_VLM_MODEL"] = MODEL
    from findpics import store
    from findpics.engine import _judge_rows
    from findpics.vlm import VLLMJudge
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    ks, K = shard.split("/"); K = int(K)
    ks = [int(k) for k in ks.split(",")]
    JA = jobs()
    di = store.load("data/public/index_disbench")
    drow = {u: i for i, u in enumerate(di.items.item_id.astype(str))}
    jud = None
    OUT.mkdir(parents=True, exist_ok=True)
    for k in ks:
        if (OUT / f"part{k}.parquet").exists():
            print(f"shard {k}/{K}: exists, skipped", flush=True)
            continue
        J = JA[JA.item_id.map(lambda s: int(s) % K == k if s.isdigit() else hash(s) % K == k)]
        miss = (~J.item_id.isin(drow)).sum()
        J = J[J.item_id.isin(drow)]
        print(f"shard {k}/{K}: {len(J)} calls ({miss} not in index)", flush=True)
        jud = jud or VLLMJudge(model=MODEL, gpu_mem=0.8)
        out = []
        for (q, n), g in J.groupby(["query", "qname"]):
            rows = np.array([drow[i] for i in g.item_id], int)
            qq = questions(q)[n]
            p = _judge_rows(di, jud, rows, np.full(len(rows), -1), qq)
            out.append(g.assign(question=qq, p=p))
            print(q, n, len(g), flush=True)
        tmp = OUT / f".part{k}.{os.getpid()}.tmp"
        pd.concat(out, ignore_index=True).to_parquet(tmp)
        os.replace(tmp, OUT / f"part{k}.parquet")


# ------------------------------------------------------------------------------------------------------------ report
def load_new():
    return pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("part*.parquet"))], ignore_index=True)


def rules(q, W):
    """name -> keep mask over W (index = item_id, columns = qnames). Single wordings at each cut; two-question form =
    the subject question at cut c AND the exclusion question below 0.5."""
    R = {}
    for n in questions(q):
        if n.startswith("x_"):
            continue
        for c in CUTS:
            R[f"{n}>={c}"] = W[n] >= c
    if "q0s" in W:                     # the STORED q0 P of RESULTS 46 (same pipeline, earlier run): run-to-run check
        R["q0s(stored)>=0.999"] = W.q0s >= 0.999
    if "x_people" in W:
        for base in [n for n in questions(q) if not n.startswith("x_")]:
            for c in CUTS:
                R[f"{base}>={c} & people<0.5"] = (W[base] >= c) & (W.x_people < 0.5)
    return R


def main_report():
    from eval_judge_cut_sweep import census_exact
    from eval_recall import _labels_eye, load_scores, old_labels, score_systems
    OUT.mkdir(parents=True, exist_ok=True)
    new = load_new()
    new["item_id"] = new.item_id.astype(str)
    a4 = new[new.set == "audit4"]
    df = load_scores(); df["item_id"] = df.item_id.astype(str)
    sc = real_scores()
    old, eye = old_labels(), _labels_eye()
    cex, nscreen, unl, n_new = census_exact(df, sc, old, eye)
    old2 = dict(cex); old2.update(old)
    ex = extra_labels()
    old3 = dict(old2); old3.update({k: v for k, v in ex.items() if k not in old3})
    chk = a4[a4.qname == "q0"].merge(sc[["query", "item_id", "p"]], on=["query", "item_id"], suffixes=("", "_stored"))
    L = ["SCENE QUESTION WORDING: does another wording fix 'food / beach / sunset photos'? (public DISBench; 2026-10-12)",
         "Code: eval/eval_scene_questions.py. Judge = Qwen3-VL-4B at the phone's REAL 4-bit weights (models/q3vl4b_real),",
         "the exact pipeline of eval/recall_audit/scores_q3vl4b_real (VLLMJudge + engine._judge_rows, native DISBench",
         "pixels ~500 px long side, P(yes) = py/(py+pn)). Every wording judged on ALL 7,886 photos of the 4 RESULTS 34",
         "libraries (142,674 judge calls incl. 726 on other libraries). Truth = RESULTS 34 estimator",
         "(eval_recall.score_systems: strata of the RESULTS 34 judge, uniform blind draws inside each, earlier labels exact).",
         "  V2 = + RESULTS 45 census as an exact stratum (= RESULTS 46 'V2').",
         f"  V3 = V2 + {len(ex)} NEW blind eye labels (labels_extra.txt): every photo kept by a wording's V2-best cut or by",
         "       q0>=0.999 that had no label, plus ~1/5 decoys (unlabelled, kept by none, q0 P in [0.05, 0.999)), mixed on",
         "       sheets, labelled before the key was read. MAIN = V3 (precision of every kept set below is then exact for",
         "       the kept photos except a handful; recall still rests on the RESULTS 34 strata).",
         "  'unsure' = not a match (as RESULTS 34/45/46); V3u = unsure counted as a match (sensitivity).",
         "All rules of one search are scored JOINTLY (one shared total of real photos; finer cells than RESULTS 46, so the",
         "total and the q0 numbers move a little vs RESULTS 46). 'best cut' = the cut in {0.7, 0.99, 0.999, 0.9999} with",
         "the highest V3 F1 (point estimates).",
         ""]
    L.append("PIPELINE CHECK (re-judged q0 vs stored scores_q3vl4b_real P; vLLM batch order differs, bf16 logits):")
    for q, g in chk.groupby("query"):
        same = {c: int(((g.p >= c) == (g.p_stored >= c)).sum()) for c in (0.7, 0.999)}
        L.append(f"  {q}: n={len(g)}, max |dP| {np.abs(g.p - g.p_stored).max():.2g}, keep decision same at 0.7: "
                 f"{same[0.7]}/{len(g)}, at 0.999: {same[0.999]}/{len(g)}")
    L.append("")
    tsv, best_rows = [], []
    for q in SCENE:
        W = a4[a4["query"] == q].pivot_table(index="item_id", columns="qname", values="p")
        W["q0s"] = sc[sc["query"] == q].set_index("item_id").p.reindex(W.index)
        d = df[df["query"] == q]
        assert set(W.index) == set(d.item_id), q
        tsv.append(W.assign(query=q).reset_index())
        R = rules(q, W)
        systems = {nm: {q: set(W.index[m.to_numpy()])} for nm, m in R.items()}
        res = {}
        for tag, o, um in (("V2", old2, False), ("V3", old3, False), ("V3u", old3, True)):
            r = score_systems(df, o, eye, systems, unsure_match=um)
            res[tag] = r[r["query"] == q].set_index("system")
        V = res["V3"]
        exact3 = {i for (qq, i) in old3 if qq == q}
        L.append(f"== {q}   real photos in the 7,886 (est., V3): {V.real_est.iloc[0]:.1f}  (V2 {res['V2'].real_est.iloc[0]:.1f})")
        for n, qq in questions(q).items():
            L.append(f"   {n:10s} '{qq}'")
        L.append("   | rule | kept | V3 precision [95% boot] | V3 recall [95% boot] [95% Bayes] | F1 | kept with exact label |"
                 " V2 P / R | V3u (unsure=match) P / R |")
        for nm in R:
            a, b, u = V.loc[nm], res["V2"].loc[nm], res["V3u"].loc[nm]
            f1 = 2 * a.precision_est * a.recall / max(a.precision_est + a.recall, 1e-9)
            kept_ids = systems[nm][q]
            nex = len(kept_ids & exact3)
            L.append(f"   | {nm:34s} | {a.returned_in_library:4d} | {a.precision_est:.2f} [{a.precision_boot_lo:.2f}-"
                     f"{a.precision_boot_hi:.2f}] | {a.recall:.3f} [{a.recall_boot_lo:.3f}-{a.recall_boot_hi:.3f}] "
                     f"[{a.recall_bayes_lo:.3f}-{a.recall_bayes_hi:.3f}] | {f1:.3f} | {nex} | "
                     f"{b.precision_est:.2f} / {b.recall:.3f} | {u.precision_est:.2f} / {u.recall:.3f} |")
            best_rows.append(dict(query=q, rule=nm, base=nm.split(">=")[0] + ("&people" if "people" in nm else ""),
                                  cut=float(nm.split(">=")[1].split()[0]), kept=a.returned_in_library, exact=nex,
                                  P=a.precision_est, P_lo=a.precision_boot_lo, P_hi=a.precision_boot_hi, R=a.recall,
                                  R_lo=a.recall_boot_lo, R_hi=a.recall_boot_hi, Rb_lo=a.recall_bayes_lo,
                                  Rb_hi=a.recall_bayes_hi, F1=f1, P2=b.precision_est, R2=b.recall,
                                  Pu=u.precision_est, Ru=u.recall, real=a.real_est))
        L.append("")
    B = pd.DataFrame(best_rows)
    B.to_csv(OUT / "rules.tsv", sep="\t", index=False, float_format="%.4g")
    L.append("BEST CUT PER WORDING (max V3 F1) vs the current question q0 at 0.999 (RESULTS 46 adopted):")
    for q in SCENE:
        bq = B[B["query"] == q]
        cur = bq[bq.rule == "q0>=0.999"].iloc[0]
        L.append(f"  {q}: CURRENT q0>=0.999: kept {cur.kept}, precision {cur.P:.2f} [{cur.P_lo:.2f}-{cur.P_hi:.2f}], "
                 f"recall {cur.R:.3f} [{cur.R_lo:.3f}-{cur.R_hi:.3f}] (Bayes {cur.Rb_lo:.3f}-{cur.Rb_hi:.3f}), F1 {cur.F1:.3f}")
        for base, g in bq.groupby("base", sort=False):
            r = g.loc[g.F1.idxmax()]
            L.append(f"    {base:16s} best cut {r.cut:<7g} kept {r.kept:4d}  precision {r.P:.2f} [{r.P_lo:.2f}-{r.P_hi:.2f}]  "
                     f"recall {r.R:.3f} [{r.R_lo:.3f}-{r.R_hi:.3f}] (Bayes {r.Rb_lo:.3f}-{r.Rb_hi:.3f})  F1 {r.F1:.3f}  "
                     f"exact {r.exact}/{r.kept}  unsure=match P/R {r.Pu:.2f}/{r.Ru:.3f}")
    L.append("")
    from eval_question_variants import labels
    lab = labels()
    oth = new[new.set == "other"].pivot_table(index=["query", "item_id"], columns="qname", values="p").reset_index()
    oth["lab"] = [lab.get((q, i)) for q, i in zip(oth["query"], oth.item_id)]
    L.append("OTHER LIBRARIES (RESULTS 35 (2) earlier eye labels; biased toward judge disagreements: compare rows within a")
    L.append("search, not with the tables above). right kept / right, wrong kept / wrong, unsure kept / unsure:")
    for q in X:
        d = oth[oth["query"] == q]
        n = d.lab.value_counts().to_dict()
        L.append(f"  {q} (right {n.get('right', 0)}, wrong {n.get('wrong', 0)}, unsure {n.get('unsure', 0)}):")
        for g in GENERIC:
            parts = []
            for c in CUTS:
                k = d[g] >= c
                cnt = {v: int((k & (d.lab == v)).sum()) for v in ("right", "wrong", "unsure")}
                parts.append(f"{c}: {cnt['right']}r/{cnt['wrong']}w/{cnt['unsure']}u")
            L.append(f"    {g:7s} " + "; ".join(parts))
    oth.to_csv(OUT / "other_libraries_p.tsv", sep="\t", index=False, float_format="%.7g")
    T = pd.concat(tsv, ignore_index=True)
    T.to_csv(OUT / "p.tsv", sep="\t", index=False, float_format="%.7g")
    txt = "\n".join(L)
    (OUT / "report_tables.txt").write_text(txt)
    print(txt)


def sheets():
    """V3 labels: every photo a wording keeps at its V2-best cut that has NO eye label yet (precision of a new wording
    otherwise rests on stratum-rate imputation), mixed with ~1/5 decoys (unlabelled photos no best-cut rule keeps, q0
    P in [0.05, 0.999)) so the labeller cannot tell kept from not. Native pixels (load_image max_side=896 = judge's
    shrink limit; DISBench ~500 px). Key (query, item_id, which rules) is written to key_extra.json: NOT read before
    labels_extra.txt is written. Labelled under eval/vector_enough/LABEL_PROTOCOL.txt."""
    import json
    from PIL import Image, ImageDraw, ImageFont
    from eval_judge_cut_sweep import census_exact
    from eval_recall import _labels_eye, load_scores, old_labels
    from findpics.media import load_image
    rng = np.random.default_rng(46)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        pass
    df = load_scores(); df["item_id"] = df.item_id.astype(str)
    sc = real_scores()
    old, eye = old_labels(), _labels_eye()
    cex, *_ = census_exact(df, sc, old, eye)
    known = set(old) | set(eye) | set(cex)
    B = pd.read_csv(OUT / "rules.tsv", sep="\t")
    new = load_new(); new["item_id"] = new.item_id.astype(str)
    path = sc.set_index(["query", "item_id"]).path
    items = []
    for q in SCENE:
        W = new[(new.set == "audit4") & (new["query"] == q)].pivot_table(index="item_id", columns="qname", values="p")
        R = rules(q, W)
        bq = B[B["query"] == q]
        best = [g.loc[g.F1.idxmax()].rule for b, g in bq.groupby("base") if "&" not in b and "stored" not in b]
        best += ["q0>=0.999"]                        # the current rule gets the same exact labels (fair comparison)
        kept = set().union(*[set(W.index[R[r].to_numpy()]) for r in best if r in R])
        need = sorted(i for i in kept if (q, i) not in known)
        pool = [i for i in W.index if i not in kept and (q, i) not in known and 0.05 <= W.q0[i] < 0.999]
        dec = list(rng.choice(pool, size=min(len(pool), max(len(need) // 4, 4)), replace=False)) if pool else []
        print(q, "unlabelled kept", len(need), "decoys", len(dec), flush=True)
        items += [(q, i) for i in need + dec]
    items = [items[j] for j in rng.permutation(len(items))]
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    key, s = [], 0
    ims = [(q, i, load_image(path[(q, i)], max_side=896)) for q, i in items]
    for q in SCENE:                    # one search per sheet (the labeller must know which criterion applies)
        for orient in ("land", "port"):
            its = [t for t in ims if t[0] == q and (t[2].width >= t[2].height) == (orient == "land")]
            cols, rows_ = (2, 3) if orient == "land" else (3, 2)
            for b in range(0, len(its), 6):
                chunk = its[b:b + 6]
                Wd = max(im.width for *_, im in chunk); Hd = max(im.height for *_, im in chunk)
                sheet = Image.new("RGB", (cols * Wd + (cols - 1) * 6, rows_ * Hd + (rows_ - 1) * 6), (255, 255, 255))
                d = ImageDraw.Draw(sheet)
                s += 1; name = f"{q.split()[0]}_{s:02d}.jpg"
                for j, (qq, i, im) in enumerate(chunk):
                    x, y = (j % cols) * (Wd + 6), (j // cols) * (Hd + 6)
                    sheet.paste(im, (x, y))
                    d.rectangle([x, y, x + 26, y + 28], fill=(255, 255, 0)); d.text((x + 6, y + 1), str(j + 1), fill=(0, 0, 0), font=font)
                    key.append(dict(sheet=name, number=j + 1, query=qq, item_id=i))
                sheet.save(OUT / "sheets" / name, quality=95)
    json.dump(key, open(OUT / "key_extra.json", "w"), indent=1)
    print(len(key), "photos on", s, "sheets")


def extra_labels():
    """Blind labels of variant-kept photos that had no label (eval/scene_questions/labels_extra.txt + key_extra.json)."""
    import json
    f, kf = OUT / "labels_extra.txt", OUT / "key_extra.json"
    if not f.exists() or not kf.exists():
        return {}
    key = {(r["sheet"], r["number"]): r for r in json.load(open(kf))}
    out = {}
    for line in open(f):
        if line.startswith("#") or not line.strip():
            continue
        s, n, v = line.split()[:3]
        r = key[(s, int(n))]
        out[(r["query"], str(r["item_id"]))] = v
    return out


if __name__ == "__main__":
    {"gen": gen, "report": main_report, "sheets": sheets}[sys.argv[1]]()
