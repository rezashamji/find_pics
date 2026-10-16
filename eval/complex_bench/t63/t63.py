"""RESULTS 63: truth pass for complex_bench categories b, c, g on dev + TRAIN (rules: AUDIT_PROTOCOL.txt "ADDED 10-16").

Steps (main env, CPU):
  python eval/complex_bench/t63/t63.py units     # after D (direct/d27): unit table, agreed-strata eye draw, L unit list
  python eval/complex_bench/t63/t63.py sample    # after L (direct/l122): disputed-strata eye draw, sheets, key.json
  python eval/complex_bench/t63/t63.py score     # after labels: flip rule, estimates, corrected queries, re-score
Files: t63/units.parquet, t63/l_units.parquet, t63/sheets/*.jpg, t63/requests.txt (what the labeller reads),
       t63/key.json (NOT read while labelling), t63/labels_main.txt, t63/queries/{dev,train}.json, t63/report.txt
"""
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import ROOT, city, load_items  # noqa: E402
from direct import CATS, requests  # noqa: E402

T63 = ROOT / "t63"
TAKE = dict(FADD=30, FREM=30, KADD=15, KREM=15, A11=15, N1=15, N2=10, N3=10)
AGREED = ("A11", "N1", "N2", "N3")
DISPUTED = ("FADD", "KADD", "FREM", "KREM")
IN_NEW_BASE = {"A11", "KREM"}   # in the corrected truth regardless of flips; FREM in unless removed, FADD in if added


def load_direct(tag):
    fs = sorted((T63 / "direct" / tag).glob("*.parquet"))
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    D["item_id"] = D.item_id.astype(str); D["album"] = D.album.astype(int)
    return D.drop_duplicates(["qid", "album", "item_id"], keep="last")


def unit_table():
    Q = requests()
    D = load_direct("d27")
    users_done = {f.stem for f in (T63 / "direct" / "d27").glob("*.parquet")}
    rows = []
    for q in Q:
        if q["user"] not in users_done:
            raise SystemExit(f"D missing for {q['user']}")
        for k, al in enumerate(q["answers"][-1][: (2 if q["cat"] == "g" else 1)]):
            for i in al:
                rows.append((q["qid"], k, str(i)))
    Tset = pd.DataFrame(rows, columns=["qid", "album", "item_id"]).assign(T=1)
    U = D.merge(Tset, on=["qid", "album", "item_id"], how="outer")
    U["T"] = U["T"].fillna(0).astype(int)
    meta = pd.DataFrame([(q["qid"], q["cat"], q["split"], q["user"]) for q in Q], columns=["qid", "cat", "split", "user"])
    U = U.merge(meta, on="qid")
    U["pD"] = U.p.astype(float); U = U.drop(columns="p")
    # a truth photo the direct judge was never asked (person album: face not matched by direct.py) -> report, treat pD=nan
    U["D"] = (U.pD >= 0.5).astype(int)
    return U


def stratum(r, L):
    if r.T == 1 and r.D == 1:
        return "A11"
    if r.T == 0 and r.D == 0:
        return "N1" if r.pD >= 0.2 else "N2" if r.pD >= 0.05 else "N3"
    if np.isnan(L):
        return "X"
    if r.T == 0:
        return "FADD" if L == 1 else "KADD"
    return "FREM" if L == 0 else "KREM"


def draw(U, strata, rng, used):
    """Uniform draw within (cat, stratum), at most one unit per (qid, photo)."""
    out = []
    for cat in CATS:
        for s in strata:
            pool = U[(U.cat == cat) & (U.s == s)]
            ix = list(pool.index); rng.shuffle(ix)
            got = 0
            for j in ix:
                key = (U.at[j, "qid"], U.at[j, "item_id"])
                if key in used:
                    continue
                used.add(key); out.append(j); got += 1
                if got >= TAKE[s]:
                    break
    return out


def cmd_units():
    U = unit_table()
    nan = U.pD.isna().sum()
    U["s"] = [stratum(r, np.nan) for r in U.itertuples()]
    rng = random.Random(63)
    used = set()
    agreed_pick = draw(U, AGREED, rng, used)
    U["eye"] = 0; U.loc[agreed_pick, "eye"] = 1
    U.to_parquet(T63 / "units.parquet")
    L = U[(U.s == "X") | (U.eye == 1)][["qid", "album", "item_id"]]
    L.to_parquet(T63 / "l_units.parquet")
    lines = [f"units {len(U)}; truth units with no direct answer (pD nan) {nan}"]
    for (cat, sp), g in U.groupby(["cat", "split"]):
        lines.append(f"{cat} {sp}: units {len(g)}, T=1 {int(g['T'].sum())}, D=1 {int(g.D.sum())}, both {int(((g['T'] == 1) & (g.D == 1)).sum())}, "
                     f"T only {int(((g['T'] == 1) & (g.D == 0)).sum())}, D only {int(((g['T'] == 0) & (g.D == 1)).sum())}")
    lines.append(f"L units (disputes + agreed eye sample): {len(L)}")
    (T63 / "units.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def cmd_sample():
    from audit import sheet, tile   # noqa: E402
    U = pd.read_parquet(T63 / "units.parquet")
    Ld = load_direct("l122").rename(columns={"p": "pL"})
    U = U.merge(Ld, on=["qid", "album", "item_id"], how="left")
    U["L"] = np.where(U.pL.isna(), np.nan, (U.pL >= 0.5).astype(float))
    miss = int(((U.s == "X") & U.L.isna()).sum())
    if miss:
        raise SystemExit(f"{miss} disputed units have no L answer")
    U["s"] = [stratum(r, r.L) if r.s == "X" else r.s for r in U.itertuples()]
    rng = random.Random(6363)
    used = set(zip(U[U.eye == 1].qid, U[U.eye == 1].item_id))
    pick = draw(U, DISPUTED, rng, used)
    U.loc[pick, "eye"] = 1
    U.to_parquet(T63 / "units_l.parquet")
    E = U[U.eye == 1].copy()
    Q = {q["qid"]: q for q in requests()}
    it = load_items().set_index("item_id")
    E["key"] = E.qid + "|" + E.item_id
    order = list(E.index); random.Random(7).shuffle(order)
    # group by request so a request's photos sit together; requests in shuffled order
    qorder = list(dict.fromkeys(E.loc[order, "qid"]))
    E = E.set_index("qid", drop=False).loc[qorder].reset_index(drop=True)
    (T63 / "sheets").mkdir(exist_ok=True)
    key, req_lines, tiles = [], [], []
    faces = person_faces(E, Q)
    for n, r in enumerate(E.itertuples(), 1):
        q = Q[r.qid]
        p = it.loc[r.item_id]
        path = p.path
        a = q["program"][int(r.album)] if isinstance(q["program"][int(r.album)], dict) else {}
        if a.get("person"):   # the same red box every judge saw (identity = the bench's face match, not audited here)
            path = T63 / "boxed" / f"{r.item_id}_{a['person']}.jpg"
            path.parent.mkdir(exist_ok=True)
            faces[(a["person"], r.item_id)].save(path, quality=95)
        tiles.append(tile(path, str(n), f"{p.t:%Y-%m-%d %a %H:%M} | {city(p.place)}"))
        key.append(dict(n=n, qid=r.qid, cat=r.cat, split=r.split, album=int(r.album), item_id=r.item_id, stratum=r.s,
                        T=int(r.T), D=int(r.D), L=None if np.isnan(r.L) else int(r.L), pD=float(r.pD),
                        pL=None if np.isnan(r.pL) else float(r.pL)))
        prs = f" [{a['person']} = the person in the red box]" if a.get("person") else ""
        req_lines.append(f"{n:4d} sheet {(n - 1) // 6:03d} [{q['cat']}] (today {q['today']}) {' || '.join(q['turns'])}{prs}")
    for s in range(0, len(tiles), 6):
        sheet(tiles[s:s + 6]).save(T63 / "sheets" / f"s{s // 6:03d}.jpg", quality=92)
    (T63 / "key.json").write_text(json.dumps(key, indent=0))
    (T63 / "requests.txt").write_text("\n".join(req_lines) + "\n")
    print(len(key), "eye units,", len(tiles) // 6 + 1, "sheets")
    print(E.groupby(["cat", "s"]).size().unstack(fill_value=0))


def person_faces(E, Q):
    """{(name, item_id): boxed PIL image} for eye units of named-person albums (oracle.person_matches, as direct.py)."""
    need = defaultdict(set)
    for r in E.itertuples():
        a = Q[r.qid]["program"][int(r.album)]
        if isinstance(a, dict) and a.get("person"):
            need[Q[r.qid]["user"]].add((a["person"], r.item_id))
    if not need:
        return {}
    from findpics import store
    from findpics.engine import _boxed
    from oracle import person_matches
    from PIL import Image
    idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
    user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
    out = {}
    for u, want in need.items():
        sub = store.subset(idx, np.where(user_of == u)[0])
        ids = sub.items.item_id.astype(str).to_numpy()
        pm = person_matches(sub, json.loads((ROOT / "libraries" / f"{u}.json").read_text()))
        for nm, (prow, frow, _) in pm.items():
            for r_, f_ in zip(prow, frow):
                if (nm, ids[r_]) in want:
                    out[(nm, ids[r_])] = _boxed(sub, Image.open(sub.items.path.iloc[r_]).convert("RGB"), int(f_))
    return out


def read_labels():
    L = {}
    for line in open(T63 / "labels_main.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split(None, 2)
        L[int(p[0])] = p[1].strip().lower()
    return L


def is_match(lab, cat, album):
    if lab == "unsure":
        return None
    if cat == "g":
        return lab == f"album{album}" or lab == "both"   # "both": the photo fits both groups (g label added before key read)
    return lab == "match"


def estimate(N, M, in_set):
    """N: {stratum: size}, M: {stratum: (match, labelled)}, in_set: strata inside the version's set."""
    TP = FP = FN = 0.0
    for s, n in N.items():
        if s not in M or M[s][1] == 0:
            continue
        m = M[s][0] / M[s][1]
        if s in in_set:
            TP += n * m; FP += n * (1 - m)
        else:
            FN += n * m
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return P, R, TP, FP, FN


def boot(N, recs, in_set, B=2000, seed=0):
    """Stratified bootstrap over the labelled units within each stratum."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for s, m in recs:
        by[s].append(m)
    out = []
    for _ in range(B):
        M = {s: (int(np.sum(rng.choice(v, len(v)))), len(v)) for s, v in by.items()}
        P, R, *_ = estimate(N, M, in_set)
        out.append((P, R))
    a = np.array(out)
    return np.percentile(a[:, 0], [2.5, 97.5]), np.percentile(a[:, 1], [2.5, 97.5])


def cmd_score():
    U = pd.read_parquet(T63 / "units_l.parquet")
    key = json.loads((T63 / "key.json").read_text())
    lab = read_labels()
    missing = [k["n"] for k in key if k["n"] not in lab]
    if missing:
        raise SystemExit(f"unlabelled: {missing[:20]} ... ({len(missing)})")
    out = ["RESULTS 63: truth pass for b, c, g (dev + TRAIN pooled per category; rules AUDIT_PROTOCOL.txt ADDED 10-16)", ""]
    flips = {}
    verified = {}
    for cat in CATS:
        K = [k for k in key if k["cat"] == cat]
        recs = defaultdict(list)
        unsure = 0
        for k in K:
            m = is_match(lab[k["n"]], cat, k["album"])
            if m is None:
                unsure += 1; continue
            recs[k["stratum"]].append(int(m))
        N = U[U.cat == cat].groupby("s").size().to_dict()
        M = {s: (sum(v), len(v)) for s, v in recs.items()}
        out.append(f"== {cat}: units per stratum (eye match / labelled; {unsure} unsure excluded)")
        for s in ("A11", "FADD", "KADD", "FREM", "KREM", "N1", "N2", "N3"):
            mm = M.get(s, (0, 0))
            out.append(f"  {s:5s} N={N.get(s, 0):7d}  eye match {mm[0]}/{mm[1]}")
        fa = M.get("FADD", (0, 0)); fr = M.get("FREM", (0, 0))
        add_ok = fa[1] >= 10 and fa[0] / fa[1] >= 0.70
        rem_ok = fr[1] >= 10 and (fr[1] - fr[0]) / fr[1] >= 0.70
        flips[cat] = (add_ok, rem_ok)
        out.append(f"  FLIP RULE: add (eye agrees {fa[0]}/{fa[1]}) -> {'ADD' if add_ok else 'no'}; remove (eye agrees "
                   f"{fr[1] - fr[0]}/{fr[1]}) -> {'REMOVE' if rem_ok else 'no'}")
        new_in = set(IN_NEW_BASE) | ({"FADD"} if add_ok else set()) | (set() if rem_ok else {"FREM"})
        versions = [("T old truth", {"A11", "FREM", "KREM"}), ("D 27B direct", {"A11", "FADD", "KADD"}),
                    ("L 122B (on disputes)", {"A11", "FADD", "KREM"}), ("NEW corrected truth", new_in)]
        flat = [(s, m) for s, v in recs.items() for m in v]
        for name, ins in versions:
            P, R, TP, FP, FN = estimate(N, M, ins)
            (pl, ph), (rl, rh) = boot(N, flat, ins)
            f1 = 2 * P * R / (P + R) if P + R else 0
            out.append(f"  {name:22s} precision {P:.2f} [{pl:.2f}, {ph:.2f}]  recall {R:.2f} [{rl:.2f}, {rh:.2f}]  F1 {f1:.2f}  "
                       f"est right/wrong/missed {TP:.0f}/{FP:.0f}/{FN:.0f}")
            if name.startswith("NEW"):
                verified[cat] = P >= 0.80 and R >= 0.70
        # eye agreement with each opinion on the disputed sample
        agree = {o: [0, 0] for o in ("T", "D", "L")}
        for k in K:
            if k["stratum"] not in DISPUTED:
                continue
            m = is_match(lab[k["n"]], cat, k["album"])
            if m is None:
                continue
            for o in ("T", "D", "L"):
                agree[o][0] += int(k[o] == int(m)); agree[o][1] += 1
        out.append("  eye agreement on disputed units: " + ", ".join(f"{o} {a}/{n}" for o, (a, n) in agree.items()))
        out.append(f"  VERIFIED: {'YES' if verified[cat] else 'no (stays PROVISIONAL)'}")
        out.append("")
    # corrected queries
    (T63 / "queries").mkdir(exist_ok=True)
    U["new"] = U["T"]
    for cat, (add_ok, rem_ok) in flips.items():
        if add_ok:
            U.loc[(U.cat == cat) & (U.s == "FADD"), "new"] = 1
        if rem_ok:
            U.loc[(U.cat == cat) & (U.s == "FREM"), "new"] = 0
    NEW = {(r.qid, r.album): set() for r in U[["qid", "album"]].drop_duplicates().itertuples()}
    for r in U[U.new == 1].itertuples():
        NEW[(r.qid, int(r.album))].add(r.item_id)
    flipcount = []
    for sp in ("dev", "train"):
        F = json.loads((ROOT / "queries" / f"{sp}.json").read_text())
        keep, dropped, changed = [], [], 0
        added = removed = 0
        for q in F["queries"]:
            if q["cat"] in CATS:
                old = q["answers"][-1]
                na = []
                for k in range(len(old)):
                    s_new = NEW.get((q["qid"], k), set(map(str, old[k])))
                    o = set(map(str, old[k]))
                    added += len(s_new - o); removed += len(o - s_new)
                    na.append(sorted(s_new))
                if any(not a for a in na):
                    dropped.append(q["qid"]); continue
                changed += any(set(a) != set(map(str, b)) for a, b in zip(na, old))
                q = dict(q, answers=q["answers"][:-1] + [na], verified=verified.get(q["cat"], False))
            keep.append(q)
        F["queries"] = keep
        (T63 / "queries" / f"{sp}.json").write_text(json.dumps(F))
        flipcount.append(f"{sp}: requests changed {changed}, photos added {added}, removed {removed}, dropped (empty) {dropped}")
    out += ["CORRECTED TRUTH FILES t63/queries/{dev,train}.json:"] + flipcount
    for cat in CATS:
        for sp in ("dev", "train"):
            g = U[(U.cat == cat) & (U.split == sp)]
            a = int(((g.new == 1) & (g["T"] == 0)).sum()); r_ = int(((g.new == 0) & (g["T"] == 1)).sum())
            out.append(f"  {cat} {sp}: requests {g.qid.nunique()}, truth units {int(g['T'].sum())} -> {int(g.new.sum())} "
                       f"(+{a} / -{r_})")
    (T63 / "report.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"units": cmd_units, "sample": cmd_sample, "score": cmd_score}[sys.argv[1]]()
